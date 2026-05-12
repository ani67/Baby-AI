"""UnifiedMindTrainer + GradNorm balancer for v2.0 (Wave-3 / Agent 7).

Joint end-to-end training of the unified differentiable mind:

    encoder -> memory_transformer -> affect_module -> decoder
                            |
                       memory_bank (trained_slots + experience_slots)

Five losses, balanced by GradNorm every GRADNORM_EVERY=200 steps
(spec § 4.7, CORRECTION 4):

    mask    : JEPA-style masked memory reconstruction (mse on recovered slots)
    lm      : next-token cross-entropy through the decoder
    align   : cross-modal alignment (mse between text/vision reps), 0 when single-modal
    affect  : predict-next-affect / consistency with surprise magnitude
    surp    : surprise binary classification

Between GradNorm updates, weights are held constant (detached) so the
training step is one combined backward — no per-loss autograd walk.

MPS rules (spec § 3) honored:
  - bfloat16 params/activations, float32 only for loss math
  - empty_cache guarded by `device.type == 'mps'`
  - no in-place ops on autograd-tracked tensors
  - shared params used for the GradNorm grad walk live in mt.layers[0]
"""
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from backend.unified_config import (
    BATCH_SIZE,
    GRAD_ACCUM,
    GRADNORM_EVERY,
    MAX_STEPS,
    VOCAB_SIZE,
    WARMUP_STEPS,
)

log = logging.getLogger('training')


@dataclass
class TrainingConfig:
    max_steps: int = MAX_STEPS
    batch_size: int = BATCH_SIZE
    grad_accum: int = GRAD_ACCUM
    warmup_steps: int = WARMUP_STEPS
    lr_mem: float = 1e-4
    lr_enc: float = 1e-4
    lr_dec: float = 1e-4
    lr_affect: float = 3e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    # initial weights: mask, lm, align, affect, surp
    w_init: tuple = (1.0, 1.0, 0.5, 0.3, 0.2)
    gradnorm_alpha: float = 1.5
    gradnorm_lr: float = 1e-2
    gradnorm_every: int = GRADNORM_EVERY
    checkpoint_every: int = 1000
    log_every: int = 50
    empty_cache_every: int = 50
    vocab_size: int = VOCAB_SIZE


class GradNorm(nn.Module):
    """Per-task loss-weight learner. Weights = exp(log_w).

    Updated every gradnorm_every steps via the explicit gradnorm_step() below
    (NOT through the regular backward of the joint loss — that pass uses
    detached weights as constants).
    """

    def __init__(self, n_tasks: int, alpha: float, init_weights: tuple):
        super().__init__()
        self.alpha = alpha
        self.log_w = nn.Parameter(torch.log(torch.tensor(init_weights, dtype=torch.float32)))
        self.register_buffer('initial_losses', torch.zeros(n_tasks, dtype=torch.float32))
        self.register_buffer('initialized', torch.tensor(False))

    @property
    def w(self) -> torch.Tensor:
        return torch.exp(self.log_w)

    @torch.no_grad()
    def init_losses(self, losses):
        if not bool(self.initialized):
            for i, l in enumerate(losses):
                self.initial_losses[i] = float(l)
            self.initialized.fill_(True)

    def gradnorm_step(self, losses, shared_params, gradnorm_opt):
        """Per-task grad-norm walk on a fresh forward.

        Multiplying `weights[i] * loss` constructs a graph node that depends on
        log_w (via weights = exp(log_w)). The autograd.grad call then yields
        per-task grads w.r.t. the shared parameters — the per-task grad-norm
        signal that GradNorm needs. Losing this structure (e.g. by calling
        `loss.backward()` instead) collapses the balancer.
        """
        gradnorm_opt.zero_grad()
        weights = self.w  # tracks log_w
        grad_norms = []
        for i, loss in enumerate(losses):
            scaled = weights[i] * loss
            grads = torch.autograd.grad(
                scaled, shared_params,
                retain_graph=True, allow_unused=True,
            )
            n2 = 0.0
            for g in grads:
                if g is not None:
                    n2 += float((g.detach().float() ** 2).sum())
            grad_norms.append(n2 ** 0.5)
        gn = torch.tensor(grad_norms, dtype=torch.float32, device=self.log_w.device)
        mean = gn.mean()
        loss_ratios = torch.stack([
            l.detach().float() / (self.initial_losses[i].to(l.device) + 1e-8)
            for i, l in enumerate(losses)
        ]).to(self.log_w.device)
        rel = loss_ratios / (loss_ratios.mean() + 1e-8)
        target = (mean * (rel ** self.alpha)).detach()
        gn_loss = F.l1_loss(gn, target)
        gn_loss.backward()
        gradnorm_opt.step()


class UnifiedMindTrainer:
    """Joint trainer for the 6 v2.0 components.

    Optimizers (separate per component, spec § 4.7):
        mem  -> memory_transformer parameters
        enc  -> encoder parameters
        aff  -> affect_module parameters
        dec  -> decoder parameters
        bank -> memory_bank.trained_slots + alpha_logit
                (experience_slots is a buffer, updated by soft_write, NOT
                touched by the optimizer)
    """

    TASKS = ('mask', 'lm', 'align', 'affect', 'surp')

    def __init__(
        self,
        cfg: TrainingConfig,
        memory_bank,
        memory_transformer,
        encoder,
        affect_module,
        decoder,
        device,
    ):
        self.cfg = cfg
        self.device = device
        self.bank = memory_bank
        self.mt = memory_transformer
        self.enc = encoder
        self.aff = affect_module
        self.dec = decoder
        for m in (memory_transformer, encoder, affect_module, decoder):
            m.to(device).to(torch.bfloat16)
        # memory_bank itself uses bfloat16 already; alpha_logit stays fp32

        self.opts = {
            'mem': AdamW(
                [p for p in memory_transformer.parameters() if p.requires_grad],
                lr=cfg.lr_mem, weight_decay=cfg.weight_decay,
            ),
            'enc': AdamW(
                [p for p in encoder.parameters() if p.requires_grad],
                lr=cfg.lr_enc, weight_decay=cfg.weight_decay,
            ),
            'aff': AdamW(
                affect_module.parameters(),
                lr=cfg.lr_affect, weight_decay=cfg.weight_decay,
            ),
            'dec': AdamW(
                decoder.parameters(),
                lr=cfg.lr_dec, weight_decay=cfg.weight_decay,
            ),
            'bank': AdamW(
                [memory_bank.trained_slots, memory_bank.alpha_logit],
                lr=cfg.lr_mem, weight_decay=0.0,
            ),
        }
        self.balancer = GradNorm(len(self.TASKS), cfg.gradnorm_alpha, cfg.w_init).to(device)
        self.gn_opt = AdamW(self.balancer.parameters(), lr=cfg.gradnorm_lr)
        self.step = 0
        self.best = float('inf')

    # ---------- losses ----------------------------------------------------

    def compute_losses(self, batch: dict):
        """Compute the five task losses for one example (or one micro-batch).

        Returns
        -------
        losses   : dict[str, Tensor] with keys 'mask', 'lm', 'align', 'affect', 'surp'
        enc_out  : the encoder forward dict (re-used by the caller)
        """
        device = self.device
        cfg = self.cfg
        text = batch.get('text', '')

        # 1) encode current input
        enc_out = self.enc(text=text, image=batch.get('image'))
        active_idx = enc_out['active_slot_indices']

        # 2) memory transformer over the FULL bank
        memory_full = self.mt(self.bank.slots)
        mem_active_xfm = memory_full[active_idx]

        # 3) affect from transformed memory
        aff_out = self.aff(memory_full)

        losses = {}

        # ---- L_mask: mask 15% of active memory, reconstruct via one MT layer
        #
        # We mask within the FULL memory tensor (not the active subset) because
        # mt.layers[0]'s sparse self-attn gathers via neighbor indices into the
        # full bank — running it on a sliced subset would index out of bounds.
        # The masked positions are chosen from the active set (where signal
        # lives) so the reconstruction target is meaningful.
        M_active = active_idx.shape[0]
        n_mask = max(1, int(M_active * 0.15))
        sel = torch.randperm(M_active, device=device)[:n_mask]
        mask_idx = active_idx[sel]  # indices into the FULL bank
        original = memory_full[mask_idx].detach().float()
        masked_full = memory_full.clone()
        masked_full[mask_idx] = 0.0
        recovered_full = self.mt.layers[0](
            masked_full, self.bank, aff_out['attention_bias'],
        )
        losses['mask'] = F.mse_loss(
            recovered_full[mask_idx].float(), original,
        )

        # ---- L_lm: next-token cross-entropy
        # The decoder cross-attends to (memory ⊕ just-encoded input).
        # Without concatenating updated_inputs the decoder generates from
        # the bank alone — no conditioning on what was just said — and LM
        # loss can never drop below log(vocab). Concatenation also makes
        # the encoder gradient-connected to the LM loss (its primary
        # training signal).
        if 'target_ids' in batch and batch['target_ids'] is not None:
            tgt = batch['target_ids'].to(device).long()  # (B, T)
            bos = torch.full(
                (tgt.shape[0], 1), 1, device=device, dtype=torch.long,
            )
            inp = torch.cat([bos, tgt[:, :-1]], dim=1)
            decoder_pool = torch.cat(
                [mem_active_xfm, enc_out['updated_inputs']],
                dim=0,
            )                                          # (K + L, D)
            logits, _ = self.dec(inp, decoder_pool)
            losses['lm'] = F.cross_entropy(
                logits.reshape(-1, cfg.vocab_size),
                tgt.reshape(-1),
                ignore_index=0,
            )
        else:
            losses['lm'] = torch.tensor(0.0, device=device)

        # ---- L_align: cross-modal alignment, only when both modalities present
        if batch.get('image') is not None and text:
            tr = self.enc(text=text)['input_rep']
            vr = self.enc(image=batch['image'])['input_rep']
            tr_n = F.normalize(tr.float(), dim=-1)
            vr_n = F.normalize(vr.float(), dim=-1)
            losses['align'] = F.mse_loss(tr_n, vr_n)
        else:
            losses['align'] = torch.tensor(0.0, device=device)

        # ---- L_affect: predicted next affect (if labels) or self-consistency
        if 'next_affect' in batch and batch['next_affect'] is not None:
            tgt_aff = batch['next_affect'].to(device).float().squeeze(0)
            losses['affect'] = F.mse_loss(
                aff_out['affect_vector'].float(), tgt_aff,
            )
        else:
            surp = enc_out['surprise'].float()
            mag = aff_out['affect_vector'].float().norm()
            losses['affect'] = F.mse_loss(mag.unsqueeze(0), surp.unsqueeze(0))

        # ---- L_surp: surprise binary classification
        if 'is_surprising' in batch and batch['is_surprising'] is not None:
            label = batch['is_surprising'].to(device).float().squeeze()
            pred = enc_out['surprise'].float().unsqueeze(0)
            losses['surp'] = F.binary_cross_entropy_with_logits(
                pred.expand_as(label.unsqueeze(0)), label.unsqueeze(0),
            )
        else:
            losses['surp'] = torch.tensor(0.0, device=device)

        return losses, enc_out

    # ---------- training step --------------------------------------------

    def train_step(self, batch: dict) -> dict:
        cfg = self.cfg
        device = self.device

        for opt in self.opts.values():
            opt.zero_grad(set_to_none=True)

        per_loss_running = {t: 0.0 for t in self.TASKS}
        total_running = 0.0

        for _ in range(cfg.grad_accum):
            losses, _ = self.compute_losses(batch)
            task_losses = [losses[t] for t in self.TASKS]
            self.balancer.init_losses(task_losses)

            # combined backward with current weights as constants (corr. 4)
            w = self.balancer.w.detach()
            total = sum(weight * l for weight, l in zip(w, task_losses))
            total = total / cfg.grad_accum
            total.backward()

            for t, l in zip(self.TASKS, task_losses):
                per_loss_running[t] += float(l.detach()) / cfg.grad_accum
            total_running += float(total.detach())

        # Per-component gradient-norm logging — measured BEFORE clipping
        # so we see the true pre-clip magnitude. clip_grad_norm_ rescales
        # everything by max_grad_norm / total_norm, which would otherwise
        # hide whether grads are actually present.
        grad_norms = {}
        for name, m in [
            ('mt', self.mt),
            ('enc', self.enc),
            ('aff', self.aff),
            ('dec', self.dec),
        ]:
            total = 0.0
            for p in m.parameters():
                if p.grad is not None:
                    total += float((p.grad.detach() ** 2).sum())
            grad_norms[name] = total ** 0.5
        bank_grads = []
        if self.bank.trained_slots.grad is not None:
            bank_grads.append(
                float((self.bank.trained_slots.grad ** 2).sum()))
        if self.bank.alpha_logit.grad is not None:
            bank_grads.append(
                float((self.bank.alpha_logit.grad ** 2).sum()))
        grad_norms['bank'] = sum(bank_grads) ** 0.5

        # clip across all trainable params (moved here after pre-clip logging)
        all_params = []
        for m in (self.mt, self.enc, self.aff, self.dec):
            all_params.extend(p for p in m.parameters() if p.requires_grad)
        all_params.extend([self.bank.trained_slots, self.bank.alpha_logit])
        torch.nn.utils.clip_grad_norm_(all_params, cfg.max_grad_norm)

        for opt in self.opts.values():
            opt.step()

        self.step += 1

        # GradNorm weight update every N steps (fresh forward, single shot)
        if self.step % cfg.gradnorm_every == 0:
            try:
                losses, _ = self.compute_losses(batch)
                task_losses = [losses[t] for t in self.TASKS]
                # shared params for the per-task grad walk: first MT layer
                shared = [p for p in self.mt.layers[0].parameters() if p.requires_grad]
                self.balancer.gradnorm_step(task_losses, shared, self.gn_opt)
            except Exception as e:
                log.warning(f"GradNorm update skipped at step {self.step}: {e}")

        if self.step % cfg.empty_cache_every == 0:
            if device.type == 'mps':
                torch.mps.empty_cache()

        return {
            'total_loss': total_running,
            **per_loss_running,
            'weights': self.balancer.w.detach().tolist(),
            'grad_norms': grad_norms,
        }

    # ---------- checkpoint ------------------------------------------------

    def save(self, mind_name: str, tag: str = ''):
        from backend.unified_config import checkpoint_dir, memory_bank_path
        d = Path(checkpoint_dir(mind_name))
        d.mkdir(parents=True, exist_ok=True)
        torch.save({
            'step': self.step,
            'mem': self.mt.state_dict(),
            'enc': self.enc.state_dict(),
            'aff': self.aff.state_dict(),
            'dec': self.dec.state_dict(),
            'balancer': self.balancer.state_dict(),
            'opts': {k: v.state_dict() for k, v in self.opts.items()},
            'best': self.best,
            'cfg': asdict(self.cfg),
        }, d / f'ckpt_{self.step:08d}{tag}.pt')
        self.bank.save(memory_bank_path(mind_name))

    def load(self, ckpt_path: str):
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.mt.load_state_dict(ckpt['mem'])
        self.enc.load_state_dict(ckpt['enc'])
        self.aff.load_state_dict(ckpt['aff'])
        self.dec.load_state_dict(ckpt['dec'])
        self.balancer.load_state_dict(ckpt['balancer'])
        for k, v in ckpt['opts'].items():
            if k in self.opts:
                self.opts[k].load_state_dict(v)
        self.step = ckpt.get('step', 0)
        self.best = ckpt.get('best', float('inf'))

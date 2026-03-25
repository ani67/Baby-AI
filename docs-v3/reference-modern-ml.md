# Reference: Modern ML Architecture Principles

*Extracted from Karpathy's microGPT, Zero-to-Hero, llm.c, minBPE, and related sources. Evaluated against Baby AI's architecture.*

---

## 1. Core Principles from microGPT

### 1.1 The Transformer Recipe

```
Input tokens → Embedding (token + position)
     → [Attention Block → MLP Block] × N layers
     → Output logits → Softmax → Sample

Each block has:
  Attention: Q/K/V projections → dot product → softmax → weighted sum
  MLP:       project up 4× → activation → project down
  Both wrapped in: residual connection + normalization
```

**Key insight**: Attention is a *communication* mechanism (tokens talk to each other). MLP is a *computation* mechanism (each token thinks independently). Both are needed.

### 1.2 Why Attention Works

```
Q = "what am I looking for?"
K = "what do I contain?"
V = "what do I offer?"

attention = softmax(Q · K^T / √d) · V

This lets every position selectively read from every other position.
The model LEARNS what to attend to via gradient descent on Q/K/V weights.
```

### 1.3 Residual Connections

```
x = x + block(x)

Not: x = block(x)

Why: gradients flow directly through the skip connection.
Deep networks train because gradients don't vanish through residuals.
Without them: training breaks above ~5 layers.
```

### 1.4 The Scaling Law

| Aspect | microGPT | Production GPT |
|--------|----------|----------------|
| Params | 4,192 | 1.6B+ |
| Layers | 1 | 96+ |
| Data | 32K names | Trillions tokens |
| Batch | 1 document | Millions tokens |
| Time | 1 minute | Months on 1000s GPUs |

Same algorithm, different scale. The math is identical.

---

## 2. Zero-to-Hero Curriculum Map

```
1. Micrograd        Backprop from scratch, computational graphs
2. Bigram Model     Simplest language model, torch basics
3. MLP              Multi-layer perceptron, train/val/test splits
4. Activations      Gradient flow, batch normalization, initialization
5. Backprop Ninja   Manual backward pass, no autograd
6. WaveNet          Hierarchical/tree architectures, dilated convolutions
7. GPT              Transformer, attention, autoregressive generation
8. Tokenizer        BPE, string→token pipeline
```

### Key Progression
- Start with scalars (micrograd) → tensors (MLP) → sequences (attention)
- Each step adds ONE concept, verifies it works, then builds on it
- Always build from scratch before using libraries

---

## 3. llm.c: Systems-Level Principles

### 3.1 No Framework Tax
- GPT-2 training in ~1000 lines of C
- 7% faster than PyTorch on same workloads
- Proves the math doesn't need the framework

### 3.2 Simplicity Budget
> "Reject 2% performance gains that require 500+ lines of complex code"

Three tiers:
1. **Root**: clean, readable (~1000 lines)
2. **Dev/cuda**: educational kernel library
3. **Integration**: cuBLAS/cuDNN for production

### 3.3 Validation by Reproduction
- Match PyTorch outputs bit-for-bit
- If C version ≠ Python version, C version is wrong
- Reference implementation is the test suite

---

## 4. Tokenization (minBPE)

### BPE Algorithm
```
1. Start: 256 byte-level tokens
2. Find most frequent adjacent pair
3. Merge into new token
4. Repeat until target vocab size

"aaabdaaabac" → "ZabdZabac" → "ZYdZYac" → "XdXac"
where Z=aa, Y=Zb, X=ZY
```

### Why It Matters
- Tokenization is preprocessing but affects everything downstream
- Rare words get split into subwords (model can still process them)
- Byte-level fallback means ANY input is tokenizable

---

## 5. Key Architectural Patterns

### 5.1 Mixture of Experts (MoE)
```
Input → Router (learned gating network)
  → selects top-K experts (out of N total)
  → only K experts compute, rest idle
  → weighted sum of expert outputs

Benefit: 8× params but same compute (sparse activation)
Used in: Mixtral, Switch Transformer, GPT-4 (rumored)
```

### 5.2 Grouped Query Attention (GQA)
```
Standard:  H heads, H key/value sets
GQA:       H heads, G key/value sets (G < H)
           Multiple query heads share one K/V group

Benefit: reduces KV-cache memory by H/G factor
Used in: Llama 2/3, Gemma
```

### 5.3 RoPE (Rotary Position Embedding)
```
Encode position by rotating Q/K vectors in 2D subspaces.
Relative position = angle difference between rotations.
Naturally handles variable-length sequences.
```

### 5.4 RMSNorm (vs LayerNorm)
```
LayerNorm: normalize to zero mean + unit variance, then scale/shift
RMSNorm:   normalize to unit RMS only (no mean centering)

Simpler, fewer params, empirically equivalent. Used in Llama.
```

### 5.5 Multimodal (VQVAE / Diffusion)
```
VQVAE: image → encoder → nearest codebook vector → decoder → image
  Discrete latent space — images become sequences of codebook indices
  Can be fed to transformers as "visual tokens"

Diffusion: learn to reverse a noise process
  Forward: image → progressively add noise → pure noise
  Reverse: noise → progressively denoise → image
  Transformer variant: DiT (Diffusion Transformer)
```

---

## 6. Optimization Essentials

### Adam Optimizer
```
m = β₁·m + (1-β₁)·grad           momentum (smoothed gradient)
v = β₂·v + (1-β₂)·grad²          adaptive learning rate per param
param -= lr · m / (√v + ε)        update

β₁=0.9, β₂=0.999, ε=1e-8 (standard)
AdamW: adds weight decay BEFORE the Adam step (not L2 regularization)
```

### Precision Hierarchy
```
fp32:   full precision, 4 bytes    — training reference
fp16:   half precision, 2 bytes    — fast but overflow risk
bf16:   brain float, 2 bytes       — same range as fp32, less precision
fp8:    quarter precision, 1 byte  — inference only (usually)

Mixed precision: forward in bf16, accumulate gradients in fp32
```

### Distributed Training
```
DDP:   Data Distributed Parallel — same model on each GPU, average gradients
ZeRO:  Partition optimizer state/gradients/params across GPUs
       Stage 1: partition optimizer state
       Stage 2: + partition gradients
       Stage 3: + partition parameters (full model sharding)
```

---

## 7. Inference Optimization

### KV-Cache
```
Without cache: recompute all K/V for all positions each token
With cache:    store K/V from previous tokens, only compute new position

Memory: O(batch × layers × heads × seq_len × head_dim)
Speedup: O(n) per token instead of O(n²) for full recomputation
```

### Quantization
```
int8:  map fp32 weights to 8-bit integers with scale factor
int4:  even more compressed, quality degrades
GPTQ:  layer-by-layer quantization minimizing reconstruction error
AWQ:   activation-aware quantization (keep important weights precise)
```

---

## 8. Post-Training

### SFT (Supervised Fine-Tuning)
```
Base model: predicts next token (internet text patterns)
SFT model:  trained on curated (prompt, response) pairs
LoRA:       freeze base weights, add small trainable matrices
            Original: W·x, LoRA: (W + A·B)·x where A,B are small
```

### RLHF / DPO
```
RLHF: train reward model on human preferences → PPO to maximize reward
DPO:  skip reward model, directly optimize on preference pairs
      Simpler, more stable, increasingly preferred
```

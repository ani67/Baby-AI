#!/usr/bin/env python3
"""Prepare v2.0 training data from existing curriculum corpus.

Reads data/{mind}/surprised_sentences.jsonl (one record per line, with affect
vector). Produces:
  - data/{mind}/v2_train.jsonl
  - data/{mind}/v2_val.jsonl
  - data/{mind}/v2_test.jsonl
  - data/{mind}/v2_vocab.json (BPE tokenizer, 16384 tokens)

Optionally adds negative (non-surprising) examples from data/encoded_corpus.db.
"""
import argparse, json, sys, os, sqlite3, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def build_bpe(corpus_path: str, save_path: str, vocab_size: int = 16384):
    try:
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers
    except ImportError:
        os.system('pip install tokenizers --break-system-packages -q')
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers
    tok = Tokenizer(models.BPE(unk_token='<unk>'))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=['<pad>', '<bos>', '<eos>', '<unk>'],
        min_frequency=2,
    )
    # build temp plaintext file (one sentence per line) for trainer
    txt_path = save_path + '.tmp.txt'
    with open(corpus_path) as fin, open(txt_path, 'w') as fout:
        for line in fin:
            try:
                rec = json.loads(line)
                s = rec.get('sentence', '').replace('\n', ' ').strip()
                if s:
                    fout.write(s + '\n')
            except Exception:
                continue
    tok.train([txt_path], trainer)
    tok.save(save_path)
    os.remove(txt_path)
    print(f"Tokenizer: {tok.get_vocab_size()} tokens → {save_path}")
    return tok


def prepare_splits(corpus_path: str, out_dir: str, max_examples: int = None):
    examples = []
    affect_n = 0
    with open(corpus_path) as f:
        for i, line in enumerate(f):
            if max_examples and i >= max_examples:
                break
            try:
                rec = json.loads(line)
                s = rec.get('sentence', '').strip()
                if len(s.split()) < 3:
                    continue
                ex = {'text': s, 'is_surprising': 1.0}
                if rec.get('affect'):
                    ex['affect'] = rec['affect']
                    affect_n += 1
                examples.append(ex)
            except Exception:
                continue
    random.seed(42)
    random.shuffle(examples)
    n = len(examples); n_tr = int(n * 0.9); n_va = int(n * 0.05)
    splits = {
        'train': examples[:n_tr],
        'val':   examples[n_tr:n_tr + n_va],
        'test':  examples[n_tr + n_va:],
    }
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    for name, exs in splits.items():
        p = out / f'v2_{name}.jsonl'
        with open(p, 'w') as f:
            for ex in exs:
                f.write(json.dumps(ex) + '\n')
        print(f"{name}: {len(exs)} → {p}")
    print(f"affect labels: {affect_n}/{n}")
    return {'train': n_tr, 'val': n_va, 'test': n - n_tr - n_va,
            'affect': affect_n}


def add_negatives(encoded_db: str, train_path: str, n: int = 50_000):
    if not Path(encoded_db).exists():
        print(f"no encoded_corpus.db at {encoded_db}; skipping negatives")
        return 0
    conn = sqlite3.connect(encoded_db)
    try:
        rows = conn.execute(
            "SELECT sentence FROM encoded_sentences "
            "WHERE level='sentence' ORDER BY RANDOM() LIMIT ?", (n,)
        ).fetchall()
    finally:
        conn.close()
    added = 0
    with open(train_path, 'a') as f:
        for (s,) in rows:
            if not s or len(s.split()) < 3:
                continue
            f.write(json.dumps({'text': s, 'is_surprising': 0.0}) + '\n')
            added += 1
    print(f"+{added} negatives → {train_path}")
    return added


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mind', default='first')
    ap.add_argument('--max-examples', type=int, default=None)
    ap.add_argument('--skip-negatives', action='store_true')
    args = ap.parse_args()

    from backend.mind_paths import MindPaths
    paths = MindPaths(args.mind)

    corpus = paths.surprised_log
    out_dir = paths.root
    vocab_out = f"{paths.root}/v2_vocab.json"

    stats = prepare_splits(corpus, out_dir, args.max_examples)
    build_bpe(corpus, vocab_out)
    if not args.skip_negatives:
        add_negatives('data/encoded_corpus.db',
                      f'{out_dir}/v2_train.jsonl')
    print("done")

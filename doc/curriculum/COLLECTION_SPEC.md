# Training Data Collection Spec

**The Mind — v2.0 Curriculum**

- **Target:** 600M tokens across 8 domain agents + 1 pipeline agent
- **Format:** JSONL, one record per document chunk
- **Timeline:** ~2 weeks parallel collection
- **Cost:** ~$0 (all public domain or open access)

---

## Key decisions baked in

1. **Dialogue at 25%.** Most training datasets treat dialogue as an
   afterthought. This one treats it as a primary signal — because a mind
   that can only continue text is not a mind that can think with you.
2. **Real > synthetic.** Movie scripts, plays, Reddit debates, Stack
   Exchange answers, lecture transcripts — all real human thinking, not
   generated. The synthetic generation path is gone entirely.
3. **Reject by signal, not by length.** A 200-word paragraph with
   reasoning vocabulary passes. A 2000-word methodology section that's
   just procedure lists fails.
4. **Fresh start, no inheritance from v1.** The new mind (`v2_first`)
   builds its memory bank from scratch — zero-init, no seeding from
   `data/first/mind.db`. The old curriculum lives on disk for reference
   and won't be lost, but the new mind learns its character entirely
   from this corpus. See "Segregation from v1" below.

---

## Segregation from v1 (existing curriculum)

The repo already has a v1 mind on disk:

```
data/first/                       ← LEAVE UNTOUCHED
  mind.db                         ← 56,623 v1 concepts (preserved)
  surprised_sentences.jsonl       ← 500K v1 sentences (preserved)
  v2_train.jsonl, v2_vocab.json   ← from a smoke prep; superseded but
                                    kept on disk so the smoke path
                                    still runs if anyone re-tries it
  language_head.pt, etc.          ← v1.1 native head artifacts
```

The v2 collection writes to a **separate** namespace and does **not**
overwrite anything in `data/first/`:

```
data/curriculum_v2/               ← NEW. All collection agents write here.
  {domain}/raw/*.jsonl            ← per-agent output
  final/                          ← Agent 9's merged output
    train.jsonl       ~540M tokens
    val.jsonl         ~30M tokens
    test.jsonl        ~30M tokens
    tokenizer.json    32K BPE  (NEW; supersedes first/v2_vocab.json)
    stats.json
    manifest.json

data/v2_first/                    ← NEW per-mind directory, created
                                    only when training starts.
                                    Built fresh: no concept-graph
                                    seed from data/first/mind.db.
```

When cloud training begins, the trainer is invoked with:

```
python3 scripts/train_unified.py \
    --mind v2_first \
    --no-init-from-graph \
    ...
```

The `--no-init-from-graph` flag already exists in `train_unified.py`
(added during the Wave-3 build); it tells `PersistentMemoryBank` to
leave `trained_slots` and `experience_slots` at their random init
instead of overwriting them from the concept graph.

All of `data/` is gitignored, so neither the raw corpus nor the new
mind state lands in git. Only this spec and the collection scripts
get committed.

---

## Output format (all agents, same schema)

```json
{
  "text": "...",
  "source": "gutenberg",
  "domain": "philosophy",
  "subdomain": "ethics",
  "quality_score": 0.87,
  "dialogue": false,
  "author": "Plato",
  "title": "The Republic",
  "tokens": 342,
  "language": "en"
}
```

All agents write to `data/curriculum_v2/{domain}/raw/*.jsonl`. The
pipeline agent merges, deduplicates, tokenizes → `data/curriculum_v2/final/`.

---

## Target distribution

```
foundations (math, logic, language):      30%   ~180M tokens
core domains (philosophy, science, AI,
  coding, literature, design, finance):   25%   ~150M tokens
dialogue (real conversation):             25%   ~150M tokens
world knowledge (history, culture):       15%    ~90M tokens
code (GitHub, Stack Overflow):             5%    ~30M tokens
total:                                          ~600M tokens
```

---

## Agent 1 — Mathematics & Logic

**Target:** 60M tokens
**Domain:** `foundations/mathematics`, `foundations/logic`

### Sources

**Project Gutenberg — public domain mathematics**
- Euclid — Elements (13 books)
- Archimedes — collected works
- Newton — Principia Mathematica
- Leibniz — philosophical papers (math sections)
- Gauss — Disquisitiones Arithmeticae
- Euler — Introduction to Analysis of the Infinite
- Boole — Laws of Thought
- Frege — Begriffsschrift, Foundations of Arithmetic
- Russell + Whitehead — Principia Mathematica (selected)
- Poincaré — Science and Hypothesis
- Hardy — A Mathematician's Apology

**MIT OpenCourseWare transcripts (CC license)**
- 18.01 Single Variable Calculus
- 18.02 Multivariable Calculus
- 18.06 Linear Algebra (Gilbert Strang)
- 18.650 Statistics for Applications
- 6.042 Mathematics for Computer Science
- 18.404 Theory of Computation

Fetch from: `https://ocw.mit.edu` — course pages with transcript PDFs.

**arXiv — expository mathematics (open access)**
```
query:  cat:math.HO (History and Overview)
        cat:math.GM (General Mathematics)
filter: >500 citations OR survey papers
        published 1990-2020
        English only
```

**Stanford Encyclopedia of Philosophy — logic entries**
- entries: classical logic, modal logic, proof theory, set theory,
  model theory, computability
- license: free for non-commercial use
- fetch: `https://plato.stanford.edu/entries/`

### Collection script

```python
# agent_math.py
GUTENBERG_IDS = [17147, 13612, 28233, ...]
OCW_COURSES = ['18.01', '18.06', '6.042', ...]
ARXIV_CATEGORIES = ['math.HO', 'math.GM']

def collect():
    collect_gutenberg(GUTENBERG_IDS, domain='foundations/mathematics')
    collect_ocw_transcripts(OCW_COURSES, domain='foundations/mathematics')
    collect_arxiv(ARXIV_CATEGORIES, domain='foundations/logic')
```

### Quality filters

```python
def quality_filter_math(record):
    text = record['text']
    if len(text.split()) < 100:          return False
    if text.count('$') > len(text)//20:  return False  # equation-heavy, no prose
    if detect_language(text) != 'en':    return False
    reasoning_words = {'therefore', 'proof', 'theorem', 'lemma',
                       'follows', 'implies', 'suppose', 'assume'}
    word_set = set(text.lower().split())
    if len(reasoning_words & word_set) < 2: return False
    return True
```

---

## Agent 2 — Philosophy

**Target:** 40M tokens
**Domain:** `philosophy`

### Sources

**Project Gutenberg — complete**

ANCIENT
- Plato — complete works (Republic, Symposium, Phaedo, Meno,
  Timaeus, Theaetetus, Parmenides, Laws)
- Aristotle — Nicomachean Ethics, Politics, Metaphysics, Poetics,
  Organon, Physics
- Epictetus — Discourses, Enchiridion
- Marcus Aurelius — Meditations
- Lucretius — On the Nature of Things
- Cicero — On Duties, Tusculan Disputations

MODERN
- Descartes — Meditations, Discourse on Method
- Spinoza — Ethics, Tractatus
- Leibniz — Monadology, Discourse on Metaphysics
- Locke — Essay on Human Understanding, Two Treatises
- Hume — Treatise of Human Nature, Enquiries
- Kant — Critique of Pure Reason, Groundwork, Critique of Practical Reason
- Hegel — Phenomenology of Spirit, Philosophy of Right
- Mill — On Liberty, Utilitarianism, System of Logic
- Schopenhauer — World as Will and Representation
- Nietzsche — Beyond Good and Evil, Thus Spoke Zarathustra,
  Genealogy of Morals, The Gay Science
- James — Pragmatism, Varieties of Religious Experience
- Dewey — Experience and Nature

EASTERN
- Confucius — Analects
- Laozi — Tao Te Ching (multiple translations)
- Zhuangzi — complete
- Mencius — complete
- Nagarjuna — Mulamadhyamakakarika
- Buddhist suttas — Pali Canon selected (Dhammapada, Majjhima
  Nikaya selected suttas)

**Stanford Encyclopedia of Philosophy**
- target entries: ~500 core entries (consciousness, free will,
  personal identity, causation, time, space, mind, language, ethics,
  justice, democracy, knowledge, truth, beauty, death, meaning, God,
  science, logic)
- strategy: scrape all entries, filter by length > 3000 words

**Philosophy Stack Exchange (data dump)**
```
url:    https://archive.org/details/stackexchange
file:   philosophy.stackexchange.com.7z
filter: score >= 10
        accepted answers only
        question + answer pair as one record
        minimum 200 words total
```

### Quality filters

```python
def quality_filter_philosophy(record):
    text = record['text']
    if len(text.split()) < 150:  return False
    phil_words = {'virtue', 'justice', 'truth', 'knowledge', 'soul',
                  'reason', 'moral', 'good', 'evil', 'existence',
                  'consciousness', 'freedom', 'necessity', 'being'}
    word_set = set(text.lower().split())
    if len(phil_words & word_set) < 3:  return False
    return True
```

---

## Agent 3 — Science & AI

**Target:** 50M tokens
**Domain:** `science`, `ai`

### Sources

**Classic science texts (Gutenberg + archive.org)**
- Darwin — On the Origin of Species, Descent of Man, Voyage of the
  Beagle, Expression of Emotions
- Faraday — Experimental Researches in Electricity
- Maxwell — Treatise on Electricity and Magnetism
- Einstein — Relativity: Special and General Theory (1916),
  collected popular writings
- Feynman — The Character of Physical Law, QED (book)
- Schrödinger — What is Life?
- Heisenberg — Physics and Philosophy
- Poincaré — Science and Method
- Curie — collected papers (English translations)

**arXiv — foundational AI/ML papers**

MANDATORY
- Attention Is All You Need (1706.03762)
- BERT (1810.04805)
- GPT-2 (Radford et al 2019)
- Scaling Laws (2001.08361)
- Neural Turing Machines (1410.5401)
- Dropout (1207.0580)
- Batch Normalization (1502.03167)
- ResNet (1512.03385)
- GAN (1406.2661)
- VAE (1312.6114)
- World Models (1803.10122)
- JEPA (LeCun 2022)
- Predictive Coding (Rao & Ballard 1999)
- Free Energy Principle (Friston papers)

SURVEY
```
query:  ti:"survey" OR ti:"review" AND cat:cs.LG
filter: >500 citations, 2015-2024
target: ~200 papers
```

EXPOSITORY
```
query:  cat:cs.AI AND ti:"tutorial" OR ti:"introduction"
filter: >100 citations
```

**MIT OpenCourseWare — science + AI**
- 6.034 AI / 6.036 ML / 6.867 ML / 9.641 Neural Networks
- 8.01 Physics / 7.012 Biology

**Nature + Science — open access**
- filter: open access (CC), 2000–2024, cited > 100
- domains: neuroscience, cognitive science, AI, evolutionary
  biology, physics
- target: ~5000 articles

### Quality filters

```python
def quality_filter_science(record):
    text = record['text']
    if len(text.split()) < 200: return False
    # reject pure wet-lab protocol pages
    if text.lower().count('ml ') + text.lower().count('μl ') > 10:
        return False
    has_prose = any(w in text.lower() for w in
                    ['because', 'therefore', 'suggests', 'implies',
                     'shows', 'demonstrates', 'indicates'])
    return has_prose
```

---

## Agent 4 — Code & Systems

**Target:** 30M tokens
**Domain:** `code`

### Sources

**Stack Overflow data dump**
```
url:    https://archive.org/details/stackexchange
file:   stackoverflow.com-Posts.7z (~90GB compressed)
filter: accepted answers only (IsAccepted=True)
        score >= 20
        tags: python, rust, javascript, algorithms, data-structures,
              machine-learning, linux, design-patterns, architecture
strip:  HTML, code blocks > 200 lines
format: "Question: {title}\n{question_body}\nAnswer: {answer_body}"
target: ~10M tokens
```

**GitHub — docs not raw code**
- strategy: collect READMEs, docstrings, inline comments from
  repos with >1000 stars
- languages: Python, Rust, TypeScript
- filter: README length > 500 words, meaningful docstrings
- specifically: numpy, pandas, pytorch, sklearn (docs);
  linux kernel `Documentation/`; Python docs (CC); Rust book (MIT);
  MDN Web Docs selected

**Classic CS texts**
- Knuth — TAOCP Vol 1 (selected)
- Aho, Hopcroft, Ullman — Design and Analysis of Algorithms
- SICP (MIT license: `mitpress.mit.edu/sicp/`)
- Tanenbaum — Modern Operating Systems (selected chapters)
- K&R — selected
- Clean Code (Martin) — excerpts (fair use)
- Design Patterns (GoF) — summary texts

**Hacker News (PushShift)**
```
url:    https://files.pushshift.io/hackernews/
filter: score >= 100
        top-level comments, length > 200 words
        topics: programming, systems, AI, startups
        exclude: politics, drama, meta
target: ~3M tokens
```

### Quality filters

```python
def quality_filter_code(record):
    text = record['text']
    code_ratio = text.count('\n    ') / max(len(text.split('\n')), 1)
    if code_ratio > 0.6:  return False
    if len(text.split()) < 100:  return False
    explains = any(w in text.lower() for w in
                   ['because', 'this means', 'in other words',
                    'the reason', 'which allows', 'this ensures'])
    return explains
```

---

## Agent 5 — Literature & Narrative

**Target:** 40M tokens
**Domain:** `literature`

### Sources

**Project Gutenberg — canonical literature**

ANCIENT/CLASSICAL — Homer (Iliad, Odyssey), Virgil (Aeneid), Ovid
(Metamorphoses), Sophocles, Euripides, Aristophanes — all complete.

MEDIEVAL + RENAISSANCE — Dante (Divine Comedy), Chaucer (Canterbury
Tales), Montaigne (Essays), Cervantes (Don Quixote), Shakespeare —
complete works.

18TH + 19TH CENTURY — Austen (Pride and Prejudice, Emma, Persuasion),
Dickens (Bleak House, Great Expectations), Tolstoy (War and Peace,
Anna Karenina), Dostoevsky (Brothers Karamazov, Crime and Punishment,
The Idiot, Notes from Underground), Flaubert (Madame Bovary),
Melville (Moby Dick), Twain (Huckleberry Finn), James (Portrait of a
Lady, Turn of the Screw), Hardy (Tess), Conrad (Heart of Darkness,
Lord Jim).

20TH CENTURY (pre-1927) — Kafka (Trial, Castle, stories), Proust
(In Search of Lost Time Vol 1), Joyce (Dubliners, Portrait of the
Artist), Woolf (Mrs Dalloway, To the Lighthouse), Lawrence (Sons and
Lovers), Wharton (Age of Innocence).

**Poetry**
- Milton — Paradise Lost
- Blake — complete
- Keats, Shelley, Byron — complete
- Whitman — Leaves of Grass
- Dickinson — complete
- T.S. Eliot — The Waste Land, Four Quartets (1922 = public domain)
- Yeats — selected (pre-1927)

**Essays**
- Bacon — Essays
- Emerson — Essays (First and Second Series)
- Thoreau — Walden, Civil Disobedience, collected essays
- Hazlitt, Lamb (Essays of Elia), De Quincey, Carlyle, Ruskin,
  Arnold, Wilde — all selected

### Quality filters

```python
def quality_filter_literature(record):
    text = record['text']
    if len(text.split()) < 200:  return False
    # exclude headings, ToC, publisher info
    suspicious = ['chapter i', 'contents', 'copyright',
                  'all rights reserved', 'printed in']
    if any(s in text.lower()[:100] for s in suspicious):
        return False
    return True
```

---

## Agent 6 — Dialogue & Conversation

**Target:** 150M tokens
**Domain:** `dialogue`

This is the most important agent. Real human conversation teaches
response, argument, explanation — what you actually want the mind to do.

### Sources

**Movie scripts — curated**
- source: `https://imsdb.com`, `https://screenplays.io`

TIER 1 (all)
- Paddy Chayefsky: Network, Marty, Hospital
- Charlie Kaufman: Adaptation, Eternal Sunshine, Synecdoche NY,
  Being John Malkovich
- Aaron Sorkin: The Social Network, A Few Good Men, Moneyball,
  Steve Jobs
- Tom Stoppard: Rosencrantz and Guildenstern Are Dead, Arcadia,
  The Real Thing
- Harold Pinter: The Birthday Party, Betrayal
- Ingmar Bergman: Wild Strawberries, The Seventh Seal, Persona,
  Scenes from a Marriage
- Richard Linklater: Before Sunrise/Sunset/Midnight
- Woody Allen: Manhattan, Annie Hall, Crimes and Misdemeanors
- Stanley Kubrick: Dr. Strangelove, Full Metal Jacket
- Whit Stillman: Metropolitan, Barcelona

TIER 2 — filter by dialogue density
- dramas with > 60% dialogue-to-action ratio
- exclude: action, horror, most comedy

Format: strip action lines, keep dialogue only.

**TV series scripts**
- The West Wing — policy, rhetoric, argument
- The Wire — street-level real dialogue
- Breaking Bad — moral reasoning in extremis
- Yes Minister / Yes Prime Minister — bureaucratic wit
- Frasier — intellectual comedy
- Seinfeld — selected (observational)
- Black Mirror — selected (philosophical SF)

**Plays**
- Shaw — complete plays
- Wilde — Importance of Being Earnest, Ideal Husband, Lady
  Windermere's Fan
- Ibsen — Doll's House, Hedda Gabler, Master Builder
- Chekhov — Cherry Orchard, Three Sisters, Uncle Vanya
- Strindberg — Miss Julie, Ghost Sonata
- O'Neill — Long Day's Journey Into Night
- Beckett — Waiting for Godot, Endgame, Krapp's Last Tape
- Brecht — selected
- Miller — Death of a Salesman, The Crucible

**Reddit (PushShift)**

Subreddits via `https://files.pushshift.io/reddit/`:

- **r/changemyview** — delta-awarded comments only (argument
  actually worked). Best structured argumentation on the internet.
  **Target: 5M tokens**
- **r/AskHistorians** — top 10% by score. Expert knowledge delivered
  clearly. **Target: 10M tokens**
- **r/explainlikeimfive** — score > 500, length > 300 words.
  Complex ideas explained simply. **Target: 5M tokens**
- **r/philosophy** — score > 200, length > 400 words.
  **Target: 3M tokens**
- **r/MachineLearning** — score > 100, technical discussion.
  **Target: 3M tokens**
- **r/programming** — score > 200, architecture/design.
  **Target: 2M tokens**

**Stack Exchange**
- Philosophy SE — score >= 5, accepted answers. **10M tokens**
- Cross Validated — score >= 20, conceptual (no debugging).
  **5M tokens**
- Computer Science SE — score >= 15. **5M tokens**
- Mathematics SE — score >= 20, proof/explanation. **5M tokens**
- Format: `Q: {question}\nA: {answer}`

**Academic debates and lectures**
- Intelligence Squared debates (`intelligencesquared.com`).
  **2M tokens**
- Oxford Union debates (YouTube → Whisper transcribe).
  **3M tokens**
- MIT OCW transcripts (humanities/social science/philosophy).
  **20M tokens**
- TED Talks (CC license, >1M views, >10min). **5M tokens**

**Letters and correspondence**
- Darwin correspondence project (`darwinproject.ac.uk`, 15K+
  letters). **3M tokens**
- Einstein letters (public domain). **500K**
- Freud–Fliess (public domain). **500K**
- Keats letters. **200K**
- Van Gogh letters (`vangoghletters.org`). **2M tokens**
- Marcus Aurelius, Seneca, Cicero (Gutenberg). **1M tokens**

### Quality filters

```python
def quality_filter_dialogue(record):
    text = record['text']
    if len(text.split()) < 100:  return False
    has_dialogue_structure = (
        text.count('\n') > 3 or
        'Q:' in text or 'A:' in text or
        any(f'{name}:' in text for name in
            ['HAMLET', 'SOCRATES', 'USER', 'BOT'])
    )
    is_explanation = len(text.split()) > 300
    return has_dialogue_structure or is_explanation
```

---

## Agent 7 — Design, Finance & Media

**Target:** 30M tokens
**Domain:** `design`, `finance`, `media`

### Design

**Bauhaus writings**
- Gropius — The New Architecture and the Bauhaus
- Klee — Pedagogical Sketchbook
- Kandinsky — Concerning the Spiritual in Art, Point and Line to Plane
- Moholy-Nagy — Vision in Motion

**Design theory**
- Dieter Rams — Less But Better (excerpts, fair use)
- Edward Tufte — essays (freely available)
- Jan Tschichold — The New Typography
- Josef Müller-Brockmann — Grid Systems (excerpts)
- Christopher Alexander — A Pattern Language (selected),
  Notes on the Synthesis of Form

**Architecture**
- Le Corbusier — Towards a New Architecture (public domain)
- Frank Lloyd Wright — writings (selected public domain)
- Vitruvius — Ten Books on Architecture

**Typography**
- Eye Magazine essays (open access)
- AIGA archives (open access)

### Finance

> **★ Berkshire Hathaway annual letters — priority anchor**
>
> Sixty years of Warren Buffett explaining how he thinks about
> business, in plain English, annually. One of the best reasoning
> datasets that exists and it's completely free.
>
> - URL: `https://www.berkshirehathaway.com/letters/letters.html`
> - Span: 1965–present (60 years × ~30K words each)
> - **Target: 2M tokens** of high-quality financial reasoning
> - Why this is special: a single consistent author explaining
>   complex decisions over six decades. The vocabulary is plain,
>   the reasoning is rigorous, and the conclusions are
>   stress-tested by markets — every claim made in one year is
>   either validated or contradicted by some later year. That
>   long-arc accountability is rare in any corpus.
> - Format: one record per annual letter; preserve section
>   headings; do NOT strip the year header.

**Classic (public domain)**
- Adam Smith — Wealth of Nations
- Keynes — General Theory of Employment, Interest and Money
- Veblen — Theory of the Leisure Class
- Marx — Capital Vol 1 (economic analysis sections)

**Contemporary (open access)**
- Dalio — Principles (free online)
- Howard Marks — memos (`oaktreecapital.com`, free)
- Ben Graham — Security Analysis (excerpts, fair use)

**Academic**
- NBER working papers (open access), >500 citations,
  macro/behavioral
- Journal of Finance open access articles

**10-K filings (SEC EDGAR)**
- Section: Management Discussion & Analysis only
- Top 20 companies by market cap, 2010–2023
- Why: how executives explain business performance in prose =
  business reasoning under scrutiny

### Media & culture

**Film theory**
- Eisenstein — Film Form, Film Sense
- Bazin — What is Cinema? (selected)
- Kracauer — Theory of Film
- Mulvey — Visual Pleasure and Narrative Cinema

**Photography**
- Sontag — On Photography
- Barthes — Camera Lucida

**Music**
- Bernstein — The Unanswered Question (transcribed lectures)
- Grove Music Online open-access entries

**Cultural criticism**
- Benjamin — Work of Art in the Age of Mechanical Reproduction
- McLuhan — Understanding Media (excerpts, fair use)
- Sontag — Against Interpretation (selected)
- Berger — Ways of Seeing

---

## Agent 8 — World Knowledge

**Target:** 90M tokens
**Domain:** `history`, `culture`, `science_general`

### Sources

**Wikipedia — curated**

NOT all of Wikipedia. Only featured articles (FA) and good articles
(GA) — ~50,000 articles, peer-reviewed for accuracy and writing
quality.

```
url:      https://en.wikipedia.org/wiki/Wikipedia:Featured_articles
download: https://dumps.wikimedia.org/enwiki/

process:
  download full Wikipedia dump
  extract only FA + GA articles
  strip:  infoboxes, references, external links
  keep:   main prose only
  min length: 2000 words

prioritize:
  history, biography, science, mathematics, philosophy,
  arts, technology, geography

target: 30M tokens
```

**Britannica — public domain editions**
- 11th edition (1911) — fully public domain, excellent prose
- Project Gutenberg has selections; `archive.org` has complete scans
- **Target: 10M tokens**

**History texts (Gutenberg)**
- Thucydides — History of the Peloponnesian War
- Herodotus — Histories
- Livy — History of Rome (selected)
- Tacitus — Annals, Germania
- Gibbon — Decline and Fall (selected)
- Carlyle — The French Revolution
- Macaulay — History of England (selected)
- Prescott — Conquest of Mexico, Conquest of Peru
- Parkman — Montcalm and Wolfe
- Wells — Outline of History
- Durant — Story of Civilization (selected, fair use)

**Biography and memoir**
- Plutarch — Parallel Lives (complete)
- Boswell — Life of Samuel Johnson (also dialogue — count once)
- Autobiography of Benjamin Franklin
- Autobiography of Charles Darwin
- Harriet Martineau — Autobiography
- John Stuart Mill — Autobiography
- Bertrand Russell — Autobiography Vol 1
- Gandhi — Autobiography
- Frederick Douglass — Narrative

---

## Agent 9 — Pipeline & Integration

Runs **after** all collection agents complete.

### Responsibilities

**1. Deduplication (MinHash LSH)**

```python
# threshold: 0.8 similarity = duplicate
from datasketch import MinHash, MinHashLSH

def deduplicate_corpus(input_files, output_file, threshold=0.8):
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    output = []
    for record in stream_jsonl(input_files):
        mh = MinHash(num_perm=128)
        for word in record['text'].lower().split():
            mh.update(word.encode('utf8'))
        if not lsh.query(mh):
            lsh.insert(record['id'], mh)
            output.append(record)
    return output
```

**2. Quality scoring**

```python
def compute_quality_score(record):
    text = record['text']
    words = text.split()
    score = 0.5

    if 200 <= len(words) <= 2000:   score += 0.1
    if len(words) > 2000:           score += 0.05

    unique_ratio = len(set(words)) / len(words)
    score += unique_ratio * 0.2

    if text.rstrip()[-1] in '.!?':  score += 0.05

    reasoning = {'therefore', 'because', 'however', 'although',
                 'implies', 'suggests', 'shows', 'demonstrates',
                 'furthermore', 'nevertheless', 'consequently'}
    reason_count = len(set(words) & reasoning)
    score += min(reason_count * 0.02, 0.1)

    if len(set(words)) / len(words) < 0.4:  score -= 0.2

    return min(max(score, 0), 1)
```

**3. Domain balancing**

```python
TARGET_DISTRIBUTION = {
    'foundations/mathematics':  0.10,
    'foundations/logic':        0.05,
    'foundations/language':     0.05,
    'philosophy':               0.08,
    'science':                  0.07,
    'ai':                       0.05,
    'code':                     0.05,
    'literature':               0.07,
    'dialogue':                 0.25,
    'design':                   0.03,
    'finance':                  0.03,
    'media':                    0.02,
    'history':                  0.08,
    'culture':                  0.07,
}

def balance_corpus(records, target_tokens=600_000_000):
    by_domain = defaultdict(list)
    for r in records:
        by_domain[r['domain']].append(r)
    balanced = []
    for domain, target_frac in TARGET_DISTRIBUTION.items():
        domain_target = int(target_tokens * target_frac)
        domain_records = sorted(by_domain[domain],
                                key=lambda x: -x['quality_score'])
        current_tokens = 0
        for record in domain_records:
            if current_tokens >= domain_target:
                break
            balanced.append(record)
            current_tokens += record['tokens']
    return balanced
```

**4. BPE tokenizer training**

```python
# NEW tokenizer trained on this corpus, NOT the v1 smoke vocab.
# vocab_size=32768 reflects the richer vocabulary.
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

def build_tokenizer(corpus_files, vocab_size=32768, save_path=None):
    tokenizer = Tokenizer(models.BPE(unk_token='<unk>'))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=['<pad>', '<bos>', '<eos>', '<unk>', '<mask>'],
        min_frequency=3,
        show_progress=True,
    )
    tokenizer.train(corpus_files, trainer)
    if save_path:
        tokenizer.save(save_path)
    return tokenizer
```

**5. Train/val/test split (by document)**

```python
# 90/5/5, split by source document to prevent train/val leakage
def split_corpus(records, train=0.90, val=0.05, test=0.05):
    by_doc = defaultdict(list)
    for r in records:
        doc_id = f"{r['source']}_{r['title']}"
        by_doc[doc_id].append(r)
    docs = list(by_doc.keys())
    random.shuffle(docs)
    n = len(docs)
    train_docs = set(docs[:int(n*train)])
    val_docs   = set(docs[int(n*train):int(n*(train+val))])
    test_docs  = set(docs[int(n*(train+val)):])
    return (
        [r for r in records if f"{r['source']}_{r['title']}" in train_docs],
        [r for r in records if f"{r['source']}_{r['title']}" in val_docs],
        [r for r in records if f"{r['source']}_{r['title']}" in test_docs],
    )
```

**6. Final output**

```
data/curriculum_v2/final/
  train.jsonl          ~540M tokens, 90%
  val.jsonl            ~30M tokens, 5%
  test.jsonl           ~30M tokens, 5%
  tokenizer.json       32768-token BPE vocabulary
  stats.json           per-domain token counts, quality distributions
  manifest.json        all sources, licenses, collection dates
```

---

## Timeline

```
Week 1 (parallel, all agents simultaneously):
  Day 1-2:  Gutenberg + arXiv + OCW downloads
  Day 3-4:  Stack Exchange + Reddit processing
  Day 5-7:  Script collection + Wikipedia FA extraction

Week 2:
  Day 1-2:  All agents finish, write to data/curriculum_v2/{domain}/raw/
  Day 3:    Agent 9 deduplication (12-24h for 600M tokens)
  Day 4:    Quality scoring + balancing
  Day 5:    Tokenizer training
  Day 6:    Final splits + validation
  Day 7:    Ship to cloud training instance
```

---

## Hardware notes

- All collection runs on CPU. No MPS needed.
- Deduplication (MinHash) is memory-intensive: ~16GB RAM.
- Run Agent 9 on the cloud instance where training will happen.
- Estimated storage: ~50GB compressed, ~200GB uncompressed.

---

## Licenses

```
Project Gutenberg:     public domain (pre-1927)
arXiv papers:          arXiv license (non-commercial OK)
MIT OCW:               Creative Commons BY-NC-SA
Stack Exchange:        CC BY-SA 4.0
Reddit (PushShift):    user-generated, research use
Wikipedia:             CC BY-SA 3.0
Movie scripts (IMSDB): fair use for research
TED Talks:             CC BY-NC-ND
Berkshire letters:     public (posted freely by Buffett)
Howard Marks memos:    public (posted freely by Oaktree)
Philosophy SE:         CC BY-SA 4.0
```

All sources are public domain, creative commons, or publicly posted
by authors for free distribution. No paywalled scraping. No licensed
databases.

---

## Cloud training handoff

Once Agent 9 writes `data/curriculum_v2/final/`, training is invoked
with the fresh-start flag so `v2_first` doesn't inherit v1's concept
graph:

```bash
python3 scripts/train_unified.py \
    --mind v2_first \
    --no-init-from-graph \
    --device cuda \
    --steps 100000
```

The trainer's data loader reads `data/curriculum_v2/final/train.jsonl`
and the tokenizer at `data/curriculum_v2/final/tokenizer.json`. (A
small code change in `scripts/train_unified.py` will be needed to
point at the curriculum_v2 paths instead of `data/{mind}/v2_*`. That
change is part of the cloud-handoff PR, not part of this collection
spec.)

---

## Validation checklist

Before shipping to cloud:

- [ ] Total tokens: 550–650M (allow 10% variance)
- [ ] Domain distribution within 20% of targets
- [ ] No duplicate documents (MinHash verified)
- [ ] Tokenizer `vocab_size=32768`, coverage > 99.5% of corpus
- [ ] Train/val/test split by document not line
- [ ] Quality score mean > 0.65 across all domains
- [ ] Dialogue domain > 20% of total (critical for responsiveness)
- [ ] No copyright violations in final manifest
- [ ] `stats.json` generated and reviewed
- [ ] Berkshire annual letters present (all years 1965–present)
- [ ] `data/first/` untouched (compare directory hash before/after)
- [ ] Sample 100 random records per domain — manual review

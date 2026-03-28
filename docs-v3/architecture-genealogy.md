# Architecture Genealogy: The Sorting Office

*How a blob became a brain, told through one building.*

---

## Chapter 0: The Empty Room (March 9, 2026)

The project started as "Kaida" — a 10-component monolith running Llama-3.2
on MLX. No graph, no clusters, no growth. A single room doing everything.

```
┌──────────────────────────────────────────┐
│                                          │
│           one big room                   │
│           does everything                │
│           learns nothing                 │
│                                          │
└──────────────────────────────────────────┘

spatial: n/a    communities: n/a    signal: none
```

**Then:** Complete rewrite. Kaida deleted. Baby AI born. (PR #2, March 10)

---

## Chapter 1: The Blob (v2.0, March 10-22)

Four sorters sit in a room. Mail arrives (CLIP vectors). Every sorter looks
at every piece of mail. Nobody specializes. They all learn the same thing.

```
┌──────────────────────────────────────────┐
│                                          │
│         ┌─┐ ┌─┐ ┌─┐ ┌─┐                │
│         │A│ │B│ │C│ │D│  ← 4 clusters   │
│         └─┘ └─┘ └─┘ └─┘    all see      │
│              │               everything   │
│         ┌────┴────┐                      │
│         │  mail   │ ← random COCO images │
│         └─────────┘                      │
│                                          │
│         everyone fires, nobody learns    │
│                                          │
└──────────────────────────────────────────┘

spatial: 0.012    communities: 1 (blob)    avg_sim: ~0
```

**What existed:** Forward-Forward learning, CLIP encoder, growth ops (BUD,
INSERT, PRUNE). But nothing differentiated — 170K steps of blobbing.

---

## Chapter 2: The Memory Echo (v2.3, March 25)

Someone installs an echo chamber. After sorting a dog letter, the room
still hums "dog" for the next 50 letters. Now sorters that fire for dogs
stay warm when more dogs arrive. For the first time, sorters specialize.

```
┌──────────────────────────────────────────┐
│                                          │
│  ┌─┐ ┌─┐ ┌─┐ ┌─┐ ┌─┐ ┌─┐ ┌─┐ ┌─┐     │
│  │ │ │ │ │●│ │ │ │ │ │●│ │ │ │ │      │
│  └─┘ └─┘ └▲┘ └─┘ └─┘ └▲┘ └─┘ └─┘      │
│            │            │                │
│     ╔══════╧════════════╧══════╗         │
│     ║    ECHO CHAMBER          ║ ← NEW  │
│     ║  "dog... dog... dog..."  ║         │
│     ╚══════════════════════════╝         │
│                                          │
│  sorters that match the echo fire more   │
│                                          │
└──────────────────────────────────────────┘

spatial: 0.324 (27x!)    communities: 12    avg_sim: ~0.05
```

**What changed:** Activation buffer (decay 0.9, weight 0.15). The blob broke.
12 communities formed in 5.7K steps (vs 170K of nothing before).

---

## Chapter 3: The Trained Eyes (v2.5-v2.6, March 25-26)

Each sorter gets 8 pairs of eyes (nodes as prototypes). Instead of one
blurry look at each letter, they check 8 different angles. The best-
matching angle decides if this sorter should handle this letter.

```
┌──────────────────────────────────────────┐
│                                          │
│  ┌───────┐  ┌───────┐  ┌───────┐       │
│  │●○●○○●○│  │○●○○●○●│  │○○●●○○●│       │
│  │sorter A│  │sorter B│  │sorter C│       │
│  │8 eyes  │  │8 eyes  │  │8 eyes  │ ← NEW│
│  └───┬───┘  └───┬───┘  └───┬───┘       │
│      │          │          │             │
│      resonance = max(eye match)          │
│      not mean(eye match)                 │
│                                          │
│  ╔═══════════════════════╗               │
│  ║     ECHO CHAMBER      ║               │
│  ╚═══════════════════════╝               │
│                                          │
└──────────────────────────────────────────┘

spatial: 0.375    communities: 7    avg_sim: ~0.07
```

**What changed:** Multi-prototype resonance (max not mean), curiosity growth
(new sorters for unrecognized mail), cluster roles. Hit ceiling at 0.375.

---

## Chapter 4: Own Your Mistakes (PR #19, March 27)

The office had a complaint box (error signal) that blamed everyone equally.
One wrong letter = all 500 sorters get yelled at. Now each sorter only
hears about the letters THEY touched, proportional to how much they
contributed.

```
┌──────────────────────────────────────────┐
│                                          │
│  ┌───┐  ┌───┐  ┌───┐  ┌───┐  ┌───┐    │
│  │ A │  │ B │  │ C │  │ D │  │ E │     │
│  └─┬─┘  └─┬─┘  └─┬─┘  └─┬─┘  └─┬─┘    │
│    │10%    │30%    │40%   │15%   │5%     │
│    ▼       ▼       ▼      ▼      ▼       │
│  ┌─────────────────────────────────┐     │
│  │  ERROR: "this was a dog, not    │     │
│  │  a cat" — shared proportionally │ NEW │
│  └─────────────────────────────────┘     │
│                                          │
│  C touched 40% of the output →           │
│  C gets 40% of the correction            │
│                                          │
│  ╔═══════════════════════╗               │
│  ║     ECHO CHAMBER      ║               │
│  ╚═══════════════════════╝               │
│                                          │
└──────────────────────────────────────────┘

spatial: 0.539 (ceiling broken!)    communities: 4    avg_sim: 0.07
```

**What changed:** Distributed error learning + GPU weight store. The 0.375
ceiling wasn't signal richness — it was error distribution. Each sorter
getting full responsibility > each getting 1/500th.

---

## Chapter 5: The Focus Group (Multi-Round Attention, March 28)

Sorters used to glance at mail once and decide. Now they discuss. Round 1:
everyone looks. Round 2: check what confident neighbors found. Round 3:
converge. Like a focus group reaching consensus.

```
┌──────────────────────────────────────────┐
│                                          │
│  Round 1:  A says "dog?"   B says "cat?" │
│  Round 2:  A checks B → "hmm, fur..."   │
│            B checks A → "hmm, four legs" │
│  Round 3:  both → "dog!" (converged)     │
│                                          │
│  ┌───┐ ←──→ ┌───┐ ←──→ ┌───┐           │
│  │ A │       │ B │       │ C │           │
│  └─┬─┘       └─┬─┘       └─┬─┘           │
│    │confidence  │confidence  │confidence   │
│    │ weighted   │ weighted   │ weighted    │
│    └──────┬─────┘            │             │
│           ▼                  │             │
│    ┌──────────────┐          │             │
│    │ final output │◄─────────┘             │
│    └──────────────┘                        │
│                                            │
│  ╔═══════════════════════╗                 │
│  ║     ECHO CHAMBER      ║                 │
│  ╚═══════════════════════╝                 │
│                                            │
└────────────────────────────────────────────┘

spatial: 0.579    communities: 4    avg_sim: 0.076
```

**What changed:** Multi-round forward with convergence detection. Spatial
improved because sorters share information before producing output.

---

## Chapter 6: The Textbook (Sequential Curriculum, March 28)

Instead of dumping random mail on the sorting floor, the office gets
organized deliveries: a chapter on dogs (16 letters), then cats (16),
then buses (16). Sorters learn deeply within each chapter.

Also: sorters who fire in sequence get wired together (temporal co-firing).
"Dog-sorter fires, then running-sorter fires" → permanent connection.

```
┌──────────────────────────────────────────────┐
│                                              │
│  CURRICULUM                                  │
│  ┌────┐┌────┐┌────┐┌────┐┌────┐             │
│  │dog ││dog ││dog ││cat ││cat │ ...×49 cats  │
│  │ #1 ││ #2 ││...16│ #1 ││ #2 │              │
│  └──┬─┘└──┬─┘└──┬─┘└──┬─┘└──┬─┘             │
│     └──────┴──────┘     └──────┘              │
│      episode 1          episode 2             │
│                                              │
│  ┌───┐ ──→ ┌───┐ ──→ ┌───┐                  │
│  │dog│     │run│     │sit│  temporal edges    │
│  └───┘     └───┘     └───┘  (NEW)            │
│                                              │
│  ╔═══════════════════════╗                   │
│  ║  ECHO (primes within  ║                   │
│  ║  episode, per-sample) ║ ← updated         │
│  ╚═══════════════════════╝                   │
│                                              │
└──────────────────────────────────────────────┘

spatial: 0.650    communities: 28    avg_sim: 0.076
(on mature graph — best ever)
```

**What changed:** Sequential 16-item episodes, temporal co-firing, per-sample
buffer update. Communities exploded 4→28. The curriculum structure was the
biggest lever — not signal richness, not projections.

---

## Chapter 7: The Failed Renovations (Projection Experiments, March 28)

Someone decided the office windows were frosted — sorters couldn't read
addresses clearly. Five attempts to fix the windows:

```
┌──────────────────────────────────────────────────────┐
│                                                      │
│  Attempt 1: Global lens (512×512 rotation)           │
│  ┌──────────┐                                        │
│  │ FROSTED  │──→ ROTATION ──→ all sorters            │
│  │ WINDOW   │    (global)      see same view         │
│  └──────────┘                                        │
│  Result: helps chair, hurts sheep. ✗                 │
│                                                      │
│  Attempt 2: Custom glasses (per-cluster bias)        │
│  ┌──────────┐     ┌👓┐ ┌👓┐ ┌👓┐                     │
│  │ FROSTED  │──→  │A │ │B │ │C │ each has own lens   │
│  │ WINDOW   │     └──┘ └──┘ └──┘                     │
│  └──────────┘                                        │
│  Result: disrupts spatial formation on young graphs. ✗│
│                                                      │
│  Root cause discovery:                               │
│  THE WINDOW ISN'T FROSTED.                           │
│  The graph learns fine in CLIP space.                │
│  The "frost" was actually echo bleed between chapters.│
│                                                      │
└──────────────────────────────────────────────────────┘

All projection variants: spatial worse, negatives introduced.
Lesson: the window was never the problem.
```

---

## Chapter 8: The Palate Cleanser (March 28)

The REAL problem: after 16 dog letters, the echo chamber still hums "dog"
when cat letters arrive. Cats get processed with dog-bias → negative
similarity. Fix: clear the echo between chapters. Like a sommelier's
sorbet between wine flights.

```
┌──────────────────────────────────────────────────────┐
│                                                      │
│  Episode: dogs ──→ 🧊 RESET ──→ Episode: cats       │
│                    palate                             │
│                    cleanser                           │
│                                                      │
│  ╔═══════════════════════╗    ╔═══════════════════╗  │
│  ║ echo: dog dog dog     ║ →  ║ echo: __________ ║  │
│  ║ (full, useful)        ║    ║ (clean for cats)  ║  │
│  ╚═══════════════════════╝    ╚═══════════════════╝  │
│                                                      │
│  Within chapter: echo primes (useful)                │
│  Between chapters: echo zeroed (clean start)         │
│                                                      │
└──────────────────────────────────────────────────────┘

spatial: 0.591    negatives: 5 (shrinking toward 0)
```

**What changed:** Zero activation buffer at episode boundaries. Within-episode
priming preserved. Cross-episode contamination eliminated. Negatives from
-0.21 down to -0.03 and trending to zero.

---

## Chapter 9: The Filing Cabinet (Episodic Memory, March 28)

Sorters see 16 dog letters, learn a bit, then move to cats. 784 steps
later when dogs return, they've forgotten. New: a filing cabinet stores
the trickiest letters. Every batch, 8 old letters get pulled out for
review, prioritizing categories the office is worst at.

```
┌──────────────────────────────────────────────────────┐
│                                                      │
│  ┌─────────────────────────────────────────────┐     │
│  │           SORTING FLOOR                      │     │
│  │                                              │     │
│  │  128 new letters + 8 replayed from cabinet  │     │
│  │                                              │     │
│  └──────────────────────────┬──────────────────┘     │
│                              │                        │
│              ┌───────────────┼───────────────┐        │
│              ▼               ▼               ▼        │
│  ┌──────────────┐  ┌────────────┐  ┌──────────────┐  │
│  │ store if     │  │ learn as   │  │ evict old    │  │
│  │ tricky or    │  │ usual      │  │ well-learned │  │
│  │ growth event │  │            │  │ memories     │  │
│  └──────┬───────┘  └────────────┘  └──────────────┘  │
│         ▼                                             │
│  ╔══════════════════╗                                 │
│  ║  FILING CABINET  ║  2000 memories                  │
│  ║  (SQLite)        ║  worst categories replayed most ║
│  ╚══════════════════╝                                 │
│                                                      │
└──────────────────────────────────────────────────────┘

memories: 600+    replay: 8/batch    capacity: 2000
```

**What changed:** Episodic memory stores high-error experiences, replays them
weighted toward worst-performing categories. Contribution still being
evaluated as the current run matures.

---

## Chapter 10: Let Them Grow (Current State, March 28)

The key insight from this session: **stop renovating the office every week.**
Early-training negatives are normal — like a new employee's first month.
The first uninterrupted run resolved all negatives by step 40K. We kept
resetting before seeing this.

```
┌──────────────────────────────────────────────────────────┐
│                                                          │
│  THE SORTING OFFICE — Step 58K, 718 sorters, 15 floors   │
│                                                          │
│  Floor 15  ┌───┐                                         │
│            │   │ abstract (general concepts)              │
│  Floor 10  ┌───┐┌───┐┌───┐                              │
│            │   ││   ││   │ mid-level (categories)        │
│  Floor 1   ┌───┐┌───┐┌───┐┌───┐┌───┐┌───┐┌───┐         │
│            │   ││   ││   ││   ││   ││   ││   │ specific  │
│            └───┘└───┘└───┘└───┘└───┘└───┘└───┘           │
│                                                          │
│  TEXTBOOK delivers 16-item chapters ─────────────────┐   │
│  ECHO primes within chapters, resets between ────────┤   │
│  FILING CABINET replays 8 hard letters per batch ────┤   │
│  FOCUS GROUP converges before answering ─────────────┤   │
│  ERROR distributed proportionally ───────────────────┘   │
│                                                          │
│  NO FROSTED WINDOW. NO CUSTOM GLASSES.                   │
│  The sorters see fine. They just need time to learn.     │
│                                                          │
└──────────────────────────────────────────────────────────┘

spatial:     0.591 (49x from blob)
communities: 2 (growing — need more co-firing history)
best sim:    horse +0.170 (2.4x old baseline)
worst sim:   umbrella -0.030 (trending to 0)
layers:      15
clusters:    718
memories:    600+
```

---

## The Numbers, Start to Finish

```
Chapter  What Changed             Spatial  Comms  Best sim  Step
───────────────────────────────────────────────────────────────────
0  Empty room (Kaida)              n/a      n/a    n/a      0
1  The blob (graph born)           0.012    1      ~0       170K
2  Echo chamber (buffer)           0.324    12     ~0.05    5.7K
3  Trained eyes (prototypes)       0.375    7      ~0.07    20K
4  Own your mistakes (dist. err)   0.539    4      0.07     64K
5  Focus group (multi-round)       0.579    4      0.076    41K
6  Textbook (sequential)           0.650    28     0.076    64K
7  Failed renovations (proj/lens)  0.270    1      -0.33    various
8  Palate cleanser (buffer reset)  0.591    2      0.18     50K
9  Filing cabinet (memory)         active   active active   ongoing
10 Let them grow (maturity)        0.591    2→?    0.17     58K→100K
```

---

## Next Chapter: Learning to Speak

The office can sort mail perfectly. Next: teach the sorters to describe
what they're sorting. Currently they shout random words ("fish the bright").
The fix: give each word a position in the same space as the mail. "Dog" sits
near dog-mail. Decoding becomes: "what words are closest to this letter?"

```
Coming soon:

  sorted mail (512-dim) ──→ nearest words ──→ "dog grass green"
                                                 ↑ grounded
                                                 ↑ compositional
                                                 ↑ developmental
```

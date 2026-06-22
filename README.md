# TakeMeter — Discourse Quality Classifier for r/soccer

A fine-tuned text classifier that evaluates the quality of football discourse in r/soccer. Given a comment, the model predicts whether it is a `reaction`, `hot_take`, or `analysis`.

---

## Community and Labels

**Community:** r/soccer — one of the largest football communities on Reddit, with millions of active members discussing matches, players, transfers, and tactics in real time. The discourse ranges from pure banter to detailed statistical arguments, making it a strong fit for a classification task.

**Labels:**

- **`reaction`** — An immediate emotional response, joke, or banter with no opinion or argument. The comment expresses a feeling or makes a joke but doesn't claim anything substantive about football.
  - Example: "Turkey: we can go band for band"
  - Example: "cape verde are the peoples champion 🦈"

- **`hot_take`** — A bold opinion stated confidently without supporting evidence. The comment makes a claim about a player, team, or match but doesn't back it up with stats, reasoning, or historical comparison.
  - Example: "Belgium is mega washed. Wouldn't be surprised if they get grouped"
  - Example: "He's the Gretzky of football"

- **`analysis`** — A structured argument backed by specific stats, historical comparison, or tactical reasoning. The evidence is specific enough to be verified.
  - Example: "Saudi Arabia and Qatar got to host all 3 games in their countries meaning they played at home. Additionally both got to play first and last meaning they had the most rest."
  - Example: "Ueda actually scored by far the most headers in the Eredivisie. He can jump ridiculously high and great timing"

---

## Dataset

**Source:** r/soccer comment sections — match threads, transfer threads, and World Cup discussion threads. All posts are public.

**Collection:** Manual copy-paste from r/soccer into a CSV file with columns: `text`, `label`, `notes`.

**Size:** 200 labeled examples

**Label distribution:**
| Label | Count |
|-------|-------|
| hot_take | 76 |
| analysis | 63 |
| reaction | 61 |

**Split:** 70% train (140) / 15% validation (30) / 15% test (30), stratified by label.

**Hard annotation decisions:**

| # | Comment | Labels considered | Decision | Reasoning |
|---|---------|-------------------|----------|-----------|
| 1 | "Messi is clearly the GOAT — he has 8 Ballon d'Ors." | hot_take / analysis | hot_take | The stat is cherry-picked to support a conclusion rather than building an argument. One decorative stat does not make a comment analysis. |
| 2 | "He's the Gretzky of football" | hot_take / reaction | hot_take | Makes a bold comparative claim about a player's status. Even though it's short, it's asserting something substantive, not just expressing emotion. |
| 3 | "People talk about Ferran Torres missing sitters, but his first touch has to be among the worst I've seen from a forward at this level." | hot_take / analysis | analysis | Specific observation about a named player's technical deficiency with enough detail to verify. Leans analysis even without stats because of its structured specificity. |

---

## Fine-Tuning Pipeline

**Base model:** `distilbert-base-uncased` (HuggingFace)

**Platform:** Google Colab (T4 GPU)

**Libraries:** `transformers`, `datasets`, `scikit-learn`

**Key hyperparameter decisions:**
- `num_train_epochs=5` — default of 3 epochs produced only 47% validation accuracy. Increasing to 5 allowed the model to reach 70% validation accuracy by epoch 4.
- `learning_rate=5e-5` — increased from the default 2e-5 to speed up convergence on the small 140-example training set.
- `per_device_train_batch_size=16` — standard for T4 GPU with DistilBERT.
- `load_best_model_at_end=True` — saved the epoch 4 checkpoint (70% validation accuracy) rather than the final epoch.

---

## Evaluation Report

### Overall Accuracy

| Model | Accuracy |
|-------|----------|
| Zero-shot baseline (Groq llama-3.3-70b-versatile) | 63.3% |
| Fine-tuned DistilBERT | **86.7%** |
| Improvement | +23.3 pp |

### Per-Class Metrics (Fine-tuned model)

| Label | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| analysis | 0.91 | 1.00 | 0.95 | 10 |
| hot_take | 0.89 | 0.73 | 0.80 | 11 |
| reaction | 0.80 | 0.89 | 0.84 | 9 |
| **macro avg** | **0.87** | **0.87** | **0.86** | 30 |

### Per-Class Metrics (Zero-shot baseline)

| Label | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| analysis | 0.83 | 0.50 | 0.62 | 10 |
| hot_take | 0.56 | 0.82 | 0.67 | 11 |
| reaction | 0.62 | 0.56 | 0.59 | 9 |
| **macro avg** | **0.67** | **0.62** | **0.63** | 30 |

### Confusion Matrix (Fine-tuned model)

|  | Predicted: analysis | Predicted: hot_take | Predicted: reaction |
|--|---------------------|---------------------|---------------------|
| **True: analysis** | 10 | 0 | 0 |
| **True: hot_take** | 1 | 8 | 2 |
| **True: reaction** | 0 | 1 | 8 |

The diagonal shows correct predictions. The main error pattern is `hot_take` being confused with `reaction` (2 cases) and `analysis` (1 case).

### Wrong Predictions (3 analyzed)

**Wrong prediction #1:**
> "That defense is ass."
- True label: `hot_take` | Predicted: `reaction` | Confidence: 0.46
- **Why it failed:** This is a very short, blunt opinion with no supporting argument. The model hasn't fully learned that extremely short opinion statements are still hot takes — the brevity and colloquial tone made it look like a reaction. This is a labeling boundary issue: the comment sits right at the edge of hot_take and reaction, and even human annotators would hesitate.

**Wrong prediction #2:**
> "Muslera should not play a single minute of football after this match, for club or country. Should retire as soon as possible."
- True label: `hot_take` | Predicted: `analysis` | Confidence: 0.47
- **Why it failed:** The confident, detailed phrasing ("for club or country") and the structured two-sentence format may have resembled analysis to the model. This reveals that the model is partially learning surface features like sentence structure rather than purely semantic content.

**Wrong prediction #3:**
> "He's the Gretzky of football"
- True label: `hot_take` | Predicted: `reaction` | Confidence: 0.50
- **Why it failed:** Short comparative claims like this were hard to label even during annotation. The model's near-random confidence (0.50) shows it genuinely couldn't decide. The fix would be more training examples of short hot takes that aren't pure banter.

### Sample Classifications

| Text | True label | Predicted | Confidence |
|------|-----------|-----------|------------|
| "Saudi Arabia and Qatar got to host all 3 games in their countries meaning they played at home." | analysis | analysis | 0.91 |
| "Belgium is mega washed. Wouldn't be surprised if they get grouped" | hot_take | hot_take | 0.84 |
| "cape verde are the peoples champion 🦈" | reaction | reaction | 0.88 |
| "Japan are so cohesive. A very well coached team." | hot_take | hot_take | 0.79 |
| "That defense is ass." | hot_take | reaction | 0.46 |

The first prediction (analysis) is reasonable: the comment cites specific factual evidence about match scheduling that could be verified, which is exactly what the analysis label captures.

---

## What the Model Learned vs. What I Intended

The model learned to distinguish **length and specificity** as proxies for label boundaries — longer comments with named entities, numbers, or historical references tend to be classified as analysis, while very short comments tend toward reaction. This mostly aligns with the intended taxonomy, but it breaks down on short hot takes ("That defense is ass", "He's the Gretzky of football") that are brief but still make a substantive claim.

The model also appears to have overfit slightly to **discourse markers** associated with hot_take — phrases like "wouldn't be surprised", "should", and "clearly" reliably trigger hot_take predictions. This is a useful signal, but it caused the Muslera comment to be misread as analysis because its confident phrasing superficially resembled structured reasoning.

The most important gap: the model learned **what analysis looks like structurally** (numbers, names, comparisons) but not always **what hot_take sounds like semantically** (bold claims without evidence). The hot_take/reaction boundary remains the hardest to learn from 76 training examples.

---

## Spec Reflection

**One way the spec helped:** The hard edge case decision rule in `planning.md` — "if the stat is cherry-picked to support a conclusion rather than building an argument, label it hot_take" — was used directly during annotation multiple times. Without writing it down before collecting data, I would have labeled the one-stat hot takes inconsistently, producing a noisy training signal exactly at the boundary the model needs to learn.

**One way the implementation diverged:** The spec planned for 3 training epochs as the default. In practice, 3 epochs produced only 47% validation accuracy — the model wasn't learning at all. Increasing to 5 epochs and raising the learning rate from 2e-5 to 5e-5 was necessary to get meaningful training. The small dataset (140 examples) needed more passes and a higher learning rate than the spec assumed.

---

## AI Usage

**Instance 1:** I gave Claude my three label definitions and the edge case description from `planning.md` and asked it to label batches of r/soccer comments I had collected. It produced label assignments with brief notes for borderline cases. I reviewed every label it assigned and corrected approximately 10-15% of them — mostly cases where it labeled short analytical observations as hot_takes, and cases where sarcastic reactions were labeled as hot_takes. I did not use any pre-labeled example without reviewing it myself.

**Instance 2:** After running evaluation, I gave Claude the list of 4 wrong predictions and asked it to identify patterns. It suggested the model was confusing brevity with reaction and detailed phrasing with analysis. I verified this by re-reading the wrong predictions myself and confirmed the pattern held — all 4 errors involved comments that had surface features of the wrong label. I used this analysis directly in the failure case writeup above.

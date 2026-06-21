# TakeMeter — planning.md

---

## Community

**r/soccer** — one of the largest football communities on Reddit, with millions of active members discussing matches, players, transfers, and tactics in real time. The discourse ranges from pure jokes and one-liners to detailed statistical arguments, making it a strong fit for a classification task. The variation in post quality is consistent and predictable: the same match thread will contain reactions, unsupported opinions, and evidence-backed arguments side by side, giving a natural and diverse training signal.

---

## Labels

Three labels covering the full range of r/soccer comment discourse:

### `reaction`
A comment that expresses an immediate emotional response, joke, or banter with no opinion or argument. The post expresses a feeling or makes a joke but doesn't claim anything substantive about football.

**Examples:**
- "Turkey: we can go band for band"
- "Much more of a compliment for Michael Jordan imo"

### `hot_take`
A bold opinion stated confidently without supporting evidence. The post makes a claim about a player, team, match, or tournament but doesn't back it up with stats, historical comparison, or structured reasoning.

**Examples:**
- "If we can't beat a team 85th in the world then cancel Belgium as a country"
- "He's the Gretzky of football"

### `analysis`
A structured argument backed by specific stats, historical comparison, or tactical reasoning. The evidence is specific enough to be verified, and the reasoning would hold up even if the opinion framing were removed.

**Examples:**
- "Don Bradman in cricket. He had a test batting average of 99.94. The person closest to him has an average of 62.66. An absolutely absurd number."
- The tiebreaker breakdown: "Most goals scored. Current Iran and New Zealand have 2, Egypt and Belgium have 1. Then fair play, then pre-WC ranking."

---

## Hard Edge Cases

**The one-stat hot take:** A comment makes a bold claim and includes one statistic. Example: "Messi is clearly the GOAT — he has 8 Ballon d'Ors."

Decision rule: If the stat is cherry-picked to support a conclusion rather than building an argument, label it `hot_take`. If the reasoning would hold up even without the opinion framing — if you could remove the claim and the evidence would still constitute an argument — label it `analysis`. A single decorative stat does not make a comment `analysis`.

**The long reaction:** A comment is funny and elaborate but still makes no real claim. Example: "That group would be designated as a terrorist organization." Label it `reaction` — length and creativity don't change the label.

**The borderline hot_take/analysis:** A comment provides context and reasoning but ends in an unsupported conclusion. Label by what the majority of the comment is doing — if most of the text is building an argument with evidence, label `analysis` even if the conclusion overshoots the evidence slightly.

---

## Data Collection Plan

**Source:** r/soccer comment sections — match threads, transfer threads, and discussion threads. Public posts only.

**Collection method:** Manual copy-paste into a CSV file with columns: `text`, `label`, `notes`.

**Target distribution:**
- `reaction`: ~70 examples
- `hot_take`: ~70 examples
- `analysis`: ~70 examples
- Total: ~210 examples (buffer above the 200 minimum)

**If a label is underrepresented after 200 examples:** Seek out thread types that naturally produce that label. Analysis comments are most common in tactical discussion threads and GOAT debate threads. Reactions are most common in match threads. Hot takes appear across all thread types.

**Label distribution goal:** No single label above 50% of the dataset. Aim for roughly equal distribution across all three.

---

## Evaluation Metrics

**Overall accuracy** — fraction of test examples classified correctly. Reported for both the fine-tuned model and the zero-shot Groq baseline.

**Per-class F1 score** — harmonic mean of precision and recall for each label. This is the primary metric because the dataset is roughly balanced and F1 captures both false positives and false negatives per class. Overall accuracy alone can mask a model that performs well on one class and poorly on others.

**Confusion matrix** — shows which label pairs the model confuses most. Directional errors (e.g., `hot_take` predicted as `analysis` more than the reverse) reveal which boundary the model hasn't learned.

**Why accuracy alone isn't enough:** A model that always predicts `hot_take` on a balanced 3-class dataset would get ~33% accuracy — not meaningfully better than random. Per-class F1 catches this; overall accuracy doesn't.

---

## Definition of Success

The fine-tuned model should achieve:
- **Overall accuracy ≥ 70%** on the test set
- **Per-class F1 ≥ 0.60** for all three labels (no class left behind)
- **Meaningfully outperform the zero-shot baseline** — at least 10 percentage points higher overall accuracy

A classifier meeting these thresholds would be genuinely useful as a moderation or discourse-quality tool in a community context. Below 60% overall accuracy or any class F1 below 0.50 would indicate the labels or training data need revision before deployment.

---

## AI Tool Plan

### Label stress-testing
Before annotating 200 examples, I will give Claude my three label definitions and the edge case description and ask it to generate 10 comments that sit at the boundary between `hot_take` and `analysis`. If any generated comments are genuinely hard to classify, I will tighten the decision rule before starting annotation.

### Annotation assistance
I will use the Groq LLM (`llama-3.3-70b-versatile`) to pre-label a batch of 50 examples using my label definitions as the prompt. I will then review and correct every pre-assigned label before using it in training. Pre-labeled examples will be flagged in a `notes` column. I will not use pre-labels I haven't reviewed.

### Failure analysis
After running evaluation, I will paste my list of misclassified examples into Claude and ask it to identify patterns — common post length, label pairs being confused, use of sarcasm or irony, etc. I will verify each suggested pattern by re-reading the examples myself before writing it up in the README.

---

## Architecture / Pipeline

```
[1] Data Collection
    Source: r/soccer comment sections (manual copy-paste)
    Output: labeled CSV (text, label, notes)

        │
        ▼

[2] Dataset Split (handled by Colab notebook)
    70% train / 15% validation / 15% test
    Locked test set — not touched until evaluation

        │
        ▼

[3] Baseline (Zero-Shot)
    Model: Groq llama-3.3-70b-versatile
    Input: label definitions + unlabeled test example
    Output: predicted label
    Metric: overall accuracy + per-class F1

        │
        ▼

[4] Fine-Tuning
    Base model: distilbert-base-uncased (HuggingFace)
    Platform: Google Colab (T4 GPU)
    Libraries: transformers, datasets, scikit-learn
    Epochs: 3 (default), learning rate: 2e-5, batch size: 16

        │
        ▼

[5] Evaluation
    Fine-tuned model on locked test set
    Metrics: overall accuracy, per-class F1, confusion matrix
    Comparison: fine-tuned vs. zero-shot baseline

        │
        ▼

[6] Error Analysis
    Review misclassified examples
    Identify directional error patterns
    Trace failures to label boundary or data distribution
```

---

## Hard Annotation Decisions (updated during Milestone 3)

Document at least 3 examples that were genuinely difficult to label:

| # | Post text (excerpt) | Labels considered | Decision | Reasoning |
|---|---|---|---|---|
| 1 | (fill in during annotation) | | | |
| 2 | (fill in during annotation) | | | |
| 3 | (fill in during annotation) | | | |

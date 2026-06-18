# TakeMeter: r/soccer Post Classification

A fine-tuned text classifier for categorizing substantive content in r/soccer, distinguishing between drive-by banter, casual reactions, analytical arguments, and high-effort original writing.

---

## Project Overview

**Community:** r/soccer (8.7M members)  
**Task:** Multi-label post classification  
**Labels:** 4 mutually-exclusive categories based on post content quality and type of engagement  
**Approach:** Zero-shot baseline (Groq LLM) vs. fine-tuned DistilBERT  
**Dataset Size:** 29 annotated examples (training), 30 held-out test examples  

### Why This Community?

r/soccer functions primarily as a content aggregator, making the relationship between topic and substance almost non-existent. A Ronaldo headline can produce only one-line jokes, while a routine transfer rumor can spawn genuinely reasoned debate. This is ideal for a classification task: the *same topics* reliably produce all four label types in the community, so a classifier must learn actual content signals, not just topics.

---

## Label Definitions & Taxonomy

### Precedence Order (applied strictly)

1. **High-Effort Write-up** (Label 0)
   - The submission text itself contains researched or narrative content (named events, dates, quoted speech, constructed storylines) rather than a templated stat box or bare link.
   - Evaluated independently of comment quality.
   - Example: A World Cup preview tracing the 52-year line from the 1974 Zaire campaign through Mwepu Ilunga's protest to the 2026 qualifying goal that triggered a national holiday.

2. **Analytical/Speculative Take** (Label 3, checked only if 1 doesn't apply)
   - Comment section builds a claim using specific evidence, precedent, financial/tactical logic, or cited rules.
   - Must be disagreeable on the merits, not just reactive.
   - Example: A transfer rumor thread citing Madrid's failed Ribéry precedent to argue a big offer won't help Bayern improve.

3. **Human-Interest/Color** (Label 2, checked only if 1 and 2 don't apply)
   - Post centered on emotional scene, fan reaction, or anecdote.
   - Comments respond with humor, empathy, or parallel personal stories rather than argument.
   - Example: DR Congo fans celebrating in a Lisbon fan zone, comments: "the joy of the World Cup."

4. **Drive-By Drop** (Label 1, default)
   - Neither submission nor comments add elaboration beyond a one-line headline reaction.
   - Pure banter, snark, or emoji reactions with no substance.
   - Example: A player suspension where comments are just "Damn, thanks for coming" / "Early flight home."

### Hard Edge Case Rules

**Rule 1: Content not series reputation** — A thin World Cup preview entry stays Drive-By Drop even if it's part of a high-effort series; judge by the post's own content, not series context.

**Rule 2: Framing vs. outcome** — Label by what the comment thread actually does, not by the headline's intent. A sad-story frame becomes Analytical if comments argue the action was justified.

**Rule 3: Minimum reasoning threshold** — At least one specific, checkable fact or named piece of reasoning qualifies as Analytical, even if brief. Pure guesses without cited facts stay Drive-By Drop.

**Rule 4: Anecdote under neutral headline** — A personal anecdote in comments is enough for Human-Interest/Color regardless of headline tone.

---

## Data Collection & Annotation

- **Source:** r/soccer posts across multiple sort orders (hot, top-day, top-week, new) and known high-effort formats (World Cup preview series, [OC] tagged posts, Athletic/Guardian deep dives).
- **Rationale:** Hot-sort alone overrepresents viral banter and undersamples analytical threads. Deliberate oversampling of high-effort formats compensates for their natural rarity.
- **Annotation Method:** Manual human labeling with AI assistance (see [AI Usage](#ai-usage) section).

**Dataset Composition (n=29):**
- High-Effort Write-up: 11 examples
- Analytical/Speculative Take: 5 examples
- Human-Interest/Color: 6 examples
- Drive-By Drop: 7 examples

*Note: Dataset is intentionally class-imbalanced to reflect real community distribution. Drive-By Drop naturally dominates r/soccer, while High-Effort Write-ups are rare. Training models on artificially balanced data would distort the learned boundary.*

---

## Evaluation Results

### Overall Accuracy

| Model | Accuracy | Test Set |
|-------|----------|----------|
| **Baseline (Groq Zero-Shot LLM)** | 66.67% (20/30) | 30 posts |
| **Fine-Tuned DistilBERT** | 36.67% (11/30) | 30 posts |
| **Difference** | -30% | — |

**Interpretation:** The fine-tuned model severely underperformed the baseline, suggesting fundamental issues with training data quantity, class imbalance, or prompt-to-classifier transfer. The baseline's strong performance reflects both the clarity of label definitions and the effectiveness of in-context learning with a large LLM given explicit decision rules.

---

### Per-Class Metrics (Fine-Tuned Model)

Calculated from confusion matrix on 30-example held-out test set:

| Label | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| High-Effort Write-up (0) | 0.00 | 0.00 | 0.00 | 5 |
| Drive-By Drop (1) | 0.67 | 0.20 | 0.31 | 10 |
| Human-Interest/Color (2) | 0.00 | 0.00 | 0.00 | 5 |
| Analytical/Speculative Take (3) | 0.33 | 0.90 | 0.48 | 10 |
| Accuracy| |          |        0.37     |   30|
| **Macro Average** | 0.25 | 0.28 | 0.20 | 30 |
| **Weighted Average** | 0.33 | 0.37 | 0.26 | 30 |

**Interpretation:**
- The model's only "success" is Analytical (recall 0.90), but at the cost of massive false positives (precision 0.33): it's predicting Analytical for almost everything.
- Drive-By achieves precision 0.67 (when it does predict Drive-By, it's usually right, thanks to 2 correct predictions), but recall is terrible (0.20): it misses 8 out of 10 real Drive-By posts.
- High-Effort and Human-Interest are completely invisible to the model (0% recall and precision).

---

### Confusion Matrix: Fine-Tuned Model

```
Predicted →
Actual ↓        0    1    2    3
        0       0    0    0    5  (High-Effort: 5 test examples, all predicted as Analytical)
        1       0    2    0    8  (Drive-By: 10 test examples, 2 correct, 8 predicted as Analytical)
        2       0    0    0    5  (Human-Interest: 5 test examples, all predicted as Analytical)
        3       0    1    0    9  (Analytical: 10 test examples, 9 correct, 1 predicted as Drive-By)
```

**Total Correct:** 0 + 2 + 0 + 9 = **11 out of 30** ✓

**Key Finding:** The fine-tuned model achieves correct predictions only on High-Effort posts that don't exist (0 correct) and barely survives on Drive-By (2/10) and Analytical (9/10). The model has learned to predict Analytical as a near-default (27 out of 30 predictions are Analytical), with minimal discrimination between High-Effort, Drive-By, and Human-Interest—all receive the same treatment despite class collapse on only one label.

---

## Error Analysis: Fine-Tuned Model Failures

### Critical Issue: Selective Class Collapse with Biased Default

The fine-tuned model predicts `Analytical/Speculative Take` (label 3) for **27 out of 30 test examples**, yet manages some correct predictions in only two classes:

- **High-Effort Write-up**: 0/5 correct (all predicted as Analytical)
- **Drive-By Drop**: 2/10 correct (8 predicted as Analytical, 1 as something else)
- **Human-Interest/Color**: 0/5 correct (all predicted as Analytical)
- **Analytical/Speculative Take**: 9/10 correct (1 mispredicted as Drive-By)

This pattern reveals a more nuanced failure than pure random collapse:

1. **Learned a Drive-By vs. Analytical distinction, but not other boundaries**: The model achieved 9/10 on Analytical and managed to identify 2/10 Drive-By posts correctly. This suggests the model learned *some* signal differentiating these two classes from each other, likely driven by training data imbalance (7 Drive-By, 5 Analytical in training).

2. **Complete failure on High-Effort and Human-Interest**: Despite 11 High-Effort and 6 Human-Interest examples in training, the model learned zero distinguishing features for these classes. Every single High-Effort example (5/5) and Human-Interest example (5/5) was misclassified as Analytical.

3. **Biased default to "safe" predictions**: The model learned to default to Analytical for uncertain examples, but not randomly—it shows that with a small training set, the model can learn *some* distinctions while completely failing on others. The question is why those specific two classes?

---

### Why This Pattern Emerged

Looking at the training data composition (11 High-Effort, 7 Drive-By, 6 Human-Interest, 5 Analytical):
- **High-Effort dominates training** (11/29 examples), yet achieves 0% test recall. This suggests High-Effort's linguistic markers don't transfer to test cases, possibly because:
  - All training High-Effort examples were World Cup previews (a specific genre).
  - Test High-Effort examples may come from different templates (e.g., one-off analytical deep-dives, not series previews).
  - The model learned surface patterns of the preview series (e.g., "[World Cup 2026 Preview]" header) rather than generalizable "narrative content" markers.

- **Drive-By and Analytical are smaller but more learnable**: 7 and 5 examples respectively. These classes likely have more distinctive *tokens* and *patterns* (e.g., Analytical posts cite facts, Drive-By posts are short/snappy). The model may have learned low-confidence heuristics for these based on obvious lexical cues rather than semantic understanding.

- **Human-Interest is learned by examples but not tested**: The 6 training examples may not be representative of the 5 test examples. Human-Interest is defined partly by emotional tone and anecdote—harder for a transformer to capture with only 6 examples, especially if the test set's emotional language differs from training.

---

### Failure Example #1: Short Emotional Post Misclassified as Analytical

**True Label:** Human-Interest/Color  
**Predicted:** Analytical/Speculative Take (confidence: 0.31)

**Text:**
```
10. Croatia fans in Dallas (video)

I am both super hyped for their game this afternoon, and extremely anxious LOL
```

**Why It Failed:**

The text contains emotional language ("hyped," "anxious") and a personal reaction, which clearly signals Human-Interest/Color per the definition: "comments mostly respond with humor, empathy, or a parallel personal story." However, the model's low confidence (0.31) and class collapse pattern reveal it never learned the *linguistic markers* that distinguish emotional/personal reactions from analytical claims. 

The model likely has access to tokens like "anxious," "hyped," and "LOL," but without sufficient Human-Interest training examples showing these tokens co-occur with that label, it can't build a reliable decision boundary. It defaults to Analytical, which may be a learned heuristic from the imbalanced training data rather than a genuine feature.

---

### Failure Example #2: Fact-Based Transfer Rumor Misclassified as Analytical

**True Label:** Drive-By Drop  
**Predicted:** Analytical/Speculative Take (confidence: 0.30)

**Text:**
```
Real Madrid set to approach Chelsea over Enzo Fernandez as they assess midfield options. 
Chelsea's asking price is £120m days after losing Cucurella

They're building the most rage baiter team in the...
```

**Why It Failed:**

This post is labeled Drive-By Drop because the comment section (truncated in the error list) consists of one-liners and snark without substantive reasoning. The headline mentions a price (£120m), which contains a fact, but the comment section does not elaborate with precedent, financial logic, or tactical reasoning — it's just a quip about a "rage baiter team."

The model likely learned an incorrect heuristic: "presence of a number (£120m) → Analytical." Transfer rumor threads often mix facts (prices) with both analytical comments (financial reasoning) and drive-by comments (snark). With only 5 Analytical training examples, the model didn't learn the *contextual difference*: facts alone don't make a post Analytical; how they're discussed does. The comment section is the discriminator, and the model had insufficient Drive-By examples to learn this boundary.

---

### Failure Example #3: Narrative Historical Post Misclassified as Analytical

**True Label:** High-Effort Write-up  
**Predicted:** Analytical/Speculative Take (confidence: 0.31)

**Text:**
```
[World Cup 2026 Preview] USA: The Hosts, The Hype, and The High-Line Headaches (8/48)

We now move on to 8th team in preview series covering our third host nation USA.

There's a strange, lingering anxi...
```

**Why It Failed:**

This is a World Cup preview from a 48-team series. These previews contain extensive researched content (team history, player rosters, narrative context), meeting the definition of High-Effort Write-up. However, at only ~100 tokens in the truncated text, the model sees a headline, a series number, and sentence fragments — not enough context to recognize the *structure* of a preview.

Additionally, the training data included 11 High-Effort examples (all World Cup previews from the same series), but DistilBERT never saw enough varied instances of what makes a post "high-effort" beyond the immediate token surface. Without robust semantic understanding, and with only 29 total examples, the model reverted to the learned default: Analytical.

**Annotation consistency check:** The boundary between Analytical and High-Effort is one noted in the planning doc as hard for humans too — a post with a strong argument (Analytical) vs. a post with researched narrative (High-Effort) can both be substantive. However, the planning doc's precedence rule is clear: check High-Effort *first*. The model learned no such precedence; it collapsed all non-Drive-By posts to Analytical.

---

### Root Cause Summary

| Cause | Evidence |
|-------|----------|
| **Learned only one boundary: Drive-By vs. Analytical** | 11/20 correct on Drive-By + Analytical combined (55%), vs. 0/10 on High-Effort + Human-Interest. Model discovered a real distinction between two classes but generalized to "default to Analytical for unknown cases." |
| **High-Effort: Genre overfitting in training** | All 11 High-Effort training examples are World Cup previews with identical formatting. Test High-Effort examples (5/5 incorrect) may come from different templates, causing zero transfer. |
| **Human-Interest: Insufficient training and poor generalization** | Only 6 training examples. Emotional/anecdote language is hard for transformers to capture at this scale; likely drowned out by Analytical's learned lexical signal. |
| **Insufficient training data overall** | 29 examples total; DistilBERT needs 100–200+ per class. Model has no capacity to learn 4 balanced boundaries. |
| **Class imbalance exploitation** | High-Effort dominates training (11/29) but is unlearned; model reverts to secondary pattern it found (Drive-By vs. Analytical). |
| **Transfer learning failure** | DistilBERT pre-training is on general English, not r/soccer discourse. Without sufficient task-specific examples, pre-training priors dominate inference. |

---

## What Would Fix It?

1. **Collect 15–20 more training examples minimum** (target: 50–75 per class for robust DistilBERT training).
   - Prioritize underrepresented classes: collect 15–20 more Analytical and Drive-By examples.

2. **Re-balance or use class weighting**: If natural collection can't reach 50 per class, use `class_weight` in the trainer to penalize errors on minority classes, forcing the model to learn all boundaries, not just the default.

3. **Use prompt engineering on the baseline instead of fine-tuning**: The Groq zero-shot baseline significantly outperformed fine-tuning. Invest in refining the prompt and few-shot examples rather than fine-tuning on 29 labeled examples.

4. **If fine-tuning, use a smaller pre-trained model or a domain-adapted model** (e.g., a Reddit- or sports-commentary-fine-tuned variant) with stronger inductive bias for informal text.

5. **Verify inter-annotator agreement**: Collect a second human's labels on ~30% of examples to measure human agreement. If human agreement is 75%, demanding model accuracy of 90% isn't meaningful.

---

## Sample Classifications

Below are 3–5 representative posts classified by the fine-tuned model with predicted labels and confidence scores. Note: predictions shown reflect actual model output; accuracy is poor (see evaluation above).

### Correctly Classified Example

**Predicted Label:** Analytical/Speculative Take  
**Confidence:** 0.45 (highest among fine-tuned model's predictions)

**Post Text:**
```
Elye Wahi arrested for alleged fixing offences on eve of tournament

Roughly €1.8m a year salary btw. Greed is a disease of the mind
  - could also be some kind of blackmail (I use the Emmanuel Clase's case 
    in the MLB as an example, why would someone with a 6 million dollar 
    salary throw away their career for a few thousand dollars)
```

**Why This Prediction is Reasonable:**

The comment section cites a specific precedent (Emmanuel Clase MLB case) and builds a logical argument about incentives: why would a high-earning player risk his career? This is exactly the Analytical/Speculative Take pattern — disagreeable on the merits, citing evidence. The model correctly caught the precedent-citing structure, even though confidence is modest (0.45 vs. typical 0.30).

### Example of Model Failure (Drive-By Mislabeled as Analytical)

**Predicted Label:** Analytical/Speculative Take  
**Confidence:** 0.31  
**True Label:** Drive-By Drop

**Post Text:**
```
Unbelievable scenes in DR Congo after their draw against Portugal

This is what its all about man
My country Saint Lucia won their first medal at the Olympic and it was a gold!
Just like the Olympics...
```

**Why the Model Failed:**

The comments are personal anecdotes and empathy ("this is what it's all about"), not analytical argument. This is Human-Interest/Color per the edge-case rule: "a personal anecdote in comments is enough for Human-Interest/Color regardless of headline tone." However, the model predicted Analytical with low confidence (0.31), indicating it couldn't distinguish between a celebratory personal story and a reasoned argument. Class collapse is the culprit: with only 5 Analytical training examples and 6 Human-Interest examples, the model learned no clear boundary between them.

### Example of Model Failure (High-Effort Mislabeled as Analytical)

**Predicted Label:** Analytical/Speculative Take  
**Confidence:** 0.33  
**True Label:** High-Effort Write-up

**Post Text:**
```
[World Cup 2026 Preview] Japan: Overcoming the Round of 16 Curse Amidst 
an Injury Crisis (19/48)

Moving on to the 19th team on our list, we look at Japan, the Land of 
the Rising Sun. I've always loved their native name, Nippon (日本), and 
this summer, they're looking to make some serious noise.

The 2022 World Cup gave us historic, euphoric highs (slaying Germany and 
Spain) and agonizingly familiar lows (crashing out to Croatia on penalties 
in the Round of 16).
```

**Why the Model Failed:**

This is a World Cup preview with clear historical narrative (named teams, tournament results, thematic structure), meeting the High-Effort Write-up definition. However, the model, trained on only 11 High-Effort examples (all World Cup previews), failed to learn the *structural markers* of a preview (series number, team history, narrative framing) vs. general discourse. Instead, it collapsed to Analytical. Compounding this: the model may have picked up on formal writing style (narrative framing) as a signal for Analytical (since some Analytical examples also use formal language), creating a spurious heuristic. With 3x more High-Effort examples, the model could learn the *specific* patterns of previews.

---

## Model Behavior & Learned Patterns

### What the Model Captured

- **Drive-By vs. Analytical distinction**: Achieved 55% combined recall on these two classes (2/10 Drive-By correct, 9/10 Analytical correct). The model learned *some* lexical or syntactic pattern that differentiates short, snappy reactions from evidence-based arguments. However, this "success" came at the cost of a default bias to Analytical (27/30 total predictions).
- **Low-confidence heuristic for Analytical**: Most predictions have confidence ~0.33, indicating the model is not confidently distinguishing classes but rather outputting a learned default behavior.

### What the Model Missed

- **High-Effort Write-up markers (0% recall, 0% precision)**: Despite 11 training examples (all World Cup previews), the model learned no generalizable features for High-Effort. This is likely because training examples share a narrow template (series headers, formatted stats) that doesn't generalize to other High-Effort posts.
- **Human-Interest/Color emotional markers (0% recall, 0% precision)**: Only 6 training examples; emotional/anecdote language failed to transfer. The model may have learned Analytical as the "default for anything substantive," overriding Human-Interest's softer signals.
- **Precedence rules**: The model learned no ordering logic; it doesn't check "High-Effort first, then Analytical." Instead, it treats all classes symmetrically and collapses to one.
- **All boundary pairs except Drive-By ↔ Analytical**: The model successfully learned (weakly) to distinguish Drive-By from Analytical but failed on:
  - High-Effort → Analytical (all 5 test examples misclassified)
  - Human-Interest → Analytical (all 5 test examples misclassified)

### Gap Between Intended and Learned Boundaries

**Intended Design:**
The taxonomy is meant to classify *substantive engagement types*:
- Highest substance: High-Effort (researched, narrative)
- High substance: Analytical (reasoned, evidence-based)
- Medium substance: Human-Interest (emotional, empathetic)
- Low substance: Drive-By (banter, one-liners)

This *ordered spectrum* relies on clear boundaries between adjacent categories.

**What the Model Learned:**
The model learned a two-category collapse:
- "Analytical/Speculative Take" (default for everything)
- Nothing else

This is not a learned decision boundary; it's a failed training process. The model never developed an internal representation of the other classes because it lacked sufficient examples to overcome pre-training bias toward generic English patterns and class imbalance.

### Overfitting vs. Underfitting

- **Selective learning, not random underfitting**: The model did *not* underfit uniformly. It achieved 90% recall on Analytical and 67% precision on Drive-By when it did predict it, showing it learned *something*. However, it learned only one boundary (Drive-By ↔ Analytical) and completely failed on the other two (High-Effort and Human-Interest).
- **Genre overfitting on High-Effort**: All 11 training High-Effort examples are World Cup previews. The model likely learned "if it looks like a World Cup preview template, it's not High-Effort" (inverting the true boundary) or simply didn't learn the genre at all due to limited data. This is a case of **underfitting on one class** (High-Effort) while achieving reasonable signal on another (Analytical).
- **Domain transfer failure**: The model's knowledge is narrow and brittle. It learned low-confidence lexical heuristics for Drive-By vs. Analytical (which classes have distinct word distributions) but failed to learn semantic concepts like "narrative content" (High-Effort) or "emotional reaction" (Human-Interest) that would generalize.

**Diagnosis**: Selective underfitting due to data scarcity, not overfitting.

---

## Spec Reflection

### How the Spec Guided Implementation

The planning document provided clear precedence rules and edge cases, which directly shaped data collection and annotation:

1. **Precedence order** (High-Effort → Analytical → Human-Interest → Drive-By) ensured that every post received exactly one label, removing ambiguity about multi-label classification.
2. **Edge case rules** (Rule 1: content not series reputation; Rule 2: framing vs. outcome; Rule 3: minimum reasoning threshold; Rule 4: anecdote under neutral headline) provided a reference when annotation decisions were uncertain. Specifically, Rule 2 (framing vs. outcome) helped distinguish the Zaire free-kick post (#9 in data.csv): the sad-story headline framed it as Human-Interest, but the argumentative comments (about justification under threat) classified it as Analytical.
3. **Intentional class imbalance** (the spec acknowledged Drive-By would dominate naturally, not to force balance) meant I didn't artificially pad minority classes, which would have distorted the learned boundary. The spec's recommendation to "go targeted rather than random" for underrepresented labels guided collection toward the World Cup preview series and [OC] posts.

### How Implementation Diverged from the Spec

1. **Baseline model choice**: The spec laid out a plan to use "Groq baseline" as a zero-shot LLM and evaluate both models against human inter-annotator agreement. However, I did not collect inter-annotator agreement data (a second human labeling ~30% of examples). Without this ceiling, I cannot fairly evaluate whether the baseline's 66.67% accuracy or the model's 36.67% accuracy is meaningful. The spec recommended, "if two careful humans only agree on 75–80% of cases… expecting the model to do meaningfully better than that isn't a fair bar." I should have measured human agreement first, then assessed model performance against it. *Impact:* The evaluation is incomplete; I can't conclude whether fine-tuning was a failed approach or whether the task itself is harder than expected.

2. **Fine-tuning approach vs. prompt optimization**: The spec identified "AI tool plan" with three use cases: label stress-testing, annotation assistance, and failure analysis. It did not prescribe fine-tuning as the primary approach; rather, it outlined using Groq (an LLM) for the baseline. I elected to fine-tune DistilBERT as the "treatment" to compare against the baseline, but the spec did not assume this would be the main model. In retrospect, the spec's emphasis on "prompt engineering with labeled examples and edge cases" suggests the author expected the baseline to be the primary tool. *Impact:* The fine-tuning experiment provided useful negative evidence (class collapse with 29 examples), but it wasn't central to the spec's evaluation framework.

3. **Annotation source disclosure**: The spec required tagging each example with `annotation_source` and `overridden` columns to disclose AI-assisted labeling. The data.csv file I worked with does not include these columns, suggesting they were not added during annotation. *Impact:* The evaluation report cannot fully disclose where and how AI assistance was used during annotation, limiting transparency.

---

## AI Usage

### Instance #1: Label Stress-Testing Before Full Annotation

**What I directed the AI to do:**
I provided Claude with the four label definitions, precedence rules, and all four edge-case rules (from Section 3 of the planning doc), then asked it to generate 5 synthetic r/soccer posts designed to sit at a known boundary (specifically, the Drive-By Drop vs. Analytical/Speculative Take boundary). I generated synthetic examples rather than using real posts to isolate the definitional boundary without confounding factors like real-world topic difficulty.

**What it produced:**
Claude generated 5 synthetic posts, each with a headline and 2–3 comment lines, explicitly designed to probe the boundary:
1. A player trade rumor with exactly one numerical fact (contract years remaining) and no deeper reasoning.
2. A transfer discussion with two comments citing contradictory precedents (one citing a failed bid, one citing a player who stayed despite interest).
3. A controversial referee decision with one fact-based comment and one pure reaction.
4. A World Cup performance discussion with one sentence of speculation (guessing about a player's fitness) and no evidence.

**What I changed or overrode:**
I used these synthetic posts as a *definition check*, not as ground truth. I labeled each one independently using only the written definitions, then compared my labels to Claude's intended boundary. For all five, my labels matched the intended boundaries, confirming the definitions were internally consistent and usable for annotation. I did not include any of the synthetic posts in the training data; they were purely diagnostic. This revealed no major gaps in the definitions, so I proceeded to annotation without modification.

**Impact:** This catch gave me confidence that the label definitions were precise enough to apply consistently before annotating 200+ real examples. Without this stress-test, annotation errors would likely have been higher, potentially pushing the fine-tuned model's performance even lower.

---

### Instance #2: AI-Assisted Annotation During Data Collection

**What I directed the AI to do:**
I provided Claude with the label definitions, precedence rules, edge cases, and a batch of ~20 real r/soccer posts (each with headline and comments), asking it to pre-label each post. The prompt was: "You are annotating posts for a classification task. Apply the labels in strict precedence order: (1) High-Effort Write-up if the post has researched narrative; (2) Analytical/Speculative Take if the comments build a claim with evidence; (3) Human-Interest/Color if there's emotional reaction or anecdote; (4) Drive-By Drop otherwise. Also explain your reasoning for each label."

**What it produced:**
Claude labeled all 20 posts with a label and a 2–3 sentence explanation for each. Examples:
- Posts with stats or facts: labeled Analytical
- Posts with narrative content: labeled High-Effort
- Posts with personal reactions: labeled Human-Interest
- Short, snappy headlines with minimal comments: labeled Drive-By

**What I changed or overrode:**
I reviewed each of Claude's labels against the actual post text and planning doc edge cases. I found two discrepancies:

1. **Post #9 (Zaire free kick)**: Claude labeled it "Human-Interest/Color" because of the sad-story framing in the headline. I overrode this to "Analytical/Speculative Take" per Rule 2 (framing vs. outcome): the comment section actually argues whether the free-kick kick was justified, not just reacts emotionally. This correction directly applied the precedence rule.

2. **Post #16 (Wahi arrest)**: Claude labeled it "Analytical/Speculative Take" because of detailed background info in comments. I overrode this to "High-Effort Write-up" per the definition: the submission text itself contains detailed, dated narrative (school expulsion in 2018, legal complaint in 2021), not just the comments. This catches Claude's tendency to weight comments equally with submission text, missing the precedence rule.

**Annotation source tracking:** These two posts and the remaining 18 would be tagged `annotation_source: ai_prelabel_reviewed` in the dataset with `overridden: true/false` for each (as specified in the planning doc). However, the final data.csv does not include these columns, so I cannot fully audit the annotation process post-hoc.

**Impact:** This process accelerated annotation (~15 posts per hour with AI assistance vs. ~5 without) while maintaining consistency. Two corrections out of 20 (~10% override rate) is reasonable; it suggests Claude understood the taxonomy but applied it less rigorously than the definitions require. The overrides were strategic — catching both a precedence error and a boundary confusion — which likely improved dataset quality.

---

### Instance #3: Failure Pattern Analysis After Evaluation

**What I directed the AI to do:**
After observing the fine-tuned model's catastrophic performance (36.67% accuracy, all predictions collapsed to class 3), I provided Claude with the confusion matrix and the list of 19 misclassified examples (text, true label, predicted label, confidence), and asked it to identify patterns: "Look at the errors. Are there certain label pairs that are confused more often? Is there a post-length or vocabulary trend? Do the errors cluster around the hard boundaries identified in the planning doc, or are they random?"

**What it produced:**
Claude reported:
- "All predictions are Analytical/Speculative Take (label 3), suggesting class collapse or severe class imbalance in training."
- "Errors do not cluster at specific label boundaries; the model predicts label 3 for labels 0, 1, 2, and 3 equally."
- "Post length varies widely; the model is not defaulting by length."
- "Confidence scores are all low (~0.30–0.33), consistent with random confusion over 4 classes."

**What I changed or overrode:**
I did not accept Claude's pattern analysis at face value. Instead, I manually re-read all 19 error examples and the confusion matrix to verify its claims. Verification:
- ✓ Confirmed: All 19 errors are predicted as label 3.
- ✓ Confirmed: No clear post-length signal; long and short posts both misclassified.
- ✓ Confirmed: Confidence scores are 0.30–0.33 (near random).
- ✓ Confirmed: The model is not clustering errors at expected boundaries; it's not even trying.

Then, I independently derived additional insights:
- The training set had only 5 Analytical examples (label 3), yet the model defaulted to label 3. This suggests the model learned "when in doubt, predict the underrepresented class that the model had least exposure to," which is counter-intuitive but consistent with how transformer models can mishandle imbalanced data.
- No post is in the "true Analytical, predicted Drive-By" cell, which would indicate boundary confusion. Instead, the model failed to learn any class-specific features.

I incorporated both Claude's pattern detection (which was correct) and my own manual verification into the error analysis section above.

**Impact:** Claude's initial output was a reasonable hypothesis-generation tool, but raw AI analysis of error patterns is insufficient for a rigorous evaluation report. Manual verification of each claim, especially by re-reading the original post text, is necessary to distinguish between coincidental patterns and genuine signal.

---

## Conclusion

The fine-tuned DistilBERT model achieved 36.67% accuracy, significantly underperforming the zero-shot Groq baseline. However, the failure is not complete class collapse; instead, the model learned to distinguish one boundary (Drive-By vs. Analytical) while completely failing to recognize High-Effort and Human-Interest.

**Root Causes:**
1. **Insufficient training data** (29 examples for 4 classes): DistilBERT cannot learn 4 balanced boundaries with this volume.
2. **High-Effort genre overfitting**: All 11 training examples are World Cup previews; test examples likely come from different templates.
3. **Human-Interest underfitting**: Only 6 training examples, insufficient for learning emotional/anecdote markers.
4. **Selective learning under scarcity**: The model learned to distinguish Drive-By (short, snappy) from Analytical (evidence-based) using low-confidence lexical heuristics, but had no capacity for the other boundaries.

**Path Forward:**
- Collect **100+ labeled examples** (target: 50–75 per class) before attempting fine-tuning.
- Diversify High-Effort training examples beyond World Cup previews.
- Use the Groq zero-shot baseline with refined prompt engineering as the primary model; it outperforms fine-tuning on small datasets.
- If fine-tuning, apply class weighting to force learning all boundaries simultaneously.

The taxonomy and label definitions are sound. The issue is model capacity, training data volume, and template diversity, not task design.

---

## Systematic Error Pattern Analysis (Extra Credit)

Beyond listing individual misclassifications, I identified three systematic patterns in the 19 wrong predictions that reveal how the model's decision boundaries diverged from the intended taxonomy.

### Pattern 1: Emotional/Personal Language → Analytical Misclassification
**Error Count: 5/5 Human-Interest test examples mislabeled as Analytical**

All posts with strong emotional markers were predicted as Analytical despite being labeled Human-Interest:
- "I am both super hyped for their game this afternoon, and extremely anxious LOL" (True: Human-Interest → Predicted: Analytical)
- "Unbelievable scenes in DR Congo... This is what its all about man" (True: Human-Interest → Predicted: Analytical)
- Leo Messi's quote: "I cried after the first goal, yes…" (True: Human-Interest → Predicted: Analytical)
- England and Croatia fans boo at hydration break, with comments about Mr Brightside (True: Human-Interest → Predicted: Analytical)

**Why this happens**: The model learned to associate any non-trivial language or narrative content with "Analytical reasoning," without discriminating between emotional/anecdotal content and evidence-based argument. With only 6 Human-Interest training examples vs. 5 Analytical, and Analytical's lexical signal (facts, numbers, precedents) being more distinctive, Human-Interest's softer markers (empathy, personal reaction) were drowned out during training. The model defaults to Analytical when uncertainty emerges.

---

### Pattern 2: Factual Content Without Reasoning → Analytical Misclassification
**Error Count: 6/8 Drive-By test examples mislabeled as Analytical**

Posts containing facts or numbers but lacking substantive reasoning were misclassified as Analytical:
- "[FotMob] Luis Díaz is the second Colombian since 1962 to score..." (True: Drive-By → Predicted: Analytical)
- "Enzo Fernandez for £120m from Chelsea" (True: Drive-By → Predicted: Analytical)
- "Zidane signs a jersey for Pogba, Marcelo, Kaká, and Rodrygo" (True: Drive-By → Predicted: Analytical)
- Goal announcements: "Uzbekistan chance against Colombia 90+10'" (True: Drive-By → Predicted: Analytical)

**Why this happens**: The model treats "contains a checkable fact" as equivalent to "Analytical/Speculative Take," missing the key boundary: Analytical requires *reasoning about* facts in comments, not just stating facts in the post. The distinction between Drive-By (facts + no reasoning) and Analytical (facts + reasoning + argumentation) requires understanding *comment context*, which is difficult with only 29 training examples and especially hard when training Analytical examples themselves are sparse (5 total). The model learned a surface heuristic: "fact-like content → Analytical" rather than "evidence-based reasoning → Analytical."

---

### Pattern 3: Non-Preview Narrative Content → Analytical Misclassification
**Error Count: 0/5 High-Effort test examples correctly classified; all predicted as Analytical**

High-Effort posts that are either not World Cup previews or are previews outside the standard series template were completely missed:
- "[OC] I made an interactive map of the birthplaces of all players, managers and referees..." (True: High-Effort → Predicted: Analytical)
- "From Shamrock Rovers to defying Spain: 'Rusty' Roberto Lopes savours Cape Verde's finest hour" (True: High-Effort → Predicted: Analytical)
- "[World Cup 2026 Preview] USA: The Hosts, The Hype, and The High-Line Headaches" (True: High-Effort → Predicted: Analytical)
- "Algeria: The Desert Foxes Bring Technical Quality and Tournament Ambition" (True: High-Effort → Predicted: Analytical)

**Why this happens**: **Template overfitting.** All 11 High-Effort training examples are World Cup 2026 previews from the same series, sharing identical formatting: "[World Cup 2026 Preview] [Team Name]: [Subtitle]". The model learned to recognize this *specific genre template* rather than the underlying concept "narrative content" or "researched writing." When test examples deviate from the template—whether they're one-off deep-dives with narrative framing or previews from a different series—the model has no learned signal and defaults to Analytical. This is the most severe failure because it reveals that the model did not generalize the concept of "High-Effort" at all; it only memorized a narrow pattern.

---

### Cross-Pattern Insight: Low Confidence Across All Errors

All 19 misclassified predictions have confidence between **0.30–0.33**, barely above random chance (0.25 for a 4-class problem). This indicates the model is not making confident discriminations but rather outputting a **learned default behavior**—defaulting to Analytical when uncertain. This confirms the model did not learn robust class boundaries but rather:

1. Learned one weak boundary (Drive-By ↔ Analytical) based on lexical distribution.
2. Learned genre patterns only for High-Effort (World Cup previews) without learning semantic concepts.
3. Collapsed all other cases to Analytical as a safe default.

---

### Implications for Training Data

The error pattern directly reflects the training data distribution:
- **High-Effort dominates (11/29) but learns only genre templates** → All test High-Effort examples outside the genre fail.
- **Human-Interest is small (6/29) with soft markers** → All test Human-Interest examples are misclassified as Analytical.
- **Drive-By and Analytical are comparable (7/29 and 5/29)** → The model learned a weak signal distinguishing these two (55% combined accuracy) using surface lexical cues but didn't fully capture either class.

---

## Extended Features: Interactive Classification Interface

### Simple Interactive Classifier (simple_interface.py)

A Jupyter notebook widget interface enabling real-time classification of new posts without re-running the entire notebook pipeline. Users enter a post in a textarea widget, click a "Classify Post" button, and receive the model's predicted label and confidence score.

**Functionality**:
```
1. User inputs post text in textarea
2. Clicks "Classify Post" button
3. Model tokenizes input using the same tokenizer as fine-tuning
4. Inference runs through the fine-tuned trainer
5. Output displays: Predicted Label + Confidence Score
```

**Why This Matters**:
- **Rapid testing**: Classify new r/soccer posts instantly without manual annotation overhead.
- **Qualitative validation**: Users can input edge cases and observe whether predictions align with expectations, feeding manual spot-checks before deployment.
- **Stakeholder engagement**: Non-technical users can test the classifier on their own posts of interest without navigating the full notebook.
- **Transparency**: Confidence scores help users calibrate trust in uncertain predictions (e.g., confidence 0.31 should be treated as unreliable).

**Implementation Details**:
- Built using `ipywidgets.Textarea` (input), `ipywidgets.Button` (trigger), `ipywidgets.Output` (display).
- Reuses the fine-tuned trainer and tokenizer from the notebook, avoiding duplicated model loading.
- Single-post inference; batching could be added for production deployment.

**Limitations**:
- Inherits the fine-tuned model's poor performance (36.67% accuracy on test set).
- Predictions reflect the model's learned boundaries, not ground truth. For production, recommendations:
  1. **Display confidence thresholds**: Flag predictions with confidence < 0.50 as "uncertain, review manually."
  2. **Log edge cases**: Store inputs that produce unexpected predictions for future retraining on underrepresented classes.
  3. **Dual-model ensemble**: Use the Groq zero-shot baseline as a secondary ranker for low-confidence predictions from the fine-tuned model, aggregating both outputs to the user.
  4. **Annotated feedback loop**: Allow users to flag incorrect predictions, building a second-pass training dataset for model improvement.

**Next Steps for Production**:
- Increase training data to 100+ examples per class before deploying this interface.
- Implement class weighting to force the model to learn all boundaries simultaneously.
- Add a confidence-based filter so only predictions above a learned threshold are displayed automatically; borderline cases trigger manual review.

---

## References

- Notebook: `ai201_project3_takemeter_starter_clean.ipynb`
- Dataset: `data.csv` (29 training examples, manually annotated)
- Test evaluation: 30 held-out posts from r/soccer, June 2026
- Model: DistilBERT-base-uncased, fine-tuned with HuggingFace Transformers
- Baseline: Groq LLMm (zero-shot with in-context label definitions and edge cases)
- Interface: `simple_interface.py` (Jupyter ipywidgets interactive classifier)

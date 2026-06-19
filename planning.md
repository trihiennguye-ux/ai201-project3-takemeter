# Planning: 

## 1. Community

I chose **r/soccer** (8.7M members), the largest general-purpose soccer subreddit on Reddit. It functions mostly as a content aggregator: most submissions are headlines, transfer rumors, official club statements, goal clips, or links, rather than original writing. A smaller but recurring format is a community-run World Cup country-preview series, where individual users write up each of the 48 qualified teams.

This is a good fit for a classification task because the *topic* of a post tells you almost nothing about how substantive it is. A Ronaldo headline can produce nothing but one-line jokes, while a routine transfer rumor can produce a genuinely reasoned debate citing financial logic and historical precedent. Because the subreddit is global and high-volume, the same thread format reliably contains everything from drive-by banter to real argument to emotional fan moments to researched long-form writing — which is exactly the range needed for the labels below to actually discriminate between posts rather than just re-detecting the topic.

## 2. Labels

After reading through ~30 real posts and their comment threads, I settled on four labels, applied with a fixed precedence order so each post gets exactly one label:

1. **High-Effort Write-up** — a post whose own submission text contains specific researched or narrative content (named historical events, dates, quoted speech, a constructed storyline) rather than a templated stat box or a bare link, evaluated independently of whatever the comments do.
   - Example: a World Cup preview of DR Congo that traces the 52-year line from the 1974 Zaire campaign and Mwepu Ilunga's free-kick protest through to the exact qualifying goal that triggered a national holiday in 2026.
   - Example: a World Cup preview of Uruguay built around a direct quote from Marcelo Bielsa's two-hour press conference, framed against the Luis Suárez near-mutiny and a poor run of form.

2. **Analytical/Speculative Take** — checked only if (1) doesn't apply; a post where the comment section builds a claim about football using a specific piece of evidence, precedent, financial or tactical logic, or a cited rule, something a regular could agree or disagree with on the merits rather than just react to.
   - Example: an Olise-to-Real-Madrid transfer rumor thread where commenters cite the precedent of Madrid's failed Ribéry bid and argue that even a huge offer wouldn't help Bayern improve their squad.
   - Example: a Messi red-card discussion where a commenter distinguishes a DOGSO foul from violent conduct off the ball to argue two incidents aren't actually comparable.

3. **Human-Interest/Color** — checked only if (1) and (2) don't apply; a post centered on an emotional scene, fan reaction, or anecdote, where comments mostly respond with humor, empathy, or a parallel personal story rather than building an argument.
   - Example: a video of DR Congo fans celebrating in a Lisbon fan zone, with comments about "the joy of the World Cup."
   - Example: a Cape Verde goalkeeper's mother getting a US visa to attend his match, with comments riffing on football superstition and a parallel story about North Korea's 2010 World Cup.

4. **Drive-By Drop** — the default; neither the submission itself nor the comments add any elaboration beyond a one-line reaction to the headline.
   - Example: a player suspension announcement where the entire comment section is "Damn, thanks for coming" / "Early flight home."
   - Example: an official club statement where the comments are pure snark with no actual argument behind them.

## 3. Hard Edge Cases

Four configurations came up repeatedly enough during reading that they need a standing rule rather than a case-by-case judgment call:

**Thin entry inside an otherwise effortful series.** A World Cup preview for Germany had almost no content (no nickname, no stats, no narrative) but visible evidence of curatorial effort at the series level — the author credited a different contributor and explained their posting schedule. This is ambiguous between Drive-By Drop and High-Effort Write-up. *Rule: label strictly on the content of the post under review, not the reputation of the series it belongs to.* A thin entry stays Drive-By Drop even inside a high-effort series.

**Framing vs. outcome.** A linked deep-dive about a famous World Cup free kick was headlined as a poignant "sad story" (a Human-Interest signal), but the comment section turned into a real argument about whether the players' choice was justified given a dictator's threat against them (an Analytical signal). *Rule: label by what the comment thread actually does, not by how the post is framed.* The unit being measured is the discourse, not the headline's intent.

**Minimum bar for "reasoning."** A short transfer-speculation thread cited exactly one concrete fact (years remaining on a contract) to support a one-sentence claim, much thinner than the clearest Analytical examples. *Rule: at least one specific, checkable fact or named piece of reasoning is enough to qualify as Analytical, even if brief* — pure guesses or assertions with no cited fact stay Drive-By Drop. This keeps the bar binary and consistent rather than requiring a subjectively "developed" argument.

**Anecdote under a neutral headline.** A post about Haaland's goal registering on seismic sensors is a plain stat headline, not emotionally framed, but one comment is a personal anecdote ("I woke up my dog screaming") that doesn't analyze the claim, just echoes it. *Rule: classify by what's actually present in the comments, not by how neutral the headline was.* A personal anecdote anywhere in the thread is enough to earn Human-Interest/Color regardless of headline tone.

When a new configuration doesn't match any logged rule, I'll flag it for a second annotator before finalizing the label, rather than letting one person's gut call become a silent, unreviewed precedent for future similar posts.

## 4. Data Collection Plan

I'll pull posts directly from r/soccer's own listings (old.reddit `.json` pages or an equivalent export) across multiple sort orders — hot, top-day, top-week, and new — rather than just "hot," since hot alone overrepresents viral banter and underrepresents quieter analytical threads. I'll deliberately oversample from known high-effort formats (the World Cup preview series, posts flagged "OC," Athletic/Guardian-style deep-dive links), since High-Effort Write-up essentially never shows up in a generic random sample.

Target: roughly 30-50 examples per label for an initial 200-post set. I expect natural imbalance going in — Drive-By Drop will likely be the majority class in any unweighted sample, since most threads really are just banter, while High-Effort Write-up will be the rarest naturally occurring label.

If a label is still underrepresented after 200 examples, I won't pad it with borderline cases just to hit the quota — that would teach a model the wrong boundary. Instead: go targeted rather than random for that specific label (pull directly from the formats/flairs known to produce it), and if it still won't reach count even then, treat that as real information rather than a collection failure — it may mean the label is genuinely rare enough that it should be merged into a neighboring category, or that the definition needs slight loosening, rather than forcing artificial balance.

## 5. Evaluation Metrics

Accuracy alone is misleading here because the classes are imbalanced by nature (Drive-By Drop will dominate any real sample), so a model that just predicts the majority label everywhere would score deceptively well while being useless for the actual purpose of the tool. I'll use:

- **Macro-averaged F1** across all four classes (unweighted by frequency), so performance on the rare-but-important labels isn't hidden behind strong performance on the dominant Drive-By Drop class.
- **Per-class precision and recall**, with particular attention to recall on High-Effort Write-up and Analytical/Speculative Take. If the point of the tool is to surface substantive content, missing those (false negatives) is a worse failure than over-flagging a borderline Human-Interest post.
- **A confusion matrix**, to check *which* pairs the model confuses. From the edge-case analysis above, I already expect some genuine confusion between Human-Interest and Analytical, and between Drive-By and Analytical at the weak-reasoning boundary. Errors clustering there are acceptable; errors clustering somewhere unrelated (e.g., calling pure one-liner banter a High-Effort Write-up) would mean the model learned the wrong signal entirely.
- **Inter-annotator agreement** (e.g., Cohen's kappa) on a subset double-labeled by a second person, as a ceiling check. If two careful humans only agree on, say, 75–80% of cases at these known hard boundaries, expecting the model to do meaningfully better than that isn't a fair bar — model performance should be judged relative to human agreement, not relative to 100%.

## 6. Definition of Success

The tool's real job is helping someone scanning r/soccer find threads actually worth opening, so success isn't "matches my labels perfectly" — it's reliably promoting substantive content and reliably filtering pure banter. That means recall on High-Effort Write-up and Analytical/Speculative Take matters more than overall accuracy, even if it means occasionally over-including a borderline Human-Interest post.

"Good enough for deployment" looks like: macro-F1 in the 0.75–0.80 range, with recall on the two high-value classes (High-Effort Write-up, Analytical/Speculative Take) at or above whatever the human inter-annotator agreement turns out to be on those same classes. Asking for perfection on a boundary that two careful humans disagree on roughly a quarter of the time isn't a meaningful target. Concretely, I'd consider it deployable if it correctly recalls at least 80% of clear High-Effort Write-ups and Analytical Takes, and its confusion matrix shows mistakes clustering in the same boundary pairs already identified as genuinely hard — rather than in unrelated pairs that would suggest the model latched onto something other than the intended distinction.

## 7. AI Tool Plan
 
This project has no code to generate, so AI tools aren't used for implementation — they're used at three specific points where a second reader stress-tests my own judgment before it gets baked into 200 annotations or a final write-up.
 
**Label stress-testing.** Before annotating the full set, I'll give an AI tool (Claude) the four label definitions, the precedence order, and the four logged edge-case rules from Section 3, and ask it to generate 5–10 synthetic r/soccer-style posts (headline plus a few comments) deliberately written to sit at a specific boundary — e.g., one constructed to be ambiguous between Drive-By Drop and Analytical/Speculative Take, another between Human-Interest/Color and Analytical/Speculative Take. I'll then label each one myself using only the written definitions, without looking at which boundary the AI intended it to test. Any post I can't confidently assign a single label to — without inventing a new ad hoc rule on the spot — means the definition is still underspecified. I'll tighten the relevant definition or add a new resolution rule to Section 3, then re-run this check on a fresh batch. One caveat I'll keep in mind: AI-generated posts may not perfectly match real Reddit phrasing or structure, so this checks the internal consistency of the *definitions*, not a substitute for reading real posts during actual data collection.
 
**Annotation assistance.** I will use an LLM (Claude) to pre-label batches of posts before I review them, rather than annotating all 200 cold, given the time this would otherwise take. Each batch (roughly 20–30 posts) gets the full label definitions, precedence rule, and edge-case log as instructions, plus the post's full text (headline and comments), and returns one label per post. For disclosure, every example in the dataset gets two extra columns: `annotation_source` (`ai_prelabel_reviewed` or `human_only`, plus the model name/version used) and `overridden` (whether my review changed the AI's suggested label, and what the AI originally proposed). To guard against just rubber-stamping the AI's suggestion, I'll do a blind spot-check every ~20 examples: label a handful myself before looking at the pre-label, then compare, to get a rough read on how much anchoring bias is creeping into the review process.
 
**Failure analysis.** After training and evaluating the model on held-out data, I'll compile the list of misclassified examples (true label, predicted label, full post text) and give it to an AI tool to look for patterns: which label pairs get confused most often, whether errors cluster around the same hard boundaries already logged in Section 3, and whether there's a spurious heuristic at play — e.g., the model defaulting to Drive-By Drop on any short thread regardless of content, or to High-Effort Write-up on any long post regardless of whether it's actually narrative versus just a long templated list. I will not take the AI's claimed patterns at face value: for each pattern it proposes, I'll spot-check a random sample of the specific examples it cites by re-reading the actual post text myself, and I'll independently re-derive any quantitative claim (e.g., actually cross-tabulating errors against post length or comment count rather than trusting a qualitative description of the trend). Only patterns I can confirm this way go into the evaluation write-up — the AI's output here is treated as a list of hypotheses to check, not a finding to report directly.

---

## 8. Error Pattern Analysis (Extra Feature)
 
After evaluating the fine-tuned model on 30 held-out test examples (66.67% accuracy, 10/30 misclassified), I identified three systematic error patterns that go beyond individual misclassifications:
 
**Pattern 1: Human-Interest/Color → Drive-By Drop Misclassification (5/5 Human-Interest errors, 100% of the class)**
Every true Human-Interest post in the test set was predicted Drive-By Drop, several at high confidence. Examples: "I am both super hyped for their game this afternoon, and extremely anxious LOL" (predicted Drive-By Drop, confidence 0.85); England and Croatia fans booing at the hydration break, comments about singing "Mr Brightside" (predicted Drive-By Drop, confidence 0.81); "Unbelievable scenes in DR Congo... This is what its all about man" (predicted Drive-By Drop, confidence 0.57).
- **Root cause**: the model appears to have learned "short, casually-phrased post with no cited evidence" as the defining feature of Drive-By Drop — which also describes most Human-Interest posts, since emotional reactions and anecdotes are usually short and casual too. Training had 21 Human-Interest examples vs. 50 Drive-By Drop examples, so the model defaults to the larger class whenever a post has Drive-By Drop's surface features (brevity, casual tone), regardless of whether it also carries Human-Interest's actual signal (personal/emotional language). The high confidence on several of these (0.81, 0.85) shows this isn't hesitation — it's a confidently learned, wrong rule.
**Pattern 2: Analytical/Speculative Take → Drive-By Drop Misclassification (3/10 Analytical errors)**
Posts that open with a snappy, banter-like headline but build a real argument in the comments are misclassified as Drive-By Drop. Examples: "England [1] - 0 Croatia - Harry Kane 12'... Stop the stutter mate" (predicted Drive-By Drop, confidence 0.49); "5. Elye Wahi arrested for alleged fixing offences..." with a comment citing the Emmanuel Clase MLB precedent (predicted Drive-By Drop, confidence 0.46); "All the patches on Messi's shirt and their meanings" (predicted Drive-By Drop, confidence 0.63).
- **Root cause**: the model appears to weight the headline's casual tone heavily and under-weight the comment section, missing that Analytical requires *reasoning about* evidence in the comments, not a serious-sounding headline. This is the mirror image of Pattern 1 — both directions funnel into Drive-By Drop, the largest training class (50/140).
**Pattern 3: Drive-By Drop → Analytical/Speculative Take Misclassification (2/10 Drive-By Drop errors)**
Posts containing a sourced factual citation that's actually just a standalone trivia drop, not evidence for an argument, are misclassified as Analytical. Examples: "[FotMob] Luis Díaz is the second Colombian since 1962... to score and assist" (predicted Analytical, confidence 0.61); "Iraq 0-1 Norway - Erling Haaland 29'... Give the man a sniff" (predicted Analytical, confidence 0.54).
- **Root cause**: the model treats any number/date/named-source citation as a strong Analytical signal on its own, without checking whether the comments actually reason about that fact or just react to it. This is the smallest of the three patterns (2/30 test examples), confirming the Drive-By ↔ Analytical boundary is mostly, if imperfectly, learned — unlike the Human-Interest boundary, which failed completely.
**Confidence Is Not Uniformly Low**: confidence on the 10 errors ranges from 0.44 to 0.85, not clustered near the 0.25 random-chance baseline for 4 classes. The highest-confidence errors (0.81, 0.85) are both Human-Interest posts pulled into Drive-By Drop — the model is confidently applying a learned-but-wrong rule there, not hedging. Lower-confidence errors (0.44–0.61) cluster around posts that genuinely mix signals (a trivia-style opening line attached to either an emotional payload or a real argument).
 
**Summary**: the fine-tuned model treats "short and casually phrased" as the defining signature of Drive-By Drop, which correctly separates most true Drive-By Drop posts from Analytical ones but incorrectly swallows the entire Human-Interest class, since Human-Interest posts share Drive-By Drop's brevity and casual tone. High-Effort Write-up, by contrast, is learned perfectly (5/5) despite having the smallest training set (20 examples) — its distinctive World Cup preview template gives the model a clean signal that the Human-Interest/Drive-By boundary lacks. This points to a targeted fix (more Human-Interest examples specifically near the Drive-By boundary, or a sharper definitional rule between the two) rather than a generic "more data across all classes" fix.
 
---
 
## 9. Extended Features: Interactive Classification Interface
 
**Simple Interactive Interface (simple_interface.py):** A Jupyter notebook widget interface allowing users to classify new posts in real-time without re-running evaluation cells. Users enter a post in a textarea, click a "Classify Post" button, and receive the predicted label and confidence score. This enables:
 
- **Rapid testing** of new posts from r/soccer without manual annotation overhead.
- **Qualitative debugging** — users can input edge cases and observe whether the model's predictions align with expectations, feeding manual spot-checks for robustness before deployment.
- **Exploration and validation** — stakeholders can test the classifier on their own posts of interest before committing to production use.
**Implementation**: Built using `ipywidgets.Textarea`, `ipywidgets.Button`, and `ipywidgets.Output` in a Jupyter environment. Reuses the fine-tuned trainer and tokenizer from the notebook, avoiding duplicated code. Returns both label and confidence to help users calibrate trust in uncertain predictions.
 
**Limitations**: Relies on the fine-tuned model's performance (36.67% accuracy on test set), so predictions reflect the model's learned boundaries, not ground truth. For production use, recommendations would be:
- Display confidence thresholds so users can filter uncertain predictions (e.g., confidence < 0.50 → "uncertain, review manually").
- Log edge-case inputs for future retraining on under-represented classes.
- Use the Groq zero-shot baseline as a secondary ranker for predictions with low confidence from the fine-tuned model.
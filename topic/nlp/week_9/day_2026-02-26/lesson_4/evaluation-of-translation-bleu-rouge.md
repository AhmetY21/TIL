---
title: "Evaluation of Translation (BLEU, ROUGE)"
date: "2026-02-26"
week: 9
lesson: 4
slug: "evaluation-of-translation-bleu-rouge"
---

# Topic: Evaluation of Translation (BLEU, ROUGE)

## 1) Formal definition (what is it, and how can we use it?)

BLEU (Bilingual Evaluation Understudy) and ROUGE (Recall-Oriented Understudy for Gisting Evaluation) are automatic evaluation metrics used to assess the quality of machine translation (MT) or text summarization outputs. They work by comparing the generated text (candidate) against one or more reference translations or summaries produced by humans.

**BLEU:**

*   **What it is:** BLEU measures the *precision* of the candidate translation by counting the number of n-grams (contiguous sequences of n words) in the candidate that also appear in the reference translations.  It also incorporates a brevity penalty to penalize translations that are too short.
*   **How we can use it:**
    *   Evaluate and compare different MT systems or models.
    *   Track the progress of MT system development.
    *   Automatically optimize model parameters during training (though relying solely on BLEU can be problematic, leading to over-optimization for specific metrics).
    *   Get a quantitative assessment of the similarity between generated and reference text.
*   **Key aspects:**
    *   **N-gram precision:** Calculates the percentage of n-grams in the candidate that appear in any of the references.  Typically, n-grams up to length 4 (BLEU-4) are used.
    *   **Brevity penalty:** Prevents systems from achieving high scores simply by generating very short translations that happen to match the reference.  It penalizes candidates shorter than the effective reference length (the length of the reference that is closest in length to the candidate).
    *   **Modified precision:** To prevent inflated scores from repeating words frequently, precision is modified to count a word only as many times as it appears in the *best matching* reference.

**ROUGE:**

*   **What it is:** ROUGE, in contrast to BLEU, focuses on *recall*. It measures how much of the reference text is present in the candidate translation/summary.  There are several ROUGE variants, including ROUGE-N, ROUGE-L, and ROUGE-SU.
*   **How we can use it:**
    *   Evaluate and compare text summarization systems.
    *   Assess the quality of generated text, particularly in terms of its completeness and information content compared to a reference.
    *   Analyze the effectiveness of different summarization techniques.

*   **Key aspects (for common ROUGE variants):**
    *   **ROUGE-N:** Measures the recall of n-grams.  ROUGE-1 (unigrams) and ROUGE-2 (bigrams) are commonly used.
    *   **ROUGE-L:** Measures the longest common subsequence (LCS) between the candidate and reference, effectively capturing sentence-level structure and fluency.
    *   **ROUGE-SU:** Measures the recall of skip-bigrams (pairs of words that can be separated by other words), allowing for some flexibility in word order and capturing semantic relationships.

In essence, BLEU asks "How much of the candidate is relevant?" (precision), while ROUGE asks "How much of the reference is covered by the candidate?" (recall). They provide different perspectives on translation or summarization quality, and are often used together for a more comprehensive evaluation.

## 2) Application scenario

**BLEU Application Scenario: Evaluating Machine Translation Quality**

Imagine you are developing a machine translation system to translate English news articles into French.  You have several versions of your model (e.g., different neural network architectures, different training datasets). To determine which model performs best, you can use BLEU.  You would:

1.  Translate a set of English news articles (the "test set") into French using each model.
2.  Obtain human-translated French versions of the same articles (the "reference translations").
3.  Calculate the BLEU score for each model's output by comparing it to the reference translations.

The model with the higher BLEU score is generally considered to produce better translations, at least according to the BLEU metric.

**ROUGE Application Scenario: Evaluating Text Summarization Quality**

Consider you are building a system to automatically summarize lengthy research papers. You want to assess how well your system captures the key information from the original papers. You would:

1.  Use your summarization system to generate summaries of a collection of research papers.
2.  Obtain human-written summaries of the same papers (the "reference summaries").
3.  Calculate the ROUGE score (e.g., ROUGE-L, ROUGE-1) for each generated summary by comparing it to the reference summaries.

The system with the higher ROUGE score is considered to produce summaries that are more complete and capture a greater proportion of the important information present in the reference summaries.

## 3) Python method (if possible)

Both BLEU and ROUGE are readily available in Python libraries like `nltk` and `rouge-score`.  Here's how to use them:

**BLEU using `nltk`:**

```python
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

reference = [['this', 'is', 'a', 'test'], ['this', 'is', 'a', 'test']]  # Example reference translations
candidate = ['this', 'is', 'a', 'test']  # Example candidate translation

# Smoothing is important to avoid zero scores when n-grams are missing
smoothing_function = SmoothingFunction().method1

score = sentence_bleu(reference, candidate, smoothing_function=smoothing_function)
print(f"BLEU score: {score}")

# Calculating BLEU-4 score (default is BLEU-1)
weights = (0.25, 0.25, 0.25, 0.25) # weights for 1-gram, 2-gram, 3-gram and 4-gram
score_bleu4 = sentence_bleu(reference, candidate, weights=weights, smoothing_function=smoothing_function)
print(f"BLEU-4 score: {score_bleu4}")
```

**ROUGE using `rouge-score`:**

```python
from rouge_score import rouge_scorer

reference = "The quick brown fox jumps over the lazy dog."  # Example reference summary
candidate = "The quick brown fox jumps over the lazy dog."  # Example candidate summary

scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
scores = scorer.score(reference, candidate)

print(f"ROUGE scores: {scores}")
# Example output:
# {'rouge1': Score(precision=1.0, recall=1.0, fmeasure=1.0), 'rougeL': Score(precision=1.0, recall=1.0, fmeasure=1.0)}
```

**Explanation:**

*   **`nltk.translate.bleu_score.sentence_bleu`**:  Calculates the BLEU score for a single sentence.  It takes a list of reference translations (each a list of words) and a candidate translation (also a list of words) as input. The `SmoothingFunction` is crucial to handle cases where the candidate translation doesn't contain all the n-grams present in the reference(s).
*   **`rouge_score.rouge_scorer.RougeScorer`**: Creates a ROUGE scorer object, specifying which ROUGE variants to compute (e.g., 'rouge1', 'rougeL'). The `use_stemmer=True` option applies stemming to improve matching.
*   **`scorer.score`**: Calculates the ROUGE scores, returning a dictionary containing precision, recall, and F1-measure (fmeasure) for each specified ROUGE variant.

## 4) Follow-up question

While BLEU and ROUGE are widely used, they have limitations. For example, they primarily assess lexical similarity and might not capture semantic similarity or fluency well. What are some other, more recent, evaluation metrics that attempt to address these limitations in evaluating machine translation or text summarization?
# QA Evaluation Set

This file is a small manual evaluation set for the research-agent QA pipeline.

Use it to compare behavior before and after targeted retrieval / rendering changes.

## How to use

- Run the question against the named paper.
- Capture:
  - final answer
  - retrieved top-20
  - reranked top-20
  - selected context
  - equation candidates / table artifacts if applicable
- Mark:
  - `pass`
  - `partial`
  - `fail`
- Add notes on:
  - missing evidence
  - noisy evidence
  - bad formatting
  - incorrect follow-up resolution

## Acceptance themes

- General QA should remain stable.
- Follow-up questions should preserve the prior referent.
- Formula questions should surface canonical equations rather than fragments.
- Table questions should surface actual table-backed evidence rather than generic prose.
- Rendering should remain readable.

## Cases

### 1. LoRA

Paper:
- `LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS`

General:
- `How is LoRA better than retraining all the weights?`
- `What problem is LoRA trying to solve?`

Follow-up:
- `What's the mathematical expression?`
- `How does it work?`
- `Where does the paper say that?`

Formula:
- `Give any formulas in the paper.`
- `What is the mathematical formulation of LoRA?`
- `What equation defines the LoRA update?`
- `What does h = W0x + ΔWx become?`

Table:
- `Are there any tables in the paper?`
- `Show me the results table.`
- `What does Table 3 contain?`
- `What metrics are reported in the tables?`

Expected focus:
- canonical LoRA equations
- low-rank update definition
- actual benchmark/result tables rather than hyperparameter rows

### 2. Conformer

Paper:
- `Conformer: Convolution-augmented Transformer for Speech Recognition`

General:
- `What is the core idea of Conformer?`
- `How is it different from a vanilla Transformer?`

Follow-up:
- `What's the equation?`
- `How does it work?`

Formula:
- `What formulas or equations are used in the paper?`
- `What is the Conformer block equation?`

Table:
- `Are there any result tables in the paper?`
- `What does the main results table show?`

Expected focus:
- actual Conformer block equations if present
- evaluation tables / WER comparisons

### 3. Mask-Conformer

Paper:
- `MASK-CONFORMER: AUGMENTING CONFORMER WITH MASK-PREDICT DECODER`

General:
- `What does Mask-Conformer add to Conformer?`

Formula:
- `Are there any equations in the paper?`
- `Show the mathematical expressions if they exist.`

Table:
- `Show me the results table.`
- `What does Table 1 contain?`

Expected focus:
- decoder / mask-predict details
- table-backed performance evidence

### 4. Small Language Models are the Future of Agentic AI

Paper:
- `Small Language Models are the Future of Agentic AI`

General:
- `What is the main argument of the paper?`
- `What evidence does the paper use to support that claim?`

Follow-up:
- `Where does the paper say that?`
- `How does it work?`

Formula:
- `Are there any mathematical expressions in this paper?`

Table:
- `Are there any tables in the paper?`
- `What results are shown in the tables?`

Expected focus:
- likely little or no formal math
- system should not hallucinate equations
- should still detect tables if present

## Scoring template

For each run, record:

- `paper`:
- `question`:
- `category`: `general | follow_up | formula | table`
- `status`: `pass | partial | fail`
- `retrieval_issue`: `none | noisy_top20 | rerank | missing_artifact | formatting`
- `notes`:

## Current known checks

- LoRA formula answer should now return:
  - `W0 + ΔW = W0 + BA`
  - `W = W0 + BA`
  - `h = W0x + ΔWx = W0x + BAx`
- LoRA chunk-level formula top-20 is still expected to be noisy today.
- Table questions should now surface actual table-backed evidence when the paper contains tables.
- If a paper has no real tables, the QA response should say so instead of fabricating one.

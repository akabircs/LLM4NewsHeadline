This repository contains the cleaned, anonymous source code accompanying the manuscript [**“When Less Is Enough: Context Selection and Prompting Strategies for Bengali News Headline Generation.”**](https://arxiv.org/abs/2608.15879)

The code was consolidated from the experimental notebooks into parameterized scripts so that the paper’s analyses can be audited without exposing API credentials, notebook outputs, personal paths, or platform-specific metadata.


## Experimental settings

| Component | Setting |
|---|---|
| Gemini | `gemini-2.0-flash` |
| Llama | `meta-llama/Llama-3.3-70B-Instruct-Turbo-Free` via Together AI |
| GPT-4o | `gpt-4o` |
| Temperature | `0.7` |
| Top-p | `0.9` |
| BeliN max output | `50` tokens |
| Shironaam max output | `600` tokens |
| Semantic-alignment model | `banglagov/banBERT-Base` |
| Similarity | cosine similarity |
| Token-count proxy | `o200k_base` |

The fixed few-shot demonstrations used in the supplied experiments are stored in `config/belin_demonstrations.json` and `config/shironaam_demonstrations.json`.

## Repository layout

```text
config/
  experiment_settings.yaml
  belin_demonstrations.json
  shironaam_demonstrations.json
data/
  README.md
scripts/
  semantic_alignment.py
  token_analysis.py
  run_belin.py
  run_shironaam.py
  evaluate.py
  paired_statistics.py
  scan_secrets.py
src/
  prompts.py
  providers.py
  utils.py
.env.example
.gitignore
requirements.txt
```

## Paper-to-code map

| Paper result | Reproduction code |
|---|---|
| Main Table 1 / Figure 2: BeliN headline–paragraph alignment | `scripts/semantic_alignment.py --dataset belin` |
| Main Table 2: context-scope comparison | `scripts/run_belin.py` with `--context 1..5/full`, BNaP + Baseline, 0-shot, Gemini/Llama |
| Main Table 3: input-token reduction | `scripts/token_analysis.py` |
| Main Table 4: ICL 0–3 shot | `scripts/run_belin.py --context 3 --language bnap --variant baseline --shots 0..3`, Gemini/Llama |
| Main Table 5: BNaP/XLP × Baseline/MultiGen | `scripts/run_belin.py --context 3 --shots 3`, all prompt combinations and reported models |
| Main Table 6 | Uses evaluated results from the best reported configuration; prior-model values are literature scores and are not regenerated here |
| Main Table 7: Shironaam second benchmark | `scripts/run_shironaam.py` + `scripts/evaluate.py` |
| Supplementary Table S1 / Figure S1: Shironaam alignment | `scripts/semantic_alignment.py --dataset shironaam` |
| BCa CIs / Wilcoxon comparisons | `scripts/evaluate.py` and `scripts/paired_statistics.py` |

## Installation

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
cp .env.example .env
```

## Data

See `data/README.md`. BeliN can be loaded from Hugging Face by the scripts. Shironaam is not redistributed; pass the local train/test CSV obtained from the source cited in the manuscript.

## Reproduction examples

### 1. BeliN semantic-alignment analysis — Main Table 1 / Figure 2

```bash
python scripts/semantic_alignment.py \
  --dataset belin \
  --output-dir outputs/belin_alignment
```

The command writes:

- `alignment_rows.csv`
- `position_distribution.csv`
- `paragraph_similarity_scores.csv`

### 2. Main Table 2 — context-scope experiment

Run each context scope for Gemini and Llama. Example:

```bash
python scripts/run_belin.py \
  --provider gemini \
  --language bnap \
  --variant baseline \
  --shots 0 \
  --context 1 \
  --output outputs/table2/gemini_context1.csv
```

Repeat `--context` for `1`, `2`, `3`, `4`, `5`, and `full`; repeat with `--provider llama`.

### 3. Main Table 3 — token reduction

```bash
python scripts/token_analysis.py \
  --input data/belin_test.csv \
  --k 3 \
  --output outputs/table3_token_reduction.csv
```

### 4. Main Table 4 — zero- to three-shot ICL

```bash
python scripts/run_belin.py \
  --provider gemini \
  --language bnap \
  --variant baseline \
  --context 3 \
  --shots 1 \
  --output outputs/table4/gemini_1shot.csv
```

Repeat `--shots 0`, `1`, `2`, and `3` for Gemini and Llama.

### 5. Main Table 5 — prompting strategies

Example for Gemini XLP+MultiGen, three-shot:

```bash
python scripts/run_belin.py \
  --provider gemini \
  --language xlp \
  --variant multigen \
  --context 3 \
  --shots 3 \
  --output outputs/table5/gemini_xlp_multigen.csv
```

Run the BNaP/XLP × Baseline/MultiGen combinations for the models reported in the manuscript.

### 6. Shironaam second benchmark — Main Table 7

```bash
python scripts/run_shironaam.py \
  --input data/shironaam_test.csv \
  --output outputs/table7/shironaam_gemini_xlp_multigen_3shot.csv
```

This script intentionally uses the experiment-specific settings reported in the manuscript: first three paragraphs, XLP+MultiGen, three demonstrations, temperature `0.7`, top-p `0.9`, and `max_output_tokens=600`.

### 7. Automatic evaluation

```bash
python scripts/evaluate.py \
  --input outputs/table7/shironaam_gemini_xlp_multigen_3shot.csv \
  --output-dir outputs/table7/evaluation
```

### 8. Paired statistical comparison

```bash
python scripts/paired_statistics.py \
  --a outputs/condition_a/evaluation/instance_scores.csv \
  --b outputs/condition_b/evaluation/instance_scores.csv \
  --metric BERTScore_F1 \
  --output outputs/paired_comparison.json
```

## Notes on the Shironaam MultiGen prompt

The auxiliary cues differ across datasets, matching the manuscript:

- **BeliN:** sentiment, category, aspect.
- **Shironaam:** topic words/tags, category, image caption.

The Shironaam generation script uses the first three article paragraphs and the fixed three-shot demonstrations stored in `config/shironaam_demonstrations.json`.

## Important reproducibility note

Hosted proprietary/API-served models can change over time even when a stable family identifier is used. This repository records the identifiers and decoding settings present in the supplied experimental code, but exact future numerical replication may also depend on provider-side model/version availability.

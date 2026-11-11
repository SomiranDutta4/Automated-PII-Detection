# Automated PII Detection

An NLP project for identifying personally identifiable information (PII) in unstructured text using token-level sequence labelling. The project brings together heterogeneous PII and named-entity recognition datasets, maps their annotations into a shared BIO label space, and fine-tunes **DeBERTa-v3-base** for entity recognition.

The implementation is organised around three main stages:

1. **Dataset exploration and collection** from multiple Hugging Face datasets.
2. **Dataset consolidation and preparation**, including label normalisation, BIO tagging, rare-entity filtering and source-stratified splitting.
3. **Transformer fine-tuning**, with a standard token-classification trainer and an experimental training variant.

> **Reported result:** 64.55% F1. This is the result stated in the project resume summary. The repository archive does not include a saved evaluation report or model checkpoint that independently verifies this number. See [Evaluation and reported results](#evaluation-and-reported-results).

## Table of Contents

- [Problem Statement](#problem-statement)
- [Project Workflow](#project-workflow)
- [Datasets](#datasets)
- [Label Harmonisation and BIO Annotation](#label-harmonisation-and-bio-annotation)
- [Data Preparation](#data-preparation)
- [Model and Training](#model-and-training)
- [Evaluation and Reported Results](#evaluation-and-reported-results)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Running the Pipeline](#running-the-pipeline)
- [Input and Output Format](#input-and-output-format)
- [Implementation Notes and Limitations](#implementation-notes-and-limitations)

## Problem Statement

PII can appear in free-form text in many forms: names, contact details, financial identifiers, addresses, account information and other sensitive attributes. These entities are difficult to detect reliably with fixed patterns alone because their representation and surrounding context vary across documents and domains.

This project formulates PII detection as a **token-level sequence-labelling task**. Given a sequence of input tokens, the model predicts a BIO label for each token. The resulting labels identify the boundaries and types of entities in the text.

The training corpus combines PII-focused datasets with general-purpose NER and finance-domain data. Since these sources use different annotation schemes and entity taxonomies, their labels must be mapped into a common representation before they can be used together.

## Project Workflow

```text
Hugging Face datasets
        |
        v
Dataset download and inspection
        |
        v
Per-source JSONL files
        |
        v
Label normalisation + BIO conversion
        |
        v
Unified consolidated JSONL corpus
        |
        v
Data filtering and source-stratified splitting
        |
        +------> train.jsonl
        +------> val.jsonl
        +------> test.jsonl
        +------> evaluation subsets
        |
        v
DeBERTa-v3 token classification
        |
        v
Fine-tuned model checkpoint
```

The repository also contains an experimental trainer (`train_novel.py`) that adds optional source conditioning, curriculum learning and a hierarchical classification head. These are separate experimental capabilities; they should not be assumed to have been used for the 64.55% result unless the corresponding run configuration and output are available.

## Datasets

The dataset collection code covers ten sources. They provide a mixture of direct PII annotations, conventional NER labels and finance-domain entity annotations.

| Dataset | Main contribution |
|---|---|
| `ai4privacy/pii-masking-400k` | Synthetic PII examples with a broad range of personal-data categories |
| `ai4privacy/pii-masking-300k` | General PII data, including the FinPII finance/insurance subset |
| `gretelai/synthetic_pii_finance_multilingual` | Synthetic financial documents and finance-related identifiers |
| `nvidia/Nemotron-PII` | PII annotations across multiple document and industry contexts |
| `wikiann` (English) | General NER examples, including person, organisation and location entities |
| `Babelscape/multinerd` (English) | General-purpose multilingual NER data; the collection code filters to English |
| `DFKI-SLT/few-nerd` (supervised) | Fine-grained general NER categories |
| `conll2003` | Conventional newswire NER data |
| `nlpaueb/finer-139` | Financial-domain entity annotations from financial filings |
| `Isotonic/pii-masking-200k` | Additional PII examples in a BIO-labelled format |

The source code uses `Isotonic/pii-masking-200k` as the tenth dataset. The `pii_datasets.ipynb` notebook contains an older/mismatched section heading referring to `iiiorg/piiranha-v1`; the executable collection cell actually loads the Isotonic dataset.

The datasets differ in language coverage, annotation method, domain and licence. Review the original dataset cards and licences before redistributing the combined corpus or using it commercially. The project code does not itself grant rights to the source datasets.

## Label Harmonisation and BIO Annotation

The consolidation script (`srcripts/consolidate_pii_datasets.py`) converts source-specific annotations into a common token-and-label representation.

### BIO representation

Each token receives one label:

- `B-<ENTITY>` — first token of an entity.
- `I-<ENTITY>` — subsequent token(s) belonging to the same entity.
- `O` — token outside an annotated entity.

For example:

| Token | BIO label |
|---|---|
| `Contact` | `O` |
| `Alice` | `B-PERSON` |
| `Smith` | `I-PERSON` |
| `at` | `O` |
| `alice@example.com` | `B-EMAIL` |

### Normalisation

The `LABEL_NORM` mapping harmonises source labels into canonical entity names. For example, person labels such as `PER`, `FIRSTNAME` and `LASTNAME` are mapped to `PERSON`; organisation and location variants are mapped to shared categories; financial identifiers such as credit-card and account-number labels are normalised into consistent forms.

The converter handles more than one annotation format:

- Existing BIO sequences and integer-encoded NER labels.
- Character-offset spans in raw text.
- Span annotations stored as JSON strings or nested lists.
- Dataset-specific formats, including special handling for Few-NERD and NVIDIA Nemotron-PII.

For span-based records, text is whitespace-tokenised and character offsets are mapped to token indices. The converter also includes fallback handling for slightly misaligned offsets and span values that can be located in the source text.

The consolidated JSONL record has the following structure:

```json
{
  "tokens": ["Alice", "works", "at", "Example", "Bank"],
  "labels": ["B-PERSON", "O", "O", "B-ORG", "I-ORG"],
  "source": "conll2003"
}
```

When the optional `--include-text` flag is used, a `text` field is also retained.

## Data Preparation

`srcripts/data_preparation.py` applies the preparation rules to the consolidated corpus.

### Filtering and balancing

- **FINER-139 cap:** limits the `finer_139` source to at most 150,000 records.
- **Rare entity filtering:** counts entity mentions using `B-` labels. Entity types with fewer than 500 mentions are removed by converting their labels to `O`.
- **Nemotron cleanup:** retains PII-bearing Nemotron records after rare-label filtering and adjusts the non-Nemotron pool so Nemotron contributes approximately 10% of the resulting records.
- **Source-stratified splitting:** splits records independently within each source to preserve source proportions.

### Dataset splits

The configured split is 80% training, 10% validation and 10% test, using random seed `42`.

The preparation script also creates smaller source-stratified subsets:

| File | Purpose |
|---|---|
| `data/train.jsonl` | Training records |
| `data/val.jsonl` | Full validation split |
| `data/test.jsonl` | Held-out test split |
| `data/val_1p.jsonl` | Approximately 1% of validation data for faster training-time evaluation |
| `data/test_1p.jsonl` | Approximately 1% of test data for quick checks |
| `data/test_5k.jsonl` | 5,000-record test subset intended for benchmark comparisons |
| `data/label_mapping.json` | Label-to-ID and ID-to-label mappings, plus retained and dropped entity types |

The exact number of records depends on the downloaded source dataset versions and the filtering outcome. The script prints the resulting counts when it runs.

## Model and Training

### Base model

The standard training script fine-tunes **`microsoft/deberta-v3-base`** using the Hugging Face Transformers token-classification interface.

The model receives tokenised text and predicts a label distribution for each token. Word-level BIO labels are aligned to the subword tokens produced by the DeBERTa tokenizer. Special tokens and non-first subword pieces are assigned the ignore index (`-100`) so they do not contribute to the token-classification loss.

### Training configuration

The defaults defined in `srcripts/train.py` are:

| Parameter | Default |
|---|---:|
| Base model | `microsoft/deberta-v3-base` |
| Per-device training batch size | 16 |
| Per-device evaluation batch size | 32 |
| Gradient accumulation steps | 4 |
| Effective batch size per device | 64 |
| Learning rate | `2e-5` |
| Epochs | 10 |
| Maximum sequence length | 256 |
| Warmup ratio | 0.06 |
| Weight decay | 0.01 |
| Evaluation interval | 2,000 steps |
| Checkpoint save interval | 2,000 steps |
| Early-stopping patience | 3 |

The training script uses the Hugging Face `Trainer`, checkpoint selection based on validation F1 and early stopping. It supports resuming from a checkpoint and gradient checkpointing.

**Class-weighted loss:** the standard trainer includes a custom token-classification loss implementation with class weights. This is intended to reduce the dominance of the outside (`O`) class relative to entity labels. The exact weights and any additional options should be taken from the command-line configuration used for a particular experiment.

### Experimental trainer

`srcripts/train_novel.py` contains optional extensions:

1. **Source conditioning:** adds a source-specific special token (for example, `[SRC=wikiann]`) to the input sequence.
2. **Curriculum learning:** trains in three source-family phases: general NER, synthetic PII and financial PII.
3. **Hierarchical classification:** adds a coarse entity-group prediction objective alongside fine-grained BIO classification.

The experimental trainer is not identical to the standard trainer. Its options are activated through command-line flags, and its saved model/configuration should be identified separately from a standard DeBERTa token-classification run.

## Evaluation and Reported Results

The resume summary reports an F1 score of **64.55%** for the fine-tuned DeBERTa-v3-base model.

The repository's standard training code computes sequence-labelling precision, recall and F1 using `seqeval`. It removes positions labelled `-100` before metric calculation and uses the predicted BIO sequences and reference BIO sequences to calculate the metrics.

| Metric | Reported value |
|---|---:|
| F1 | 64.55% |

Precision and recall values are not stated in the supplied project summary, so they are intentionally not estimated here.

**Reproducibility note:** the supplied ZIP contains training and data-preparation code, but does not contain the generated data splits, a trained checkpoint, a run log or a saved test-metrics file. Consequently, the 64.55% figure is documented as a reported project result rather than a value independently recomputed from the archive.

The preparation code creates `test_5k.jsonl` for a fixed-size held-out subset. However, the supplied repository does not include a dedicated baseline benchmarking runner that compares the fine-tuned model against spaCy, Presidio or other baseline systems. The resume statement about a benchmarking pipeline should only be retained if that pipeline exists in another project version or can be supplied separately.

## Repository Structure

```text
Automated-PII-Detection/
├── notebooks/
│   ├── EDA.ipynb
│   ├── pii_datasets.ipynb
│   └── merge_datasets.py
├── srcripts/
│   ├── download_datasets.py
│   ├── consolidate_pii_datasets.py
│   ├── data_preparation.py
│   ├── download_model.py
│   ├── train.py
│   ├── train_novel.py
│   ├── inference.py
│   └── exceptions.py
└── requirements.txt
```

Note: `srcripts` is the directory name in the supplied repository archive. It is preserved as-is in the commands below. Renaming it to `scripts` would require updating imports and paths accordingly.

### Main files

| File | Responsibility |
|---|---|
| `notebooks/EDA.ipynb` | Exploratory analysis of the collected data |
| `notebooks/pii_datasets.ipynb` | Dataset inspection, metadata summary and source-level exploration |
| `notebooks/merge_datasets.py` | Earlier consolidation implementation retained in the notebooks directory |
| `srcripts/download_datasets.py` | Downloads and saves the ten source datasets |
| `srcripts/consolidate_pii_datasets.py` | Converts heterogeneous source annotations into a unified BIO corpus |
| `srcripts/data_preparation.py` | Applies source-specific filtering, rare-label filtering and dataset splitting |
| `srcripts/download_model.py` | Downloads the DeBERTa-v3-base model and tokenizer |
| `srcripts/train.py` | Standard DeBERTa token-classification training |
| `srcripts/train_novel.py` | Experimental source-conditioned/curriculum/hierarchical trainer |
| `srcripts/inference.py` | Inference and redaction utilities for a saved model |
| `srcripts/exceptions.py` | Custom exception classes used by inference |

## Installation

Python 3.10 or later is recommended. Model training requires a compatible PyTorch installation and is substantially faster with a CUDA-capable GPU.

Clone the repository:

```bash
git clone https://github.com/SomiranDutta4/Automated-PII-Detection.git
cd Automated-PII-Detection
```

Create and activate a virtual environment:

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

For GPU training, install the PyTorch build compatible with the local CUDA environment. Refer to the [official PyTorch installation selector](https://pytorch.org/get-started/locally/) rather than assuming that the default pip package matches the machine.

## Running the Pipeline

Run commands from the repository root. The scripts use relative paths such as `./pii_datasets`, `./data` and `./models`.

### 1. Download datasets

```bash
python srcripts/download_datasets.py
```

The script downloads the configured Hugging Face datasets and writes their splits under `pii_datasets/`.

### 2. Consolidate annotations

```bash
python srcripts/consolidate_pii_datasets.py \
  --data-dir ./pii_datasets \
  --output-dir ./pii_datasets/consolidated
```

To retain source text in the consolidated records:

```bash
python srcripts/consolidate_pii_datasets.py \
  --data-dir ./pii_datasets \
  --output-dir ./pii_datasets/consolidated \
  --include-text
```

### 3. Prepare train, validation and test splits

```bash
python srcripts/data_preparation.py
```

The preparation script expects the consolidated file at:

```text
./pii_datasets/consolidated/consolidated.jsonl
```

It writes the prepared splits and label mapping under `./data/`.

### 4. Download the base model

```bash
python srcripts/download_model.py
```

The model and tokenizer are saved under `./models/deberta-v3-base/`.

### 5. Fine-tune DeBERTa-v3

The standard training script exposes its training options through command-line arguments. A default run can be started with:

```bash
python srcripts/train.py
```

For a constrained GPU, reduce the per-device batch size and enable gradient checkpointing. For example:

```bash
python srcripts/train.py \
  --batch-size 2 \
  --eval-batch-size 4 \
  --grad-accum 32 \
  --gradient-checkpointing
```

To inspect all supported options:

```bash
python srcripts/train.py --help
```

The training script expects the prepared data and label mapping in the paths configured by the script. Check its argument help and configuration before running if you have changed the output directories.

### 6. Experimental training (optional)

The experimental trainer requires explicit split and output paths:

```bash
python srcripts/train_novel.py \
  --splits-dir ./data \
  --output-dir ./models/experimental \
  --source-cond \
  --curriculum \
  --hierarchical
```

This enables source conditioning, the three-stage curriculum and the hierarchical head. Omit any of those flags to disable the corresponding option. This run should be treated as a separate experiment from the standard trainer.

## Input and Output Format

### Training record

The prepared corpus is JSON Lines (`.jsonl`), with one record per line:

```json
{
  "tokens": ["My", "email", "is", "alice@example.com"],
  "labels": ["O", "O", "O", "B-EMAIL"],
  "source": "ai4privacy_400k"
}
```

Each `tokens` entry has a corresponding label in `labels`. The `source` field records which dataset supplied the example and is used by source-aware preparation and the experimental source-conditioning trainer.

### Inference module

The archive includes `srcripts/inference.py`, which defines `PIIDetector` and `FastPIIDetector` interfaces for inference with a saved model. The module is intended to return detected entity spans and redacted text. The supplied archive does not include a trained model directory, so inference cannot be demonstrated end-to-end from the ZIP alone.

## Implementation Notes and Limitations

- The project combines direct PII datasets with general NER and finance-domain NER datasets. These sources do not all represent PII in the same sense; general entities such as organisations and locations are included because they provide additional entity-recognition supervision.
- Label harmonisation necessarily merges some fine-grained source categories into broader labels. This improves compatibility across sources but removes some source-specific distinctions.
- Rare entity types below the configured mention threshold are mapped to `O`, which reduces the output label space but also means those entity categories are not learned as distinct classes.
- Span-to-token conversion depends on the quality and consistency of source offsets. The converter includes fallback logic, but approximate alignment can still introduce annotation noise.
- The 64.55% F1 score is reported from the project summary; the ZIP does not include the corresponding model checkpoint or evaluation artifact.
- The repository archive contains an experimental trainer and inference module, but no generated model/data artifacts.
- The supplied project does not contain a complete baseline benchmarking pipeline or benchmark result table. Do not describe comparisons against external baselines as implemented in this archive without adding the missing evaluation code and outputs.

## Acknowledgements

This project builds on publicly available datasets and the Hugging Face Transformers ecosystem. Please cite and comply with the usage terms of each dataset and the DeBERTa-v3 model.

## Author

**Somiran Dutta**

- GitHub: [SomiranDutta4](https://github.com/SomiranDutta4)
- Repository: [Automated-PII-Detection](https://github.com/SomiranDutta4/Automated-PII-Detection)

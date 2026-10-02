# Malayalam Agglutination Splitter

Open-source Malayalam agglutination splitter, morphological segmentation model and token-efficiency toolkit powered by ByT5.

Malayalam NLP, Malayalam tokenization, Malayalam language models, Indic AI and low-resource language technology all face the same underlying problem: a single written word can contain what another language would express as several words. This project predicts useful morpheme/component boundaries so that downstream systems can work with more meaningful pieces of Malayalam text.

## Why Malayalam agglutination matters for tokenisation

Malayalam is morphologically rich. A noun can combine a stem, plural marking, case marking and a clitic into one orthographic word. Verbs can combine a root with tense, aspect, person, mood and politeness information. Surface spelling also changes through sandhi and allomorphy. The result is a long word that carries several independent pieces of information but arrives at a language model as one uninterrupted Unicode string.

This creates several difficulties for a general-purpose subword tokenizer:

1. **Fragmentation:** a frequent stem and a rare suffix may be split into many small, opaque pieces.
2. **Vocabulary sparsity:** productive Malayalam forms are nearly unlimited, so many valid words are unseen during tokenizer training.
3. **Longer sequences:** one agglutinated word can consume several model tokens, reducing the amount of useful context that fits into a fixed context window.
4. **Unstable generalisation:** the model may learn a suffix as unrelated byte or subword fragments instead of recognising that it performs the same grammatical function across words.
5. **Cross-model inconsistency:** different tokenizers can split the same Malayalam word differently, complicating retrieval, OCR correction, translation and evaluation.

For example, an orthographic form such as `ചലനാത്മകമാക്കേണ്ടതെങ്ങനെയാണെന്ന്` contains a lexical stem, causative/derivational material, a necessity form, a manner word, a copula and a quotative/complementizer. A generic tokenizer may encode this as a long sequence of rare fragments. A morphology-aware representation can expose the reusable structure:

```text
ചലനാത്മകം + ആക്കേണ്ടത് + എങ്ങനെ + ആണ് + എന്ന്
```

The split is useful as an additional representation for training, retrieval or analysis. It does not replace the original surface word, and it does not mean that every analysis has one universally correct answer.

## How this model improves token efficiency

The ByT5 splitter is trained to map a Malayalam word or sentence to a readable component sequence. That output can be used in a preprocessing pipeline:

```text
Malayalam text
      |
      v
Agglutination splitter (ByT5)
      |
      v
Stem + suffix/clitic representation
      |
      v
Downstream tokenizer / LLM / search index / translation model
```

For a downstream subword model, consistent morpheme-aware boundaries can reduce redundant fragmentation and make recurring Malayalam stems and grammatical markers easier to model. The practical benefit must be measured for each base tokenizer: report tokens per sentence, tokens per character, context utilisation, truncation rate and task quality before and after segmentation. This repository therefore treats **better token ratio** as an evaluation hypothesis, not a guaranteed number.

The splitter itself uses ByT5 because byte-level modelling avoids depending on a pre-existing Malayalam vocabulary during the segmentation task. ByT5 is the boundary predictor; the downstream model still determines the final token count. The same split representation may help Malayalam LLMs, Indic language models, machine translation, OCR post-correction, search, spell checking and morphological analysis.

The core recipe uses [ByT5](https://github.com/google-research/byt5), a byte-level T5 model that can work directly with arbitrary Unicode text. The model is intended for Malayalam NLP research, morphological analysis, search normalisation, OCR post-correction, machine translation preprocessing and language-learning tools.

> Dataset status: the Malayalam agglutination dataset is **under preparation**. The repository currently contains no released training corpus and no real Hugging Face dataset ID. Code uses the visibly fake placeholder `YOUR_HF_USERNAME/malayalam-agglutination-dataset-under-preparation` until a licensed dataset is published.

**Keywords:** Malayalam tokenization, Malayalam tokenizer, Malayalam morphological segmentation, Malayalam agglutination, Malayalam NLP, Malayalam LLM, Indic language models, ByT5, Unicode NLP, morpheme segmentation, low-resource language AI, OCR post-correction and machine translation preprocessing.

## What the model does

The model maps an agglutinated Malayalam surface form to the project’s annotated component representation. These are the exact five examples supplied for this project.

| Surface form | Annotated split |
|---|---|
| `ഭരണനേട്ടങ്ങളൊന്നുമില്ലാതെയാണ്‌` | `ഭരണനേട്ടങ്ങൾ + ഒന്നും + ഇല്ലാതെ + ആണ്` |
| `വ്യാപൃതരാവേണ്ടിയിരിക്കുന്നുവെന്നും` | `വ്യാപൃതർ + ആവേണ്ടി + ഇരിക്കുന്നു + എന്നും` |
| `വ്യക്തമായിരിക്കുന്നതെന്ന്കമ്പനി` | `വ്യക്തം + ആയിരിക്കുന്നത് + എന്ന് + കമ്പനി` |
| `ജീവാത്മസ്വരൂപത്തെയറിഞ്ഞുകൊൾവാനുളള` | `ജീവാത്മ + സ്വരൂപത്തെ + അറിഞ്ഞുകൊൾവാൻ + ഉള്ള` |
| `വാഹനങ്ങളുണ്ടാക്കുന്നതെന്നായിരുന്നു` | `വാഹനങ്ങൾ + ഉണ്ടാക്കുന്നത് + എന്ന് + ആയിരുന്നു` |

The same examples in annotation-review notation:

```text
ഭരണനേട്ടങ്ങളൊന്നുമില്ലാതെയാണ്‌ → ഭരണനേട്ടങ്ങൾ + ഒന്നും + ഇല്ലാതെ + ആണ്
വ്യാപൃതരാവേണ്ടിയിരിക്കുന്നുവെന്നും → വ്യാപൃതർ + ആവേണ്ടി + ഇരിക്കുന്നു + എന്നും
വ്യക്തമായിരിക്കുന്നതെന്ന്കമ്പനി → വ്യക്തം + ആയിരിക്കുന്നത് + എന്ന് + കമ്പനി
ജീവാത്മസ്വരൂപത്തെയറിഞ്ഞുകൊൾവാനുളള → ജീവാത്മ + സ്വരൂപത്തെ + അറിഞ്ഞുകൊൾവാൻ + ഉള്ള
വാഹനങ്ങളുണ്ടാക്കുന്നതെന്നായിരുന്നു → വാഹനങ്ങൾ + ഉണ്ടാക്കുന്നത് + എന്ന് + ആയിരുന്നു
```

These are normalised linguistic annotations: a target component sequence may not concatenate character-for-character back to the surface word because of Malayalam sandhi and allomorphy. The future dataset card must define the lemma policy, allomorph policy and inter-annotator agreement. These examples are project data, not automatically generated demonstrations.

For machine-readable training data, store each pair in separate fields:

```jsonl
{"word":"ഭരണനേട്ടങ്ങളൊന്നുമില്ലാതെയാണ്‌","segmentation":"ഭരണനേട്ടങ്ങൾ + ഒന്നും + ഇല്ലാതെ + ആണ്"}
{"word":"വ്യാപൃതരാവേണ്ടിയിരിക്കുന്നുവെന്നും","segmentation":"വ്യാപൃതർ + ആവേണ്ടി + ഇരിക്കുന്നു + എന്നും"}
{"word":"വ്യക്തമായിരിക്കുന്നതെന്ന്കമ്പനി","segmentation":"വ്യക്തം + ആയിരിക്കുന്നത് + എന്ന് + കമ്പനി"}
{"word":"ജീവാത്മസ്വരൂപത്തെയറിഞ്ഞുകൊൾവാനുളള","segmentation":"ജീവാത്മ + സ്വരൂപത്തെ + അറിഞ്ഞുകൊൾവാൻ + ഉള്ള"}
{"word":"വാഹനങ്ങളുണ്ടാക്കുന്നതെന്നായിരുന്നു","segmentation":"വാഹനങ്ങൾ + ഉണ്ടാക്കുന്നത് + എന്ന് + ആയിരുന്നു"}
```

## Features

- ByT5 Unicode/byte-level modelling for Malayalam and code-switched text;
- configurable input and target columns;
- train/validation/test split support without embedding a real dataset in code;
- checkpoint saving and generation configuration;
- exact-match, component-level F1 and character-level diagnostics;
- optional LoRA/PEFT extension point for constrained hardware;
- reproducible CLI commands and model-card guidance.

## Quick start

```bash
git clone https://github.com/Neo017/Malayalam_Agglutination_Splitter.git
cd Malayalam_Agglutination_Splitter
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[train]'
```

The training scripts deliberately refuse to pretend that the placeholder dataset exists. Pass a real local dataset or a published dataset ID after the corpus is prepared.

### Train ByT5

Expected dataset fields are `word` and `segmentation` by default. A local `datasets.save_to_disk()` directory or a Hugging Face dataset ID may be supplied.

```bash
python scripts/train.py \
  --dataset PATH_OR_PUBLISHED_DATASET_ID \
  --model google/byt5-small \
  --input-column word \
  --target-column segmentation \
  --output-dir runs/byt5-ml-segmentation
```

### Segment words

```bash
python scripts/predict.py \
  --model runs/byt5-ml-segmentation \
  --word ചലനാത്മകമാക്കേണ്ടതെങ്ങനെയാണെന്ന്
```

### Evaluate

```bash
python scripts/evaluate.py \
  --model runs/byt5-ml-segmentation \
  --dataset PATH_OR_PUBLISHED_DATASET_ID \
  --split test
```

## Dataset contract

The planned dataset uses one example per line/row:

```json
{"word": "ചലനാത്മകമാക്കേണ്ടതെങ്ങനെയാണെന്ന്", "segmentation": "ചലനാത്മകം + ആക്കേണ്ടത് + എങ്ങനെ + ആണ് + എന്ന്"}
```

Recommended metadata kept alongside each example:

```text
id, surface_word, segmentation, lemma_policy, source, annotator_ids,
review_status, dialect, domain, license, confidence
```

Training data should include positive agglutinated words, simple words, productive and irregular forms, spelling variants, code-switching boundaries, and hard negative cases where a sequence should not be split. Keep annotator metadata out of the model input unless it is intentionally used.

### Annotation policy before release

1. Decide whether targets represent surface chunks or dictionary-normalised morphemes.
2. Publish how plural, case, tense, clitics, reduplication and sandhi are labelled.
3. Use at least two annotators for a quality subset and adjudicate disagreements.
4. Split by lemma/source family where possible to prevent near-duplicate leakage.
5. Record the licence and consent/provenance for every source.
6. Keep a held-out test set unavailable during model development.

## Evaluation metrics

The included evaluator reports:

- exact segmentation accuracy;
- morpheme-component precision, recall and F1;
- invalid-output rate when the model emits no usable segments;
- invalid-output rate and average generated component count.

Component F1 compares the `+`-separated target components. It is intentionally not a character-boundary score: normalised lemmas may not concatenate to the observed surface word after sandhi or allomorphy. It should be supplemented by Malayalam-speaker review and, for a surface-only task, a separate surface-boundary evaluator.

## Limitations

ByT5 does not automatically learn Malayalam morphology from Unicode alone. Quality depends on coverage, annotation consistency and leakage control. The examples in this README are illustrative. The model should not be used to assert a linguistically authoritative analysis without human review, especially for dialectal, poetic, borrowed or code-switched words.

## Roadmap

- publish a licensed and documented Malayalam agglutination dataset;
- add lemma-aware and surface-only task variants;
- measure annotator agreement by morphology type;
- compare ByT5-small, ByT5-base and lightweight adapters;
- add dialect, OCR-noise and code-switch evaluation;
- release a model card with data hashes, licences, metrics and known errors;
- expose an optional FastAPI inference service after benchmark validation.

## Repository layout

```text
Malayalam_Agglutination_Splitter/
├── README.md
├── LICENSE
├── pyproject.toml
├── configs/byt5-small.yaml
├── scripts/
│   ├── train.py
│   ├── predict.py
│   └── evaluate.py
├── src/malayalam_agglutination/
│   └── __init__.py
└── .gitignore
```

## Responsible release

Do not commit private corpora, credentials, unlicensed scraped text or unreviewed personal data. Before publishing a checkpoint, include the base-model licence, corpus provenance, annotation guide, train/test contamination check, hardware, software versions, command line, metrics and failure examples.

## Licence

Repository code is Apache-2.0. Any future model weights and dataset are separately licensed; their terms must be stated in the model card and dataset card.

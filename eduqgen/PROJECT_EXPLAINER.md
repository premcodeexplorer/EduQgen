# EduQGen — Complete Project Explainer

A presentation guide for **Prem & Purva**. Read top to bottom; each section first explains the *concept*, then maps it to *what we actually do in EduQGen*.

---

## 1. The Big Picture (30-second pitch)

**EduQGen takes a chunk of study notes and generates a quiz from it.** A student uploads a PDF / image / text, and the system produces multiple-choice (or descriptive / true-false / numerical) questions, each labelled with a difficulty level. They take the quiz, get scored, and the result is saved to a history dashboard.

Under the hood, four ML models work in a pipeline:

| # | Model | Job |
|---|---|---|
| 1 | **Autoencoder** (TF-IDF + Dense) | Find the *most important* sentences in the notes |
| 2 | **Bidirectional LSTM** | Score those sentences for "question-worthiness" |
| 3 | **T5-base (transformer)** | Generate the actual question text from a sentence + answer |
| 4 | **ANN classifier** | Tag each finished question as Easy / Medium / Hard |

Each model represents a different family of deep learning — that's the academic point of the project: showing four very different architectures collaborating on one real task.

---

## 2. The Four Models — Concept → Our Use

### Model 1: Autoencoder (Key Sentence Extraction)

**Concept.** An autoencoder is a neural network that learns to *compress and then reconstruct* its own input. The middle "bottleneck" layer is forced to throw away information. After training, sentences the model can rebuild *easily* are common/predictable; sentences it rebuilds *poorly* (high reconstruction error) are *unique* — they contain the most distinctive information.

**Our use.** When a student uploads notes, we:

1. Split the notes into sentences with NLTK.
2. Convert each sentence to a TF-IDF vector (so words become numbers).
3. Train a tiny autoencoder *on the spot* (50 epochs, ~2 sec).
4. Run every sentence through the model and measure reconstruction error.
5. Pick the **top-N highest-error sentences** — these are the "key facts" we'll build questions from.

> Why this works: in a paragraph of generic English, sentences with rare technical terms (e.g. *"a stack follows the LIFO principle"*) are the hardest to reconstruct — exactly the ones a quiz should test.

**File:** `ml_models/autoencoder.py` → class `SentenceAutoencoder`

---

### Model 2: Bidirectional LSTM (Sentence Importance Scoring)

**Concept.** An LSTM is a recurrent neural network that reads text *one word at a time*, remembering context as it goes. **Bidirectional** means it reads the sentence both forwards and backwards, so each word "sees" both its past and future context. This lets the model judge whether a sentence is the *kind of sentence* a question would naturally be asked about.

**Our use.** We pre-trained a BiLSTM on **SQuAD v1.1** (Stanford Question Answering Dataset), which has 100K+ Wikipedia sentences each labelled as either *contains an answer* (1) or *doesn't* (0). After training, the model outputs a probability between 0 and 1 — *"how question-worthy is this sentence?"*

In the live pipeline, the autoencoder gives us ~20 candidate sentences; the LSTM re-ranks them and we keep the top *num_questions* highest-scoring ones.

**Architecture (after our overfitting fix):**
- Embedding(20000 → 128)
- BiLSTM(32, recurrent_dropout=0.2) → Dropout(0.3)
- BiLSTM(32, recurrent_dropout=0.2) → Dropout(0.3)
- GlobalMaxPooling
- Dense(32, ReLU, L2 regularization) → Dropout(0.3)
- Dense(1, Sigmoid)

> **Talk-track for the demo:** the original LSTM hit 94% train / 72% val accuracy — classic overfitting. We added dropout, recurrent dropout, L2 regularization, halved the LSTM units, and added EarlyStopping with `restore_best_weights`. Final result: ~80% train / ~80% val — a much healthier model.

**File:** `ml_models/lstm_model.py` → class `LSTMContextModel`

---

### Model 3: T5-base (Question Generation Transformer)

**Concept.** T5 ("Text-to-Text Transfer Transformer") is a Google encoder-decoder model that treats *every* NLP task as text-in / text-out. We use a community fine-tuned variant — `valhalla/t5-base-qg-hl` — which was specifically trained to do **question generation**: you feed it a sentence with the desired answer wrapped in `<hl> ... <hl>` tags, and it produces a natural-language question whose answer is that highlighted phrase.

**Our use.** For each key sentence:

1. Extract the best *answer phrase* using NLTK noun-phrase chunking (we prefer informative multi-word phrases that appear in the latter half of the sentence — those are usually the objects/complements).
2. Wrap that phrase with `<hl>` tags.
3. Feed it to T5 with the prompt `"generate question: <highlighted sentence>"`.
4. T5 generates a question via **beam search** (`num_beams=5`) for higher quality.
5. We post-process: fix capitalization, ensure trailing `?`, strip "...what?" fragments.
6. **Quality gate** — if the T5 output is malformed (too short, leaks the answer, doesn't start with a wh-word), we fall back to a clean **fill-in-the-blank (cloze)** question instead. Cloze always reads naturally.

> We started on `t5-small` (~60M params, 240MB) and the questions were rough. Switching to `t5-base` (~220M params, 850MB) dramatically improved phrasing. The fallback cloze still catches the rare bad outputs.

**File:** `ml_models/question_generator.py` → class `QuestionGeneratorPipeline`, methods `_generate_question`, `_make_cloze`, `_extract_answer`

---

### Model 4: ANN Difficulty Classifier (Easy / Medium / Hard)

**Concept.** A plain feed-forward Artificial Neural Network — input layer, a few hidden Dense layers, softmax output. Given hand-engineered features about a question, it predicts which of three difficulty buckets it belongs in.

**Our use.** For every generated question, we extract **12 features** from the question + answer + source sentence:

- Question length, answer length, source length
- answer/source word ratio, question/source word ratio
- Number of commas in question
- Is it a "why/how/explain" reasoning question? (binary)
- Is it a "what/which/who" factual question? (binary)
- Is it a "when/where" recall question? (binary)
- Word overlap between question and source
- Does the answer contain digits? (binary)
- Answer character length

These 12 features go into a small ANN:

- Dense(128, ReLU, L2) → BatchNorm → Dropout(0.3)
- Dense(64, ReLU, L2) → BatchNorm → Dropout(0.3)
- Dense(32, ReLU, L2)
- Dense(3, Softmax) → [Easy, Medium, Hard]

We trained it on SQuAD using a **heuristic labelling rule** (since SQuAD has no difficulty labels):
- Short answer (≤2 words) + high overlap with source → **Easy**
- Medium-length answer (≤5 words) → **Medium**
- Long answer with low overlap → **Hard**

**File:** `ml_models/ann_classifier.py` → class `DifficultyClassifier`

---

## 3. The Full Pipeline — End-to-End Journey

When the user clicks **"Generate Questions"**, this is what happens:

```
Notes (PDF / image / text)
       │
       ▼
  ┌─────────────────────────┐
  │  STEP 0 — INGESTION     │  utils.py
  │  PDF → PyMuPDF text     │
  │  Image → Tesseract OCR  │
  │  Text → cleaned string  │
  └─────────────────────────┘
       │
       ▼
  ┌─────────────────────────┐
  │  STEP 1 — AUTOENCODER   │  Pick ~20 most "important" sentences
  │  (TF-IDF + reconstruct) │  via reconstruction error
  └─────────────────────────┘
       │
       ▼
  ┌─────────────────────────┐
  │  STEP 2 — LSTM SCORING  │  Re-rank by "question-worthiness"
  │  (BiLSTM, SQuAD-trained)│  Keep the top N
  └─────────────────────────┘
       │
       ▼
  ┌─────────────────────────┐
  │  STEP 3 — ANSWER PICK   │  NLTK noun-phrase chunking
  │  (NLTK POS + RegexParse)│  Pick the best NP per sentence
  └─────────────────────────┘
       │
       ▼
  ┌─────────────────────────┐
  │  STEP 4 — T5 GENERATE   │  Highlight the answer, ask T5
  │  (t5-base-qg-hl)        │  for a question. Cloze fallback
  │                         │  on bad output.
  └─────────────────────────┘
       │
       ▼
  ┌─────────────────────────┐
  │  STEP 5 — DISTRACTORS   │  Pull 3 wrong options from a
  │  (NP pool, similarity)  │  pre-built noun-phrase pool of
  │                         │  the FULL text. Filter overlaps.
  └─────────────────────────┘
       │
       ▼
  ┌─────────────────────────┐
  │  STEP 6 — ANN DIFFICULTY│  Tag Easy/Medium/Hard from
  │  (12 features → softmax)│  question features
  └─────────────────────────┘
       │
       ▼
  Final question objects → rendered as quiz UI
```

That's the journey. **Four models, one question.**

---

## 4. File-by-File Walkthrough

### 4.1 `ml_models/` — The Brains

| File | What it does |
|---|---|
| `autoencoder.py` | `SentenceAutoencoder` class. Builds a Dense(256→64→256→input) autoencoder over TF-IDF vectors. `extract_key_sentences()` fits the model on the user's text at runtime and returns the highest-error sentences. |
| `lstm_model.py` | `LSTMContextModel` class. BiLSTM with dropout + L2 regularization. `train_on_squad()` is used once during training; `get_context_scores()` is the runtime API that returns a 0–1 score per sentence. Has `EarlyStopping` + `ModelCheckpoint` callbacks. |
| `ann_classifier.py` | `DifficultyClassifier` class. `extract_features()` (static) builds the 12-feature vector. `assign_heuristic_difficulty()` labels SQuAD samples. `predict_difficulty()` is the runtime API. |
| `question_generator.py` | The orchestrator. `QuestionGeneratorPipeline` owns instances of all four models. Key methods: `load_trained_models()` (strict loader, fails fast), `_extract_answer()` (NLTK noun-phrase pick), `_generate_question()` (T5 + cleanup + cloze fallback), `_generate_distractors()` (NP-pool sampling with overlap filter), `generate_questions()` (the public end-to-end method). |

### 4.2 `backend/` — The Web Layer

| File | What it does |
|---|---|
| `app.py` | Flask entry point. Creates the app, loads the ML pipeline at startup (fails fast with a clear error if any model file is missing), registers routes. Run with `python backend/app.py`. |
| `routes.py` | All HTTP endpoints. Pages: `/` (upload), `/dashboard`, `/quiz`, `/results`, `/analysis`, `/models`, `/quiz_detail`. APIs: `/upload` (the main one — runs the full pipeline), `/submit_quiz`, `/retake_quiz`, `/delete_upload`, `/download_pdf`. Sessions are stored as JSON files keyed by a UUID cookie. |
| `utils.py` | File handling: PDF extraction (PyMuPDF), image OCR (Tesseract), text cleaning, history persistence, dashboard stats, PDF report generation, upload logging. |

### 4.3 `train_models.py` — One-time Training Runner

Run once with `python train_models.py`. It:

1. Downloads SQuAD v1.1 from HuggingFace (5000 samples).
2. Splits each context into sentences and labels them (1 if it contains the answer, else 0) — this is the LSTM's training data.
3. Trains the LSTM (12 epochs max, EarlyStopping triggers earlier).
4. Trains the ANN difficulty classifier (20 epochs).
5. Saves both models to `data/saved_models/`.
6. Saves a `training_metrics.json` with loss/accuracy curves — used by the `/models` page to draw Chart.js graphs.

### 4.4 `config.py`, `templates/`, `static/`

- `config.py` — `UPLOAD_FOLDER`, `SAVED_MODELS_DIR`, allowed extensions, default question counts.
- `templates/` — Jinja2 HTML pages (`index.html`, `quiz.html`, `dashboard.html`, `analysis.html`, `models.html`, `results.html`, `quiz_detail.html`). All share `static/css/theme.css` for the unified warm-palette design.
- `static/css/theme.css` — Single design system: navy primary `#2d3a4a`, brown secondary `#8b7355`, beige background `#f5f3ee`. Components: `.card`, `.btn`, `.list-row`, `.slim-row`, `.stat`.

---

## 5. Why This Architecture? (The Talking Points)

When the panel asks *"why so many models?"* — here's the answer:

1. **Each model solves a problem the others can't.**
   - Autoencoder = unsupervised, runs on *the user's own text* with no training data needed.
   - LSTM = supervised, learns from SQuAD what "answerable" sentences look like in general.
   - T5 = pretrained transformer, brings the heavy linguistic knowledge needed to phrase a question grammatically.
   - ANN = lightweight classifier with hand-crafted features, perfect for a small classification job.

2. **It's a real demonstration of the deep-learning toolbox.** Unsupervised learning, recurrent networks, transformers, and a feed-forward classifier — four different families on one task.

3. **It mirrors how a teacher actually writes a quiz.** Skim the notes for important bits → decide what's testable → write the question → set the difficulty. We literally model that workflow.

---

## 6. Demo Script (Suggested Flow)

1. **Open the home page** — show the warm UI, two-column hero.
2. **Paste a historical paragraph** (recall-heavy text plays to our pipeline's strengths). Pick "MCQ" and 10 questions.
3. While it generates, **switch to the Models page** — show the BiLSTM and ANN training curves. Talk about the overfitting fix (94/72 → 80/80).
4. **Take the quiz**, intentionally get a few wrong, submit.
5. **Show the results page** — score ring, per-question review with green/red accent strips.
6. **Open the Analysis page** — show the *key sentences* the autoencoder picked, with their reconstruction-error and LSTM scores side-by-side. **This is the money slide** — it visually proves the pipeline is working.
7. **Open the Dashboard** — past quizzes, accuracy over time, difficulty breakdown chart.
8. **Briefly switch to Descriptive mode** to show that the pipeline supports multiple question types.

---

## 7. Quick FAQ for the Q&A

- **"Did you train T5 yourselves?"** No — T5 is pretrained (`valhalla/t5-base-qg-hl`). Training a transformer from scratch needs hundreds of GPUs. We trained the **LSTM and ANN ourselves** on SQuAD.
- **"Why TF-IDF for the autoencoder, not embeddings?"** TF-IDF is fast, deterministic, and trains in seconds on the user's own text — no need for a pretrained vocabulary. Perfect for an *online* unsupervised step.
- **"How do you avoid bad questions?"** Three layers: (1) we extract real noun phrases as answers via NLTK chunking, (2) we score T5 output and fall back to fill-in-the-blank if it's malformed, (3) distractors are filtered against the answer for word overlap.
- **"What's the dataset?"** SQuAD v1.1 from Stanford — 100K+ Q&A pairs. We use 5000 samples for fast training.
- **"Tech stack?"** Python 3.11, Flask, TensorFlow/Keras, HuggingFace Transformers, NLTK, scikit-learn, PyMuPDF, Tesseract OCR.

---

## 8. Division of Labour — Suggested Speaking Split

| Section | Speaker |
|---|---|
| Big picture & motivation | **Purva** |
| Autoencoder + LSTM (the *selection* half) | **Prem** |
| T5 + ANN (the *generation* half) | **Purva** |
| Live demo walkthrough | **both alternating** |
| Q&A | **whoever knows the answer faster** |

Good luck. You've got this.

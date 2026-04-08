# EduQGen — All Concepts You Need to Learn

> Learn these on Claude Web before we start coding.
> Grouped by priority — learn in this exact order.
> Copy each section heading and ask Claude to explain it simply with code examples.

---

## PHASE 1: Python & NLP Basics (Learn First)

### 1.1 Text Preprocessing
- [ ] Tokenization (splitting text into words and sentences)
- [ ] Stopword removal (removing "the", "is", "a", etc.)
- [ ] Lowercasing and punctuation removal
- [ ] Stemming vs Lemmatization (difference and when to use)
- [ ] NLTK library — `nltk.tokenize`, `nltk.corpus.stopwords`

### 1.2 Word Embeddings
- [ ] What are word embeddings? (words as numbers)
- [ ] One-hot encoding vs dense vectors (why dense is better)
- [ ] Word2Vec — how it works (CBOW vs Skip-gram)
- [ ] Pre-trained Word2Vec (Google News 300-dim)
- [ ] Gensim library — loading and using Word2Vec
- [ ] How to convert a sentence to a vector (average of word vectors)

### 1.3 Text Feature Extraction
- [ ] TF-IDF — what it is and how to calculate
- [ ] Named Entity Recognition (NER) — what are entities (person, place, org)
- [ ] Part-of-Speech (POS) tagging — nouns, verbs, adjectives
- [ ] spaCy library — NER and POS in 3 lines of code

### 1.4 Similarity Measures
- [ ] Cosine similarity — formula and intuition
- [ ] Why cosine similarity works better than Euclidean for text
- [ ] Using cosine similarity to find similar words/sentences

---

## PHASE 2: Deep Learning Fundamentals (Core Knowledge)

### 2.1 Neural Network Basics
- [ ] What is a neuron (weighted sum + activation)
- [ ] Activation functions — ReLU, Sigmoid, Softmax (when to use which)
- [ ] Forward pass — input to output
- [ ] Loss functions — MSE (regression) vs Cross-Entropy (classification)
- [ ] Backpropagation — how gradients flow backward
- [ ] Optimizers — SGD vs Adam (just use Adam, but know why)
- [ ] Learning rate — what happens if too high or too low
- [ ] Epochs, batch size, iterations — what each means

### 2.2 Regularization Techniques
- [ ] Overfitting vs Underfitting (the core problem)
- [ ] L1 Regularization (Lasso) — pushes weights to zero
- [ ] L2 Regularization (Ridge) — shrinks weights, never zero
- [ ] L1 + L2 together (Elastic Net)
- [ ] Dropout — randomly turning off neurons during training
- [ ] Early Stopping — stop training when validation loss increases
- [ ] Train/Validation/Test split — why 3 sets, not 2

### 2.3 Keras / TensorFlow Basics
- [ ] `Sequential` model vs `Functional` API (you need Functional for Seq2Seq)
- [ ] `Dense` layer — fully connected
- [ ] `model.compile()` — setting loss, optimizer, metrics
- [ ] `model.fit()` — training loop
- [ ] `model.predict()` — inference
- [ ] `model.save()` and `load_model()` — saving/loading .h5 files
- [ ] Callbacks — `EarlyStopping`, `ModelCheckpoint`

---

## PHASE 3: Autoencoder (Model 1)

### 3.1 Autoencoder Theory
- [ ] What is an Autoencoder? (compress then reconstruct)
- [ ] Encoder — compresses input to bottleneck
- [ ] Decoder — reconstructs input from bottleneck
- [ ] Bottleneck / Latent Space — the compressed representation
- [ ] Why the bottleneck forces the model to learn meaningful features
- [ ] Reconstruction loss — MSE between input and output
- [ ] Undercomplete vs Overcomplete autoencoders

### 3.2 Autoencoder in This Project
- [ ] Input: 300-dim Word2Vec sentence vector
- [ ] Bottleneck: 32-dim (this becomes your sentence embedding)
- [ ] Output: 300-dim reconstructed vector
- [ ] After training: throw away decoder, keep only encoder
- [ ] Use encoder output (32-dim) for clustering

### 3.3 K-Means Clustering
- [ ] What is K-Means? (group similar points into K groups)
- [ ] How it works — centroids, assignment, update, repeat
- [ ] Choosing K — elbow method
- [ ] Why we cluster: to ensure questions from ALL topics, not just first few
- [ ] Pick 1 representative sentence per cluster (closest to centroid)
- [ ] `sklearn.cluster.KMeans` — usage

---

## PHASE 4: LSTM Seq2Seq (Model 2)

### 4.1 RNN Basics
- [ ] What is a Recurrent Neural Network? (memory across time steps)
- [ ] Hidden state — carries information from previous steps
- [ ] Vanishing gradient problem — why basic RNNs forget long sequences
- [ ] Why we need LSTM instead of vanilla RNN

### 4.2 LSTM (Long Short-Term Memory)
- [ ] LSTM cell architecture — the 3 gates:
  - Forget gate — what to throw away from memory
  - Input gate — what new info to store
  - Output gate — what to output from memory
- [ ] Cell state vs Hidden state (cell = long-term, hidden = short-term)
- [ ] Stacked LSTM — multiple LSTM layers on top of each other (you use 4)
- [ ] `return_sequences=True` vs `False` (True for stacking, False for last layer)
- [ ] `return_state=True` — needed for Seq2Seq (returns h and c states)

### 4.3 Sequence-to-Sequence (Seq2Seq) Model
- [ ] What is Seq2Seq? (input sequence -> different output sequence)
- [ ] Encoder — reads input, produces context vector (h + c states)
- [ ] Decoder — takes context vector, generates output word by word
- [ ] Context vector — the "summary" passed from encoder to decoder
- [ ] The `<start>` and `<end>` special tokens (decoder boundaries)
- [ ] Teacher forcing — during training, feed correct previous word to decoder
- [ ] Why teacher forcing speeds up training

### 4.4 Beam Search (Inference)
- [ ] Greedy decoding — always pick highest probability word (simple but bad)
- [ ] Beam search — keep top-K candidates at each step
- [ ] Beam width (K=3 is good for this project)
- [ ] How beam search produces better questions than greedy

### 4.5 Seq2Seq in Keras (Functional API)
- [ ] Encoder model: `Input` -> `Embedding` -> `LSTM` -> states
- [ ] Decoder model: `Input` -> `Embedding` -> `LSTM(initial_state=encoder_states)` -> `Dense`
- [ ] Training model vs Inference model (different architectures, same weights)
- [ ] Padding sequences with `pad_sequences()`
- [ ] Embedding layer vs pre-trained Word2Vec embeddings

---

## PHASE 5: ANN Classifier (Model 3)

### 5.1 Classification with ANN
- [ ] Multi-class classification (3 classes: Easy, Medium, Hard)
- [ ] Softmax activation — output probabilities that sum to 1
- [ ] Categorical cross-entropy loss
- [ ] `sparse_categorical_crossentropy` vs `categorical_crossentropy` (when to use which)
- [ ] `argmax` — converting probabilities to predicted class

### 5.2 Feature Engineering for Questions
- [ ] Why we extract features manually (ANN needs numeric input)
- [ ] The 8 features:
  1. Sentence length (word count)
  2. Named entity count (spaCy NER)
  3. Verb count (spaCy POS)
  4. Average word length
  5. TF-IDF score of key terms
  6. Question word type (what/who/why/how encoded as number)
  7. Has negation (0 or 1)
  8. Noun chunk count (spaCy)
- [ ] Feature scaling — StandardScaler (why it matters for ANNs)

### 5.3 Synthetic Label Generation (Weak Supervision)
- [ ] What is weak supervision? (auto-generating labels using rules)
- [ ] Why we use it (no human-labeled difficulty dataset exists)
- [ ] Rule-based labeling:
  - Easy: short + "what/who" + few entities
  - Medium: mid-length + "which/where"
  - Hard: long + "why/how" + entities + negation
- [ ] Why this is acceptable for a lab project

---

## PHASE 6: Evaluation Metrics

### 6.1 For Question Generation (LSTM)
- [ ] BLEU Score — what it measures (n-gram overlap with reference)
- [ ] BLEU-1 vs BLEU-4 (unigram vs 4-gram)
- [ ] How to calculate with `nltk.translate.bleu_score`
- [ ] What is a good BLEU score? (> 0.35 for this task)

### 6.2 For Classification (ANN)
- [ ] Accuracy — simple but misleading for imbalanced data
- [ ] Precision — of all predicted "Hard", how many are actually Hard?
- [ ] Recall — of all actual "Hard" questions, how many did we catch?
- [ ] F1 Score — harmonic mean of precision and recall
- [ ] Confusion Matrix — 3x3 grid showing predicted vs actual
- [ ] `sklearn.metrics` — `f1_score`, `confusion_matrix`, `classification_report`

### 6.3 For Autoencoder
- [ ] Reconstruction loss (MSE) — lower is better
- [ ] Cosine similarity within clusters — higher means better clustering
- [ ] Silhouette score — measures cluster quality (-1 to 1)

---

## PHASE 7: Web App & Deployment

### 7.1 Flask Basics
- [ ] What is Flask? (Python web framework)
- [ ] Routes — `@app.route('/path')`
- [ ] GET vs POST methods
- [ ] `request.files` — handling file uploads
- [ ] `jsonify()` — returning JSON responses
- [ ] `render_template()` — serving HTML pages
- [ ] Running Flask app — `app.run(debug=True)`

### 7.2 Frontend Basics (Just Enough)
- [ ] HTML forms — file upload input
- [ ] Fetch API / AJAX — sending data to Flask without page reload
- [ ] Displaying JSON results in HTML
- [ ] Chart.js — basic bar/pie chart for metrics

### 7.3 PDF Generation
- [ ] ReportLab library — creating PDFs from Python
- [ ] Adding text, tables, headers to a PDF

### 7.4 Docker
- [ ] What is Docker? (package app + dependencies in a container)
- [ ] Dockerfile — `FROM`, `COPY`, `RUN`, `EXPOSE`, `CMD`
- [ ] `docker build` and `docker run`
- [ ] Why Docker matters (runs same everywhere)

### 7.5 Deployment
- [ ] What is Render? (free cloud hosting)
- [ ] Connect GitHub repo -> auto-deploy
- [ ] Environment variables (if needed)

---

## PHASE 8: Bonus Concepts (Learn Only If Time)

### 8.1 Bloom's Taxonomy
- [ ] 4 levels: Remember, Understand, Apply, Analyse
- [ ] Keyword mapping for each level
- [ ] Why educators care about this

### 8.2 Spaced Repetition (SM-2)
- [ ] What is spaced repetition? (review at increasing intervals)
- [ ] SM-2 algorithm — how intervals are calculated
- [ ] Implementation: store next_review_date per question

### 8.3 MLflow
- [ ] What is MLflow? (experiment tracking)
- [ ] `mlflow.log_param()` — log hyperparameters
- [ ] `mlflow.log_metric()` — log loss, accuracy
- [ ] `mlflow.ui` — visual dashboard

### 8.4 Attention Mechanism (Advanced)
- [ ] What is attention? (decoder looks back at encoder outputs)
- [ ] Why it helps (handles long sentences better)
- [ ] Bahdanau attention vs Luong attention
- [ ] Only add this if LSTM quality is poor without it

---

## How to Use This List with Claude Web

Copy-paste prompts like these:

```
"Explain Word2Vec simply with a Python code example using Gensim.
I'm a B.Tech student building a question generator project."

"Explain LSTM Seq2Seq model for question generation.
Show me the Keras code for encoder and decoder.
Keep it simple, I'm learning."

"What is an Autoencoder? Explain encoder, decoder, bottleneck.
Then show me how to use the encoder output for K-Means clustering."

"Explain L1, L2 regularization and Dropout with Keras code.
I need to build a 3-class ANN classifier."

"Explain BLEU score and F1 score simply.
How do I calculate them in Python?"
```

---

## Estimated Learning Time

| Phase | Topics | Time Needed |
|-------|--------|-------------|
| Phase 1 | Python & NLP Basics | 2-3 hours |
| Phase 2 | DL Fundamentals | 2-3 hours |
| Phase 3 | Autoencoder | 1-2 hours |
| Phase 4 | LSTM Seq2Seq | 3-4 hours (hardest part) |
| Phase 5 | ANN Classifier | 1-2 hours |
| Phase 6 | Evaluation Metrics | 1 hour |
| Phase 7 | Flask & Deployment | 1-2 hours |
| **Total** | | **~12-16 hours** |

> **Tip:** You don't need to master everything before coding.
> Learn Phase 1-3 -> Build Day 1.
> Learn Phase 4 -> Build Day 2.
> Learn Phase 5-6 -> Build Day 3.
> Learn Phase 7 -> Build Day 4.

---

**Go learn. Come back when ready. I'll write every line of code.**

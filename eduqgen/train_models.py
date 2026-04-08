"""
Training Script — Run this ONCE before starting the app.
Downloads SQuAD v1.1 from HuggingFace and trains:
  1. LSTM Context Model (learns which sentences are question-worthy)
  2. ANN Difficulty Classifier (learns to label Easy/Medium/Hard)
Saves training metrics for visualization on the Models page.

Usage:
  python train_models.py
"""

import os
import sys
import json
import numpy as np
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(__file__))

from ml_models.lstm_model import LSTMContextModel
from ml_models.ann_classifier import DifficultyClassifier

METRICS_FILE = os.path.join('data', 'saved_models', 'training_metrics.json')


def prepare_squad_data(max_samples=5000):
    """Download SQuAD v1.1 and prepare training data."""
    print("=" * 50)
    print("Step 1: Downloading SQuAD v1.1 from HuggingFace...")
    print("=" * 50)
    dataset = load_dataset("rajpurkar/squad", split="train")
    print(f"Downloaded {len(dataset)} samples")

    dataset = dataset.shuffle(seed=42).select(range(min(max_samples, len(dataset))))
    print(f"Using {len(dataset)} samples for training")

    contexts = []
    labels = []
    questions = []
    answers = []
    source_sentences = []

    for item in dataset:
        context = item['context']
        answer_text = item['answers']['text'][0]
        question = item['question']

        sentences = [s.strip() for s in context.split('.') if len(s.strip()) > 10]

        for sent in sentences:
            contexts.append(sent)
            if answer_text.lower() in sent.lower():
                labels.append(1)
            else:
                labels.append(0)

        answer_sent = context
        for sent in sentences:
            if answer_text.lower() in sent.lower():
                answer_sent = sent
                break

        questions.append(question)
        answers.append(answer_text)
        source_sentences.append(answer_sent)

    print(f"Prepared {len(contexts)} sentences for LSTM")
    print(f"Prepared {len(questions)} Q&A pairs for ANN")
    print(f"Positive labels (answer sentences): {sum(labels)}")
    print(f"Negative labels (non-answer sentences): {len(labels) - sum(labels)}")

    return contexts, labels, questions, answers, source_sentences


def train_lstm(contexts, labels):
    """Train the LSTM context model."""
    print("\n" + "=" * 50)
    print("Step 2: Training LSTM Context Model...")
    print("=" * 50)

    lstm = LSTMContextModel(max_words=20000, max_len=100)
    # epochs raised so EarlyStopping(patience=2) can find the best val_loss point
    history = lstm.train_on_squad(contexts, labels, epochs=12, batch_size=64)
    lstm.save('data/saved_models')
    print("LSTM model saved to data/saved_models/")

    # Get model summary info
    lstm_info = {
        'name': 'Bidirectional LSTM',
        'total_params': int(lstm.model.count_params()),
        'epochs': 5,
        'batch_size': 64,
        'training_samples': len(contexts),
        'history': {k: [float(v) for v in vals] for k, vals in history.items()},
        'layers': []
    }
    for layer in lstm.model.layers:
        lstm_info['layers'].append({
            'name': layer.name,
            'type': layer.__class__.__name__,
            'output_shape': str(layer.output_shape if hasattr(layer, 'output_shape') else 'N/A'),
            'params': int(layer.count_params())
        })

    return lstm, lstm_info


def train_ann(questions, answers, source_sentences):
    """Train the ANN difficulty classifier."""
    print("\n" + "=" * 50)
    print("Step 3: Training ANN Difficulty Classifier...")
    print("=" * 50)

    classifier = DifficultyClassifier()
    history = classifier.train(questions, answers, source_sentences, epochs=20, batch_size=32)
    classifier.save('data/saved_models')
    print("ANN classifier saved to data/saved_models/")

    ann_info = {
        'name': 'ANN Difficulty Classifier',
        'total_params': int(classifier.model.count_params()),
        'epochs': 20,
        'batch_size': 32,
        'training_samples': len(questions),
        'history': {k: [float(v) for v in vals] for k, vals in history.items()},
        'layers': []
    }
    for layer in classifier.model.layers:
        ann_info['layers'].append({
            'name': layer.name,
            'type': layer.__class__.__name__,
            'output_shape': str(layer.output_shape if hasattr(layer, 'output_shape') else 'N/A'),
            'params': int(layer.count_params())
        })

    return classifier, ann_info


def test_models(lstm, classifier):
    """Quick test to verify models work."""
    print("\n" + "=" * 50)
    print("Step 4: Testing models...")
    print("=" * 50)

    test_sentences = [
        "Machine learning is a subset of artificial intelligence.",
        "The weather today is sunny.",
        "Neural networks consist of layers of interconnected nodes."
    ]
    scores = lstm.get_context_scores(test_sentences)
    print("\nLSTM Context Scores:")
    for sent, score in zip(test_sentences, scores):
        print(f"  [{score:.3f}] {sent}")

    difficulty, conf = classifier.predict_difficulty(
        "What is machine learning?",
        "artificial intelligence",
        "Machine learning is a subset of artificial intelligence."
    )
    print(f"\nANN Test: Difficulty = {difficulty} (confidence: {conf:.2f})")


if __name__ == '__main__':
    print("EduQGen — Model Training Script")
    print("This will download SQuAD v1.1 and train LSTM + ANN models.\n")

    os.makedirs('data/saved_models', exist_ok=True)

    # Prepare data
    contexts, labels, questions, answers, source_sentences = prepare_squad_data(max_samples=5000)

    # Train models
    lstm, lstm_info = train_lstm(contexts, labels)
    classifier, ann_info = train_ann(questions, answers, source_sentences)

    # Save all training metrics
    metrics = {
        'lstm': lstm_info,
        'ann': ann_info,
        'autoencoder': {
            'name': 'Sentence Autoencoder',
            'type': 'Unsupervised',
            'description': 'Trains on user notes at runtime. Uses TF-IDF + shallow autoencoder to find key sentences via reconstruction error.',
            'architecture': ['Input (TF-IDF vector)', 'Dense(256, relu)', 'Dense(64, relu) — Bottleneck', 'Dense(256, relu)', 'Dense(input_dim, sigmoid)'],
            'training': 'Unsupervised — trains on each uploaded note (50 epochs, ~2 seconds)'
        },
        't5': {
            'name': 'T5-Small Question Generator',
            'type': 'Pretrained (HuggingFace)',
            'model_id': 'valhalla/t5-small-qg-hl',
            'total_params': 60506624,
            'description': 'Fine-tuned T5-small for question generation from highlighted text. Pretrained on SQuAD.',
            'architecture': ['T5 Encoder (6 layers, 8 attention heads)', 'T5 Decoder (6 layers, 8 attention heads)', 'Vocabulary: 32128 tokens'],
            'training': 'Pretrained — auto-downloads from HuggingFace (~242MB)'
        },
        'dataset': {
            'name': 'SQuAD v1.1',
            'source': 'rajpurkar/squad (HuggingFace)',
            'total_samples': 87599,
            'used_samples': min(5000, 87599),
            'description': 'Stanford Question Answering Dataset — 100K+ question-answer pairs from Wikipedia articles'
        }
    }

    with open(METRICS_FILE, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nTraining metrics saved to {METRICS_FILE}")

    # Test
    test_models(lstm, classifier)

    print("\n" + "=" * 50)
    print("TRAINING COMPLETE!")
    print("Saved models in: data/saved_models/")
    print("You can now run the app: python backend/app.py")
    print("=" * 50)

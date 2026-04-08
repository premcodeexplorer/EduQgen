"""
ANN Difficulty Classifier
Classifies generated questions into Easy / Medium / Hard
based on features extracted from the question and its source sentence.
Trained on SQuAD v1.1 with heuristic difficulty labels.
"""

import os
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
import pickle


DIFFICULTY_LABELS = {0: 'Easy', 1: 'Medium', 2: 'Hard'}


class DifficultyClassifier:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()

    def _build_model(self, input_dim):
        reg = l2(0.001)
        model = Sequential([
            Dense(128, activation='relu', kernel_regularizer=reg, input_shape=(input_dim,)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu', kernel_regularizer=reg),
            BatchNormalization(),
            Dropout(0.3),
            Dense(32, activation='relu', kernel_regularizer=reg),
            Dense(3, activation='softmax')  # Easy, Medium, Hard
        ])
        model.compile(optimizer=Adam(learning_rate=0.001),
                      loss='categorical_crossentropy', metrics=['accuracy'])
        self.model = model

    @staticmethod
    def extract_features(question, answer, source_sentence):
        """Extract features for difficulty classification."""
        q_words = question.split()
        a_words = answer.split()
        s_words = source_sentence.split()

        features = [
            len(q_words),                          # question length
            len(a_words),                          # answer length
            len(s_words),                          # source sentence length
            len(a_words) / max(len(s_words), 1),   # answer/source ratio
            len(q_words) / max(len(s_words), 1),   # question/source ratio
            sum(1 for c in question if c == ','),  # commas in question
            int(any(w in question.lower() for w in ['why', 'how', 'explain'])),  # reasoning question
            int(any(w in question.lower() for w in ['what', 'which', 'who'])),   # factual question
            int(any(w in question.lower() for w in ['when', 'where'])),          # recall question
            len(set(q_words) & set(s_words)) / max(len(q_words), 1),  # word overlap
            int(any(c.isdigit() for c in answer)),   # answer contains numbers
            len(answer),                             # answer char length
        ]
        return features

    @staticmethod
    def assign_heuristic_difficulty(answer, source_sentence):
        """
        Heuristic labels for training:
        Easy = short answer, high overlap with source
        Medium = medium answer length
        Hard = long answer, low overlap, reasoning needed
        """
        a_words = set(answer.lower().split())
        s_words = set(source_sentence.lower().split())
        overlap = len(a_words & s_words) / max(len(a_words), 1)

        if len(answer.split()) <= 2 and overlap > 0.5:
            return 0  # Easy
        elif len(answer.split()) <= 5:
            return 1  # Medium
        else:
            return 2  # Hard

    def train(self, questions, answers, source_sentences, labels=None,
              epochs=20, batch_size=32,
              checkpoint_path='data/saved_models/ann_best.keras'):
        """Train the classifier with EarlyStopping + ModelCheckpoint."""
        features = []
        final_labels = []

        for i in range(len(questions)):
            feat = self.extract_features(questions[i], answers[i], source_sentences[i])
            features.append(feat)
            if labels is not None:
                final_labels.append(labels[i])
            else:
                final_labels.append(
                    self.assign_heuristic_difficulty(answers[i], source_sentences[i])
                )

        X = np.array(features)
        X = self.scaler.fit_transform(X)
        y = to_categorical(np.array(final_labels), num_classes=3)

        self._build_model(X.shape[1])

        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=2,
                          restore_best_weights=True, verbose=1),
            ModelCheckpoint(checkpoint_path, monitor='val_loss',
                            save_best_only=True, verbose=1),
        ]

        history = self.model.fit(X, y, epochs=epochs, batch_size=batch_size,
                                 validation_split=0.1, verbose=1,
                                 callbacks=callbacks)
        return history.history

    def _ensure_loaded(self):
        if self.model is None:
            raise RuntimeError(
                "ANN classifier is not loaded. Call .load() or train() first."
            )

    def predict_difficulty(self, question, answer, source_sentence):
        """Predict difficulty for a single question."""
        self._ensure_loaded()
        feat = self.extract_features(question, answer, source_sentence)
        X = self.scaler.transform([feat])
        pred = self.model.predict(X, verbose=0)
        class_idx = np.argmax(pred[0])
        return DIFFICULTY_LABELS[class_idx], float(pred[0][class_idx])

    def predict_batch(self, questions, answers, source_sentences):
        """Predict difficulty for multiple questions."""
        features = [
            self.extract_features(q, a, s)
            for q, a, s in zip(questions, answers, source_sentences)
        ]
        X = self.scaler.transform(features)
        preds = self.model.predict(X, verbose=0)
        results = []
        for pred in preds:
            idx = np.argmax(pred)
            results.append((DIFFICULTY_LABELS[idx], float(pred[idx])))
        return results

    def save(self, model_dir='data/saved_models'):
        os.makedirs(model_dir, exist_ok=True)
        self.model.save(os.path.join(model_dir, 'ann_classifier.keras'))
        with open(os.path.join(model_dir, 'ann_scaler.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f)

    def load(self, model_dir='data/saved_models'):
        self.model = load_model(os.path.join(model_dir, 'ann_classifier.keras'))
        with open(os.path.join(model_dir, 'ann_scaler.pkl'), 'rb') as f:
            self.scaler = pickle.load(f)

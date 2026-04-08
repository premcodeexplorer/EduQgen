"""
LSTM Context Model
Trained on SQuAD v1.1 to understand sentence context and produce embeddings
that capture semantic meaning — used to enrich sentences before question generation.
"""

import os
import numpy as np
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Embedding, Bidirectional, GlobalMaxPooling1D, Dropout
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle


class LSTMContextModel:
    def __init__(self, max_words=20000, max_len=100, embedding_dim=128, lstm_units=32):
        self.max_words = max_words
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
        self.model = None
        self.encoder = None

    def _build_model(self):
        input_layer = Input(shape=(self.max_len,))
        x = Embedding(self.max_words, self.embedding_dim, input_length=self.max_len)(input_layer)
        # Halved LSTM units (64 -> 32) + recurrent_dropout for regularization
        x = Bidirectional(LSTM(self.lstm_units, return_sequences=True,
                               recurrent_dropout=0.2))(x)
        x = Dropout(0.3)(x)
        x = Bidirectional(LSTM(self.lstm_units, return_sequences=True,
                               recurrent_dropout=0.2))(x)
        x = Dropout(0.3)(x)
        encoded = GlobalMaxPooling1D()(x)

        # Classification head with L2 regularization
        output = Dense(32, activation='relu', kernel_regularizer=l2(0.001))(encoded)
        output = Dropout(0.3)(output)
        output = Dense(1, activation='sigmoid')(output)

        self.model = Model(input_layer, output)
        self.model.compile(optimizer=Adam(learning_rate=0.001),
                           loss='binary_crossentropy', metrics=['accuracy'])

        # Encoder extracts context embeddings
        self.encoder = Model(input_layer, encoded)

    def train_on_squad(self, contexts, labels, epochs=5, batch_size=32,
                       checkpoint_path='data/saved_models/lstm_best.keras'):
        """
        Train on SQuAD data with EarlyStopping + ModelCheckpoint to combat overfitting.
        contexts = list of sentences, labels = 1 if sentence contains an answer, else 0.
        """
        self.tokenizer.fit_on_texts(contexts)
        sequences = self.tokenizer.texts_to_sequences(contexts)
        padded = pad_sequences(sequences, maxlen=self.max_len, padding='post', truncating='post')

        self._build_model()

        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=2,
                          restore_best_weights=True, verbose=1),
            ModelCheckpoint(checkpoint_path, monitor='val_loss',
                            save_best_only=True, verbose=1),
        ]

        history = self.model.fit(padded, np.array(labels),
                                 epochs=epochs, batch_size=batch_size,
                                 validation_split=0.1, verbose=1,
                                 callbacks=callbacks)
        return history.history

    def _ensure_loaded(self):
        if self.model is None:
            raise RuntimeError(
                "LSTM model is not loaded. Call .load() or train_on_squad() first."
            )

    def get_context_scores(self, sentences):
        """Score sentences by how 'question-worthy' they are."""
        self._ensure_loaded()
        sequences = self.tokenizer.texts_to_sequences(sentences)
        padded = pad_sequences(sequences, maxlen=self.max_len, padding='post', truncating='post')
        scores = self.model.predict(padded, verbose=0).flatten()
        return scores

    def get_embeddings(self, sentences):
        """Get LSTM context embeddings for sentences."""
        self._ensure_loaded()
        sequences = self.tokenizer.texts_to_sequences(sentences)
        padded = pad_sequences(sequences, maxlen=self.max_len, padding='post', truncating='post')
        return self.encoder.predict(padded, verbose=0)

    def save(self, model_dir='data/saved_models'):
        os.makedirs(model_dir, exist_ok=True)
        self.model.save(os.path.join(model_dir, 'lstm_model.keras'))
        with open(os.path.join(model_dir, 'lstm_tokenizer.pkl'), 'wb') as f:
            pickle.dump(self.tokenizer, f)

    def load(self, model_dir='data/saved_models'):
        self.model = load_model(os.path.join(model_dir, 'lstm_model.keras'))
        # Rebuild encoder: take output of GlobalMaxPooling1D layer
        pool_layer = next(l for l in self.model.layers
                          if l.__class__.__name__ == 'GlobalMaxPooling1D')
        self.encoder = Model(self.model.input, pool_layer.output)
        with open(os.path.join(model_dir, 'lstm_tokenizer.pkl'), 'rb') as f:
            self.tokenizer = pickle.load(f)

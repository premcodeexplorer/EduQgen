"""
Autoencoder-based Key Sentence Extractor
Uses a shallow autoencoder to find the most important sentences from notes.
Sentences with highest reconstruction error are considered most "unique/important".
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
import nltk

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)


class SentenceAutoencoder:
    def __init__(self, encoding_dim=64, max_features=5000):
        self.encoding_dim = encoding_dim
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self.model = None
        self.encoder = None

    def _build_model(self, input_dim):
        input_layer = Input(shape=(input_dim,))
        # Encoder
        encoded = Dense(256, activation='relu')(input_layer)
        encoded = Dense(self.encoding_dim, activation='relu')(encoded)
        # Decoder
        decoded = Dense(256, activation='relu')(encoded)
        decoded = Dense(input_dim, activation='sigmoid')(decoded)

        self.model = Model(input_layer, decoded)
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        self.encoder = Model(input_layer, encoded)

    def extract_key_sentences(self, text, top_n=10, epochs=50):
        """
        Extract the most important sentences from text using autoencoder.
        Sentences with highest reconstruction error = most unique/important.
        """
        sentences = nltk.sent_tokenize(text)
        if not sentences:
            return [], []

        # Always fit the vectorizer so downstream callers can .transform() safely
        tfidf_matrix = self.vectorizer.fit_transform(sentences).toarray()

        # Always build + train the autoencoder, even for short inputs, so that
        # `self.model` is never None when downstream code calls .predict() on it.
        self._build_model(tfidf_matrix.shape[1])
        self.model.fit(
            tfidf_matrix, tfidf_matrix,
            epochs=epochs,
            batch_size=min(32, len(sentences)),
            shuffle=True,
            verbose=0
        )

        # Short text: just return everything in original order
        if len(sentences) <= top_n:
            return sentences, list(range(len(sentences)))

        # Calculate reconstruction error for each sentence
        reconstructed = self.model.predict(tfidf_matrix, verbose=0)
        errors = np.mean(np.square(tfidf_matrix - reconstructed), axis=1)

        # Higher error = more unique/important sentence
        top_indices = np.argsort(errors)[-top_n:]
        top_indices = sorted(top_indices)  # Keep original order

        key_sentences = [sentences[i] for i in top_indices]
        return key_sentences, list(top_indices)

    def get_embeddings(self, text):
        """Get encoder embeddings for sentences."""
        sentences = nltk.sent_tokenize(text)
        tfidf_matrix = self.vectorizer.transform(sentences).toarray()
        return self.encoder.predict(tfidf_matrix, verbose=0)

"""Application configuration."""

import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


class Config:
    SECRET_KEY = 'eduqgen-secret-key-2026'
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'data', 'uploads')
    SAVED_MODELS_DIR = os.path.join(BASE_DIR, 'data', 'saved_models')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max upload
    ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp', 'txt'}

    # Tesseract OCR path (Windows default install location)
    TESSERACT_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    # Question generation settings
    DEFAULT_NUM_QUESTIONS = 10
    MAX_NUM_QUESTIONS = 25

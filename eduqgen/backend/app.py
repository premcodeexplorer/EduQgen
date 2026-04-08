"""
EduQGen — Flask Application Entry Point
AI-powered Question & Answer Generator from Notes
"""

import sys
import os

# Add project root to path so imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask
from config import Config
from backend.routes import main, set_pipeline
from ml_models.question_generator import QuestionGeneratorPipeline


def create_app():
    app = Flask(
        __name__,
        template_folder=os.path.join(os.path.dirname(__file__), '..', 'templates'),
        static_folder=os.path.join(os.path.dirname(__file__), '..', 'static')
    )
    app.config.from_object(Config)

    # Ensure directories exist
    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(Config.SAVED_MODELS_DIR, exist_ok=True)

    # Initialize ML pipeline — fails fast if any model file is missing or corrupt
    print("Loading ML models... (this may take a minute on first run)")
    try:
        pipeline = QuestionGeneratorPipeline()
        pipeline.load_trained_models(Config.SAVED_MODELS_DIR)
    except Exception as e:
        print("\n" + "!" * 60)
        print("FATAL: Could not start EduQGen — ML pipeline failed to load.")
        print(f"Reason: {e}")
        print("Fix: run `python train_models.py` from the project root, then restart.")
        print("!" * 60 + "\n")
        raise
    set_pipeline(pipeline)
    print("Models loaded successfully!")

    # Register routes
    app.register_blueprint(main)

    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)

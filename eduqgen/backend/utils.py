"""
Utility functions for text extraction, PDF generation, and quiz history.
Handles: PDF (PyMuPDF), Images/Screenshots (Tesseract OCR), Plain text.
"""

import os
import json
from datetime import datetime
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from config import Config

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
HISTORY_FILE = os.path.join(BASE_DIR, 'data', 'history.json')
UPLOADS_FILE = os.path.join(BASE_DIR, 'data', 'uploads_log.json')


def extract_text_from_pdf(file_path):
    """Extract text from a PDF file using PyMuPDF."""
    text = ""
    doc = fitz.open(file_path)
    for page in doc:
        text += page.get_text()
    doc.close()
    return clean_text(text)


def extract_text_from_image(file_path):
    """Extract text from an image using Tesseract OCR."""
    pytesseract.pytesseract.tesseract_cmd = Config.TESSERACT_PATH
    img = Image.open(file_path)
    text = pytesseract.image_to_string(img)
    return clean_text(text)


def clean_text(text):
    """Clean extracted text: remove extra whitespace, fix encoding issues."""
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if line:
            cleaned_lines.append(line)
    text = ' '.join(cleaned_lines)
    while '  ' in text:
        text = text.replace('  ', ' ')
    return text.strip()


def process_upload(file, upload_dir='data/uploads'):
    """Process an uploaded file and extract text."""
    os.makedirs(upload_dir, exist_ok=True)
    filename = file.filename
    file_path = os.path.join(upload_dir, filename)
    file.save(file_path)

    ext = filename.rsplit('.', 1)[-1].lower()

    if ext == 'pdf':
        text = extract_text_from_pdf(file_path)
    elif ext in ('png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'):
        text = extract_text_from_image(file_path)
    elif ext == 'txt':
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = clean_text(f.read())
    else:
        text = ""

    return text


# ── Quiz History ──────────────────────────────────────────────

def load_history():
    """Load quiz history from JSON file."""
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []


def save_history(history):
    """Save quiz history to JSON file."""
    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def add_to_history(topic, score, total, difficulty_breakdown, results):
    """Add a completed quiz to history."""
    history = load_history()
    entry = {
        'id': len(history) + 1,
        'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'topic': topic[:80],  # Truncate long topics
        'score': score,
        'total': total,
        'percentage': round(score / max(total, 1) * 100),
        'difficulty_breakdown': difficulty_breakdown,
        'results': results
    }
    history.append(entry)
    save_history(history)
    return entry


def get_dashboard_stats():
    """Calculate dashboard statistics from history."""
    history = load_history()

    if not history:
        return {
            'total_quizzes': 0,
            'total_questions': 0,
            'avg_score': 0,
            'best_score': 0,
            'current_streak': 0,
            'scores_over_time': [],
            'dates': [],
            'difficulty_stats': {'Easy': {'correct': 0, 'total': 0},
                                 'Medium': {'correct': 0, 'total': 0},
                                 'Hard': {'correct': 0, 'total': 0}},
            'recent_quizzes': [],
            'history': []
        }

    total_quizzes = len(history)
    total_questions = sum(h['total'] for h in history)
    avg_score = round(sum(h['percentage'] for h in history) / total_quizzes)
    best_score = max(h['percentage'] for h in history)

    # Streak: consecutive quizzes with >= 60%
    streak = 0
    for h in reversed(history):
        if h['percentage'] >= 60:
            streak += 1
        else:
            break

    # Scores over time
    scores_over_time = [h['percentage'] for h in history]
    dates = [h['date'] for h in history]

    # Difficulty breakdown
    diff_stats = {'Easy': {'correct': 0, 'total': 0},
                  'Medium': {'correct': 0, 'total': 0},
                  'Hard': {'correct': 0, 'total': 0}}
    for h in history:
        for r in h.get('results', []):
            d = r.get('difficulty', 'Medium')
            if d in diff_stats:
                diff_stats[d]['total'] += 1
                if r.get('is_correct'):
                    diff_stats[d]['correct'] += 1

    return {
        'total_quizzes': total_quizzes,
        'total_questions': total_questions,
        'avg_score': avg_score,
        'best_score': best_score,
        'current_streak': streak,
        'scores_over_time': scores_over_time,
        'dates': dates,
        'difficulty_stats': diff_stats,
        'recent_quizzes': list(reversed(history)),
        'history': history
    }


# ── Upload Tracking ─────────────────────────────────────────

def load_uploads_log():
    """Load list of uploaded files."""
    if os.path.exists(UPLOADS_FILE):
        with open(UPLOADS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []


def save_uploads_log(uploads):
    """Save uploads list."""
    os.makedirs(os.path.dirname(UPLOADS_FILE), exist_ok=True)
    with open(UPLOADS_FILE, 'w', encoding='utf-8') as f:
        json.dump(uploads, f, ensure_ascii=False, indent=2)


def log_upload(filename, file_type, text_preview):
    """Log an uploaded file so users can re-use it."""
    uploads = load_uploads_log()
    # Don't duplicate same filename
    for u in uploads:
        if u['filename'] == filename:
            u['date'] = datetime.now().strftime('%Y-%m-%d %H:%M')
            u['use_count'] = u.get('use_count', 1) + 1
            save_uploads_log(uploads)
            return
    uploads.append({
        'id': len(uploads) + 1,
        'filename': filename,
        'file_type': file_type,
        'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'preview': text_preview[:120],
        'use_count': 1
    })
    save_uploads_log(uploads)


# ── PDF Generation ──────────────────────────────────────────

def generate_result_pdf(quiz_data, score, total):
    """Generate a PDF with quiz results using PyMuPDF."""
    doc = fitz.open()
    page = doc.new_page()

    y = 50
    page.insert_text((50, y), "EduQGen - Quiz Results", fontsize=20, fontname="helv")
    y += 35
    page.insert_text((50, y), f"Score: {score}/{total} ({round(score/max(total,1)*100)}%)",
                      fontsize=14, fontname="helv")
    y += 30

    for i, item in enumerate(quiz_data):
        if y > 750:
            page = doc.new_page()
            y = 50

        page.insert_text((50, y), f"Q{i+1}. [{item['difficulty']}] {item['question']}",
                          fontsize=10, fontname="helv")
        y += 18
        page.insert_text((70, y), f"Answer: {item['answer']}", fontsize=9, fontname="helv")
        y += 15
        page.insert_text((70, y), f"Your Answer: {item.get('user_answer', 'N/A')}",
                          fontsize=9, fontname="helv")
        y += 15

        status = "Correct" if item.get('is_correct') else "Wrong"
        page.insert_text((70, y), f"Result: {status}", fontsize=9, fontname="helv")
        y += 15

        source = item.get('source_sentence', '')
        if source:
            display_source = source[:100] + '...' if len(source) > 100 else source
            page.insert_text((70, y), f"Source: {display_source}", fontsize=8, fontname="helv")
            y += 18
        y += 10

    output_path = os.path.join(BASE_DIR, 'data', 'uploads', 'results.pdf')
    doc.save(output_path)
    doc.close()
    return output_path

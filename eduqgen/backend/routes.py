"""
Flask Routes — All API endpoints for EduQGen.
Uses server-side JSON storage for quiz data and persistent history.
"""

import os
import json
import uuid
import nltk
from flask import (
    Blueprint, render_template, request, jsonify, session, send_file, redirect, url_for
)
from backend.utils import (
    process_upload, clean_text, generate_result_pdf,
    add_to_history, get_dashboard_stats, load_history,
    log_upload, load_uploads_log, save_uploads_log
)
from config import Config

main = Blueprint('main', __name__)

pipeline = None

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
STORE_DIR = os.path.join(BASE_DIR, 'data', 'sessions')
METRICS_FILE = os.path.join(BASE_DIR, 'data', 'saved_models', 'training_metrics.json')
os.makedirs(STORE_DIR, exist_ok=True)


def set_pipeline(p):
    global pipeline
    pipeline = p


def _save_store(session_id, data):
    path = os.path.join(STORE_DIR, f'{session_id}.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)


def _load_store(session_id):
    path = os.path.join(STORE_DIR, f'{session_id}.json')
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def _get_session_id():
    if 'sid' not in session:
        session['sid'] = str(uuid.uuid4())
    return session['sid']


def _load_metrics():
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE, 'r') as f:
            return json.load(f)
    return {}


# ── Pages ──────────────────────────────────────────────

@main.route('/')
def index():
    return render_template('index.html')


@main.route('/dashboard')
def dashboard():
    stats = get_dashboard_stats()
    uploads = load_uploads_log()
    return render_template('dashboard.html', stats=stats, uploads=list(reversed(uploads)))


@main.route('/models')
def models_page():
    metrics = _load_metrics()
    return render_template('models.html', metrics=metrics)


@main.route('/analysis')
def analysis():
    sid = _get_session_id()
    store = _load_store(sid)
    analysis_data = store.get('analysis', {})
    if not analysis_data:
        return redirect(url_for('main.index'))
    return render_template('analysis.html', analysis=analysis_data)


@main.route('/quiz_detail/<int:quiz_id>')
def quiz_detail(quiz_id):
    """View the full details of a past quiz."""
    history = load_history()
    quiz_data = None
    for h in history:
        if h['id'] == quiz_id:
            quiz_data = h
            break
    if not quiz_data:
        return redirect(url_for('main.dashboard'))
    return render_template('quiz_detail.html', quiz=quiz_data)


# ── API ────────────────────────────────────────────────

@main.route('/api/history')
def api_history():
    stats = get_dashboard_stats()
    return jsonify(stats)

@main.route('/api/uploads')
def api_uploads():
    return jsonify(load_uploads_log())


@main.route('/api/history/<int:quiz_id>')
def api_history_detail(quiz_id):
    history = load_history()
    for h in history:
        if h['id'] == quiz_id:
            return jsonify(h)
    return jsonify({'error': 'Quiz not found'}), 404


@main.route('/upload', methods=['POST'])
def upload():
    """Handle file upload or text input, generate questions with analysis data."""
    # Hard guard: refuse to run if the pipeline isn't fully loaded.
    if pipeline is None or not pipeline.is_ready():
        return jsonify({
            'error': 'Models are not loaded on the server. '
                     'Run `python train_models.py` and restart the app.'
        }), 503

    text = ""

    upload_filename = None

    if request.form.get('text_input'):
        text = clean_text(request.form['text_input'])
    elif request.form.get('reuse_file'):
        # Re-use a previously uploaded file
        reuse = request.form['reuse_file']
        file_path = os.path.join(Config.UPLOAD_FOLDER, reuse)
        if os.path.exists(file_path):
            ext = reuse.rsplit('.', 1)[-1].lower()
            upload_filename = reuse
            if ext == 'pdf':
                from backend.utils import extract_text_from_pdf
                text = extract_text_from_pdf(file_path)
            elif ext in ('png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'):
                from backend.utils import extract_text_from_image
                text = extract_text_from_image(file_path)
            elif ext == 'txt':
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = clean_text(f.read())
        else:
            return jsonify({'error': 'File not found. It may have been deleted.'}), 400
    elif 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        ext = file.filename.rsplit('.', 1)[-1].lower()
        if ext not in Config.ALLOWED_EXTENSIONS:
            return jsonify({'error': f'File type .{ext} not supported'}), 400
        text = process_upload(file, Config.UPLOAD_FOLDER)
        upload_filename = file.filename
    else:
        return jsonify({'error': 'No input provided'}), 400

    # Log the upload
    if upload_filename:
        ftype = upload_filename.rsplit('.', 1)[-1].upper() if '.' in upload_filename else 'TXT'
        log_upload(upload_filename, ftype, text)

    if not text or len(text.strip()) < 50:
        return jsonify({'error': 'Text too short. Please provide more content (at least a paragraph).'}), 400

    num_q = request.form.get('num_questions', Config.DEFAULT_NUM_QUESTIONS)
    try:
        num_q = min(int(num_q), Config.MAX_NUM_QUESTIONS)
    except (ValueError, TypeError):
        num_q = Config.DEFAULT_NUM_QUESTIONS

    question_type = request.form.get('question_type', 'mcq')

    # ── Run the full pipeline with analysis tracking ──

    try:
        # Step 1: Autoencoder extracts key sentences
        all_sentences = nltk.sent_tokenize(text)
        key_sentences, indices = pipeline.autoencoder.extract_key_sentences(
            text, top_n=min(num_q * 2, 20)
        )

        # Get reconstruction errors for scoring display.
        # extract_key_sentences() always fits the vectorizer and trains the model,
        # but we still guard here so a degenerate input can't crash the route.
        import numpy as np
        if (pipeline.autoencoder.model is not None
                and key_sentences and hasattr(pipeline.autoencoder.vectorizer, 'idf_')):
            tfidf_matrix = pipeline.autoencoder.vectorizer.transform(key_sentences).toarray()
            reconstructed = pipeline.autoencoder.model.predict(tfidf_matrix, verbose=0)
            ae_errors = np.mean(np.square(tfidf_matrix - reconstructed), axis=1)
        else:
            ae_errors = np.zeros(len(key_sentences))
        max_err = float(ae_errors.max()) if len(ae_errors) and ae_errors.max() > 0 else 1.0

        # Step 2: LSTM context scores
        try:
            lstm_scores = pipeline.lstm.get_context_scores(key_sentences)
        except Exception:
            lstm_scores = [0.5] * len(key_sentences)

        # Build analysis data for key sentences
        analysis_sentences = []
        for i, sent in enumerate(key_sentences):
            ae_score = float(ae_errors[i])
            ls = float(lstm_scores[i]) if i < len(lstm_scores) else 0.5
            analysis_sentences.append({
                'text': sent,
                'ae_score': round(ae_score, 4),
                'ae_score_pct': round((ae_score / max_err) * 100, 1),
                'lstm_score': round(ls, 4)
            })

        # Sort by combined importance
        analysis_sentences.sort(key=lambda x: x['ae_score'] + x['lstm_score'], reverse=True)

        # Step 3 & 4: Generate questions via full pipeline
        questions = pipeline.generate_questions(text, num_questions=num_q, question_type=question_type)

    except Exception as e:
        return jsonify({'error': f'Question generation failed: {str(e)}'}), 500

    if not questions:
        return jsonify({'error': 'Could not generate questions from this text. Try longer notes.'}), 400

    topic = text[:80].split('.')[0].strip()

    # Build analysis object
    analysis_data = {
        'total_sentences': len(all_sentences),
        'key_sentences': analysis_sentences[:15],
        'num_questions': len(questions),
        'word_count': len(text.split()),
        'questions': questions
    }

    sid = _get_session_id()
    _save_store(sid, {
        'questions': questions,
        'original_text': text,
        'topic': topic,
        'question_type': question_type,
        'analysis': analysis_data
    })

    return jsonify({'success': True, 'num_questions': len(questions)})


@main.route('/quiz')
def quiz():
    sid = _get_session_id()
    store = _load_store(sid)
    questions = store.get('questions', [])
    if not questions:
        return redirect(url_for('main.index'))
    return render_template('quiz.html', questions=questions)


@main.route('/delete_upload', methods=['POST'])
def delete_upload():
    """Delete an uploaded file and its log entry."""
    data = request.get_json() or {}
    filename = data.get('filename')
    if not filename:
        return jsonify({'error': 'Missing filename'}), 400

    # Remove the file on disk (best-effort)
    file_path = os.path.join(Config.UPLOAD_FOLDER, filename)
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception:
        pass

    uploads = load_uploads_log()
    uploads = [u for u in uploads if u.get('filename') != filename]
    save_uploads_log(uploads)
    return jsonify({'success': True})


@main.route('/retake_quiz')
def retake_quiz():
    """Clear previous results but keep questions, so the user can retake."""
    sid = _get_session_id()
    store = _load_store(sid)
    if not store.get('questions'):
        return redirect(url_for('main.index'))
    store.pop('results', None)
    store.pop('score', None)
    store.pop('total', None)
    _save_store(sid, store)
    return redirect(url_for('main.quiz'))


@main.route('/submit_quiz', methods=['POST'])
def submit_quiz():
    """Process quiz answers, calculate score, save to history."""
    sid = _get_session_id()
    store = _load_store(sid)
    questions = store.get('questions', [])
    if not questions:
        return jsonify({'error': 'No active quiz'}), 400

    data = request.get_json()
    user_answers = data.get('answers', {})

    score = 0
    results = []
    diff_breakdown = {'Easy': 0, 'Medium': 0, 'Hard': 0}

    for i, q in enumerate(questions):
        user_ans = user_answers.get(str(i), '')
        is_correct = user_ans.strip().lower() == q['answer'].strip().lower()
        if is_correct:
            score += 1

        d = q.get('difficulty', 'Medium')
        if d in diff_breakdown:
            diff_breakdown[d] += 1

        results.append({
            'question': q['question'],
            'answer': q['answer'],
            'user_answer': user_ans,
            'is_correct': is_correct,
            'difficulty': d,
            'source_sentence': q['source_sentence'],
            'options': q['options']
        })

    store['results'] = results
    store['score'] = score
    store['total'] = len(questions)
    _save_store(sid, store)

    topic = store.get('topic', 'Untitled Quiz')
    add_to_history(topic, score, len(questions), diff_breakdown, results)

    return jsonify({
        'score': score,
        'total': len(questions),
        'results': results
    })


@main.route('/results')
def results():
    sid = _get_session_id()
    store = _load_store(sid)
    results_data = store.get('results', [])
    score = store.get('score', 0)
    total = store.get('total', 0)
    if not results_data:
        return redirect(url_for('main.index'))
    return render_template('results.html', results=results_data, score=score, total=total)


@main.route('/download_pdf')
def download_pdf():
    sid = _get_session_id()
    store = _load_store(sid)
    results_data = store.get('results', [])
    score = store.get('score', 0)
    total = store.get('total', 0)
    if not results_data:
        return redirect(url_for('main.index'))
    pdf_path = generate_result_pdf(results_data, score, total)
    return send_file(pdf_path, as_attachment=True, download_name='eduqgen_results.pdf')

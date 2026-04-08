"""
Question Generator Pipeline
Uses T5 (valhalla/t5-small-qg-hl) to generate questions from highlighted sentences.
Supports multiple question types: MCQ, Descriptive, True/False, Multiple Select, Numerical.
Orchestrates the full pipeline: Autoencoder → LSTM → T5 → ANN.
"""

import os
import random
import re
import nltk
from transformers import T5ForConditionalGeneration, T5Tokenizer

from ml_models.autoencoder import SentenceAutoencoder
from ml_models.lstm_model import LSTMContextModel
from ml_models.ann_classifier import DifficultyClassifier

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)


# A noun phrase is a run of adjectives/nouns/proper nouns/cardinal numbers,
# optionally preceded by a determiner. This grammar drives the regex chunker.
_NP_GRAMMAR = r"""
    NP: {<DT>?<JJ.*>*<NN.*|CD>+}
"""
_NP_CHUNKER = nltk.RegexpParser(_NP_GRAMMAR)

_BAD_HEAD_WORDS = {
    'thing', 'things', 'way', 'ways', 'kind', 'kinds', 'type', 'types',
    'time', 'times', 'fact', 'facts', 'lot', 'one', 'two', 'three',
    'example', 'examples', 'something', 'someone', 'it', 'them', 'they',
    'who', 'what', 'which', 'where', 'when', 'why', 'how',
}

# Question is rejected if it's shorter than this many words (T5 sometimes
# emits 2-3 word fragments that aren't real questions).
_MIN_QUESTION_WORDS = 5


class QuestionGeneratorPipeline:
    def __init__(self):
        self.autoencoder = SentenceAutoencoder()
        self.lstm = LSTMContextModel()
        self.classifier = DifficultyClassifier()

        self.t5_tokenizer = T5Tokenizer.from_pretrained('valhalla/t5-base-qg-hl')
        self.t5_model = T5ForConditionalGeneration.from_pretrained('valhalla/t5-base-qg-hl')

        # Pool of candidate noun phrases extracted from the FULL text — used as
        # distractors so they aren't recycled from a tiny set of answers.
        self._np_pool = []

    @staticmethod
    def _clean_phrase(phrase):
        phrase = re.sub(r'\s+', ' ', phrase).strip(' .,;:!?()[]"\'')
        return phrase

    def _extract_noun_phrases(self, sentence):
        """Return a list of noun phrases (strings) found in the sentence,
        ordered by appearance, filtering out junk."""
        try:
            tokens = nltk.word_tokenize(sentence)
            tagged = nltk.pos_tag(tokens)
            tree = _NP_CHUNKER.parse(tagged)
        except Exception:
            return []

        phrases = []
        for subtree in tree.subtrees(filter=lambda t: t.label() == 'NP'):
            words = [w for w, _ in subtree.leaves()]
            # Strip leading determiners (a/the/an/this/those)
            if words and words[0].lower() in {'a', 'an', 'the', 'this', 'that', 'these', 'those'}:
                words = words[1:]
            if not words:
                continue
            phrase = self._clean_phrase(' '.join(words))
            if not phrase or len(phrase) < 3:
                continue
            head = phrase.split()[-1].lower()
            if head in _BAD_HEAD_WORDS:
                continue
            # Keep only phrases with at least one alphabetic character
            if not any(c.isalpha() for c in phrase):
                continue
            phrases.append(phrase)
        return phrases

    def _build_distractor_pool(self, full_text):
        """Extract a diverse pool of candidate noun phrases from the entire
        source text, used to seed MCQ distractors."""
        try:
            sentences = nltk.sent_tokenize(full_text)
        except Exception:
            sentences = [full_text]

        seen = set()
        pool = []
        for sent in sentences:
            for np in self._extract_noun_phrases(sent):
                key = np.lower()
                if key in seen:
                    continue
                seen.add(key)
                pool.append(np)
        self._np_pool = pool

    def _extract_answer(self, sentence):
        """Pick the best noun phrase from the sentence as the MCQ answer.

        Strategy: prefer noun phrases that appear in the LATTER half of the
        sentence (objects/complements carry the new information) AND are
        multi-word. The subject NP at the start is usually the topic, not the
        answer the student should recall."""
        nps = self._extract_noun_phrases(sentence)
        if nps:
            sent_lower = sentence.lower()
            half = len(sentence) // 2

            def score(p):
                pos = sent_lower.find(p.lower())
                # Higher score = better answer candidate
                multi = 1 if len(p.split()) >= 2 else 0
                late = 1 if pos >= half else 0
                return (multi + late, len(p), pos)

            nps.sort(key=score, reverse=True)
            return nps[0]

        # Fallback: longest non-stopword word
        from nltk.corpus import stopwords
        stop_words = set(stopwords.words('english'))
        words = [w.strip('.,!?;:()[]"\'') for w in sentence.split()]
        words = [w for w in words if w and w.lower() not in stop_words and len(w) > 3]
        if words:
            words.sort(key=len, reverse=True)
            return words[0]
        return sentence.split()[0] if sentence.split() else sentence

    def _extract_number_answer(self, sentence):
        """Extract a numeric value from the sentence for numerical questions."""
        numbers = re.findall(r'\b\d+(?:\.\d+)?(?:\s*%|\s*[a-zA-Z]+)?\b', sentence)
        if numbers:
            return numbers[0]
        return None

    @staticmethod
    def _make_cloze(sentence, answer):
        """Build a fill-in-the-blank question by hiding the answer in the sentence.
        Always reads naturally — used as a fallback when T5 output is awkward."""
        # Case-insensitive replace, only the first occurrence
        pattern = re.compile(re.escape(answer), re.IGNORECASE)
        blanked = pattern.sub('______', sentence, count=1)
        blanked = blanked.rstrip('.!?') + '.'
        return f"Fill in the blank: {blanked}"

    def _is_bad_question(self, question, answer, sentence):
        """Heuristics for T5 outputs we should NOT show to the user."""
        if not question or len(question.split()) < _MIN_QUESTION_WORDS:
            return True
        # Ends in "<word> what?" — fragmentary
        if re.search(r'\b\w+\s+what\s*\??$', question, flags=re.IGNORECASE):
            return True
        # Contains the answer literally — gives it away
        if answer.lower() in question.lower():
            return True
        # Doesn't actually start with a question word
        first = question.lstrip().split()[0].lower().rstrip(',?')
        if first not in {'what', 'which', 'who', 'where', 'when', 'why',
                          'how', 'name', 'identify', 'in', 'from'}:
            return True
        return False

    def _generate_question(self, sentence, answer):
        """Use T5 to generate a question; fall back to a cloze (fill-in-the-blank)
        whenever T5's output looks malformed."""
        highlighted = sentence.replace(answer, f'<hl> {answer} <hl>', 1)
        input_text = f"generate question: {highlighted}"

        input_ids = self.t5_tokenizer.encode(input_text, return_tensors='pt', max_length=512, truncation=True)
        outputs = self.t5_model.generate(
            input_ids,
            max_length=72,
            num_beams=5,
            no_repeat_ngram_size=2,
            early_stopping=True
        )
        question = self.t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
        question = self._clean_question(question, answer)

        # Quality gate — if T5 produced junk, fall back to a clean cloze.
        if self._is_bad_question(question, answer, sentence):
            return self._make_cloze(sentence, answer)
        return question

    @staticmethod
    def _clean_question(question, answer):
        """Tidy T5 output: fix capitalization, strip dangling 'what?', ensure '?'."""
        q = question.strip()

        # Replace awkward trailing "is what?" / "is not what?" patterns
        q = re.sub(r'\s+is\s+what\s*\??$', '?', q, flags=re.IGNORECASE)
        q = re.sub(r'\s+are\s+what\s*\??$', '?', q, flags=re.IGNORECASE)
        q = re.sub(r'\s+was\s+what\s*\??$', '?', q, flags=re.IGNORECASE)
        q = re.sub(r'\s+can\s+be\s+what\s*\??$', '?', q, flags=re.IGNORECASE)
        q = re.sub(r'\s+do\s+what\s*\??$', '?', q, flags=re.IGNORECASE)
        q = re.sub(r'\s+is\s+not\s+what\s*\??$', '?', q, flags=re.IGNORECASE)

        # Collapse whitespace and ensure trailing '?'
        q = re.sub(r'\s+', ' ', q).strip()
        if q and not q.endswith('?'):
            q = q.rstrip('.!') + '?'

        # Capitalize first letter
        if q:
            q = q[0].upper() + q[1:]
        return q

    def _generate_distractors(self, answer, all_answers):
        """Generate 3 MCQ distractors that are:
           - drawn from the full-text noun-phrase pool (diverse, not recycled)
           - not equal to or contained in the answer
           - roughly similar in word-count when possible
           - never duplicated within the same question."""
        ans_lower = answer.lower().strip()
        ans_words = set(ans_lower.split())
        target_len = max(1, len(ans_lower.split()))
        ans_is_multi = target_len >= 2

        def is_valid(cand, require_multi=False):
            c = cand.lower().strip()
            if not c or c == ans_lower:
                return False
            if c in ans_lower or ans_lower in c:
                return False
            cwords = c.split()
            if require_multi and len(cwords) < 2:
                return False
            # Reject if it's mostly the same words
            cwset = set(cwords)
            if cwset and len(cwset & ans_words) / len(cwset) > 0.5:
                return False
            return True

        # Two-pass scoring: if answer is multi-word, first try only multi-word
        # candidates so distractors look like real alternatives, not stray nouns.
        def collect(require_multi):
            scored = []
            for cand in self._np_pool:
                if not is_valid(cand, require_multi=require_multi):
                    continue
                score = -abs(len(cand.split()) - target_len)
                scored.append((score, cand))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [c for _, c in scored[:15]]

        top = collect(require_multi=ans_is_multi)
        if len(top) < 3 and ans_is_multi:
            # Fall back to allowing single-word distractors if pool is thin
            top = collect(require_multi=False)

        picked = []
        seen = {ans_lower}
        random.shuffle(top)
        for cand in top:
            if cand.lower() in seen:
                continue
            picked.append(cand)
            seen.add(cand.lower())
            if len(picked) == 3:
                break

        # Backstop: pad from other answers if pool was thin
        if len(picked) < 3:
            for cand in all_answers:
                if not is_valid(cand) or cand.lower() in seen:
                    continue
                picked.append(cand)
                seen.add(cand.lower())
                if len(picked) == 3:
                    break

        # Last resort: synthesize generic terms so the question still has 4 options
        generic_fallbacks = ['none of the above', 'all of the above',
                             'cannot be determined', 'not applicable']
        i = 0
        while len(picked) < 3 and i < len(generic_fallbacks):
            if generic_fallbacks[i] not in seen:
                picked.append(generic_fallbacks[i])
                seen.add(generic_fallbacks[i])
            i += 1

        return picked[:3]

    def _generate_numerical_distractors(self, answer_str):
        """Generate plausible wrong numerical options."""
        try:
            num = float(re.findall(r'[\d.]+', answer_str)[0])
            suffix = answer_str.replace(str(num), '').strip()
            offsets = [0.5, 1.5, 2.0, 0.75, 1.25, 3.0]
            random.shuffle(offsets)
            distractors = []
            for off in offsets:
                if random.random() > 0.5:
                    val = num * off
                else:
                    val = num + (num * (off - 1))
                val = round(val, 2) if '.' in str(num) else int(round(val))
                d = f"{val} {suffix}".strip() if suffix else str(val)
                if d != answer_str and d not in distractors:
                    distractors.append(d)
                if len(distractors) >= 3:
                    break
            return distractors
        except Exception:
            return [str(random.randint(1, 100)) for _ in range(3)]

    def _format_as_type(self, question, answer, sentence, all_answers, question_type, difficulty, confidence):
        """Format a generated question into the requested type."""

        base = {
            'question': question,
            'answer': answer,
            'difficulty': difficulty,
            'confidence': round(float(confidence), 2),
            'source_sentence': sentence,
            'question_type': question_type
        }

        if question_type == 'mcq':
            distractors = self._generate_distractors(answer, all_answers)
            options = [answer] + distractors
            random.shuffle(options)
            base['options'] = options

        elif question_type == 'true_false':
            # Create a statement from the sentence — randomly make it true or false
            is_true = random.choice([True, False])
            if is_true:
                statement = sentence
                base['answer'] = 'True'
            else:
                # Create a false version by swapping the answer with a distractor
                distractors = self._generate_distractors(answer, all_answers)
                if distractors:
                    false_sentence = sentence.replace(answer, distractors[0], 1)
                    statement = false_sentence
                else:
                    statement = sentence
                    is_true = True
                    base['answer'] = 'True'
                if not is_true:
                    base['answer'] = 'False'

            base['question'] = f"True or False: {statement}"
            base['options'] = ['True', 'False']
            base['explanation'] = f"Original: {sentence}"

        elif question_type == 'descriptive':
            # Convert to a descriptive/open-ended question
            desc_starters = [
                f"Explain the following: {question}",
                f"Describe in detail: {question}",
                f"In your own words, {question[0].lower()}{question[1:]}",
            ]
            base['question'] = random.choice(desc_starters)
            base['answer'] = sentence  # Full sentence is the expected answer
            base['options'] = []  # No options for descriptive

        elif question_type == 'multi_select':
            # Create a multi-select with 2 correct + 2 wrong
            distractors = self._generate_distractors(answer, all_answers)
            # Split the answer to create a second correct option from the source
            words = answer.split()
            if len(words) > 2:
                second_correct = ' '.join(words[:len(words)//2])
            else:
                second_correct = answer
            correct_answers = [answer, second_correct]
            all_opts = correct_answers + distractors[:2]
            random.shuffle(all_opts)
            base['question'] = f"Select ALL that apply: {question}"
            base['answer'] = ' | '.join(correct_answers)
            base['options'] = all_opts
            base['correct_options'] = correct_answers

        elif question_type == 'numerical':
            # Try to find a number in the sentence
            num_answer = self._extract_number_answer(sentence)
            if num_answer:
                base['answer'] = num_answer
                distractors = self._generate_numerical_distractors(num_answer)
                options = [num_answer] + distractors
                random.shuffle(options)
                base['options'] = options
            else:
                # Fallback to MCQ if no number found
                distractors = self._generate_distractors(answer, all_answers)
                options = [answer] + distractors
                random.shuffle(options)
                base['options'] = options
                base['question_type'] = 'mcq'

        return base

    def generate_questions(self, text, num_questions=10, question_type='mcq'):
        """
        Full pipeline:
        1. Autoencoder extracts key sentences
        2. LSTM scores sentences for question-worthiness
        3. T5 generates questions
        4. ANN classifies difficulty
        5. Format into requested question type
        """
        # Step 0: Build a global noun-phrase pool from the full text — feeds
        # diverse distractors so the same 4 phrases aren't recycled.
        self._build_distractor_pool(text)

        # Step 1: Extract key sentences
        key_sentences, indices = self.autoencoder.extract_key_sentences(
            text, top_n=min(num_questions * 2, 20)
        )

        # Step 2: Score with LSTM — keep ALL key sentences ranked, so we can
        # over-generate and drop low-quality questions while still hitting
        # num_questions.
        try:
            scores = self.lstm.get_context_scores(key_sentences)
            scored = list(zip(key_sentences, scores, indices))
            scored.sort(key=lambda x: x[1], reverse=True)
            selected = scored
        except Exception:
            selected = [(s, 1.0, i) for s, i in zip(key_sentences, indices)]

        # Step 3: Pre-extract answers for the full candidate pool — feeds the
        # distractor backstop with realistic alternatives.
        all_answers = []
        cand_answers = []
        for sentence, score, idx in selected:
            ans = self._extract_answer(sentence)
            cand_answers.append(ans)
            if ans and len(ans.split()) >= 2:
                all_answers.append(ans)

        qa_pairs = []
        used_questions = set()

        for i, (sentence, score, idx) in enumerate(selected):
            if len(qa_pairs) >= num_questions:
                break
            answer = cand_answers[i]
            # Skip junk answers — too short or just a stopword-ish single token
            if not answer or len(answer) < 3:
                continue
            question = self._generate_question(sentence, answer)

            # Quality gate: reject empty / too-short / duplicate questions
            qkey = question.lower().strip()
            if qkey in used_questions:
                continue
            used_questions.add(qkey)

            # Step 4: Classify difficulty
            try:
                difficulty, confidence = self.classifier.predict_difficulty(
                    question, answer, sentence
                )
            except Exception:
                difficulty = 'Medium'
                confidence = 0.5

            # Step 5: Format into requested type
            qa = self._format_as_type(
                question, answer, sentence, all_answers,
                question_type, difficulty, confidence
            )
            qa['source_index'] = int(idx)
            qa_pairs.append(qa)

        return qa_pairs

    # ── Required model files (relative to model_dir) ──
    REQUIRED_FILES = {
        'LSTM model':     'lstm_model.keras',
        'LSTM tokenizer': 'lstm_tokenizer.pkl',
        'ANN classifier': 'ann_classifier.keras',
        'ANN scaler':     'ann_scaler.pkl',
    }

    def load_trained_models(self, model_dir='data/saved_models'):
        """
        Strict loader: verifies every required model file exists, loads them,
        and raises a clear RuntimeError on the FIRST failure. No silent fallback.
        Call this once at app startup so the server fails fast on a bad install.
        """
        print("=" * 60)
        print(f"[startup] Verifying model files in: {os.path.abspath(model_dir)}")
        print("=" * 60)

        # 1. Existence check — print status for every required file
        missing = []
        for label, fname in self.REQUIRED_FILES.items():
            path = os.path.join(model_dir, fname)
            exists = os.path.exists(path)
            status = "OK   " if exists else "MISS "
            print(f"  [{status}] {label:<16} -> {path}")
            if not exists:
                missing.append(path)

        if missing:
            raise RuntimeError(
                "Missing model files — run `python train_models.py` first.\n"
                "Missing:\n  - " + "\n  - ".join(missing)
            )

        # 2. Actually load — any exception propagates so the server won't start
        #    in a half-loaded state.
        try:
            self.lstm.load(model_dir)
            assert self.lstm.model is not None, "LSTM .load() returned without setting .model"
            print("[startup] LSTM model loaded and ready")
        except Exception as e:
            raise RuntimeError(f"Failed to load LSTM model: {e}") from e

        try:
            self.classifier.load(model_dir)
            assert self.classifier.model is not None, "ANN .load() returned without setting .model"
            print("[startup] ANN classifier loaded and ready")
        except Exception as e:
            raise RuntimeError(f"Failed to load ANN classifier: {e}") from e

        print("[startup] All models loaded successfully")
        print("=" * 60)

    def is_ready(self):
        """True only when both trainable models are loaded and usable."""
        return (
            self.lstm is not None and self.lstm.model is not None
            and self.classifier is not None and self.classifier.model is not None
            and self.t5_model is not None
        )

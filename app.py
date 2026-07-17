from flask import Flask, render_template, request, jsonify, session
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import secrets
import datetime
import json
import uuid
import os
import re
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import timedelta
import spacy
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from textblob import TextBlob
from dotenv import load_dotenv, set_key
from google import genai

ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
load_dotenv(ENV_PATH)

# Download required NLTK data
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

try:
    nltk.data.find('punkt')
except LookupError:
    nltk.download('punkt')

app = Flask(__name__)

flask_secret_key = os.environ.get('FLASK_SECRET_KEY')
if not flask_secret_key:
    flask_secret_key = secrets.token_hex(32)
    set_key(ENV_PATH, 'FLASK_SECRET_KEY', flask_secret_key)
app.secret_key = flask_secret_key
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///mindcare.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.permanent_session_lifetime = timedelta(hours=24)

# Initialize extensions
db = SQLAlchemy(app)
CORS(app)

# Load spaCy model (optional; if missing, key_phrase personalization is skipped)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("spaCy English model not found. Install with: python -m spacy download en_core_web_sm")
    nlp = None

# Database Models
class User(db.Model):
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    conversations = db.relationship('Conversation', backref='user', lazy=True)

class Conversation(db.Model):
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(db.String(36), db.ForeignKey('user.id'), nullable=True)
    session_id = db.Column(db.String(64), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    messages = db.relationship('Message', backref='conversation', lazy=True)

class Message(db.Model):
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    conversation_id = db.Column(db.String(36), db.ForeignKey('conversation.id'), nullable=False)
    sender = db.Column(db.String(10), nullable=False)
    content = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    emotion = db.Column(db.String(50))
    sentiment_score = db.Column(db.Float)
    confidence = db.Column(db.Float)
    key_phrases = db.Column(db.Text)
    therapy_technique = db.Column(db.String(100))

class AITherapyBot:
    def __init__(self):
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.gemini_model = "gemini-3-flash-preview"

        # Crisis responses stay hardcoded (not AI-generated) so they're always
        # accurate and never depend on an LLM call succeeding.
        self.crisis_responses = [
            "Please reach out to the BC Crisis Centre right now. They're available 24/7 at 1-800-784-2433. If you're in immediate danger, please call 911.",
            "Your life matters. Please call 1-800-784-2433 (BC Crisis Centre, 24/7) or 911 if you're in immediate danger. You don't have to face this alone.",
            "Please call the BC Crisis Centre at 1-800-784-2433 right now. They're there any time of day or night.",
            "Please call 1-800-784-2433 right now. Is there someone with you who can stay by your side?",
        ]

        self.fallback_responses = [
            "I'm here with you. Can you tell me a bit more about what's going on?",
            "I want to understand. Can you say more about what you're feeling?",
        ]

        self.crisis_keywords = [
            "suicide", "suicidal", "kill myself", "killing myself", "end it all",
            "end my life", "ending my life", "ending it all", "take my own life",
            "taking my own life", "hurt myself", "hurting myself", "harm myself",
            "harming myself", "self harm", "self-harm", "self injury", "self-injury",
            "want to die", "wanting to die", "wish i was dead", "wish i were dead",
            "better off dead", "better off without me", "no point living",
            "no point in living", "no reason to live", "no reason to go on",
            "nothing to live for", "no will to live", "can't go on", "cant go on",
            "can't take it anymore", "cant take it anymore", "cut myself",
            "cutting myself", "overdose", "not worth living", "tired of living",
            "tired of being alive", "don't want to be alive", "dont want to be alive",
            "don't want to live anymore", "dont want to live anymore",
            "want to disappear forever", "want to end my life", "planning to kill myself",
            "thinking about suicide", "thoughts of suicide", "suicidal thoughts",
            "world would be better without me", "no future for me",
        ]

        # Common idioms that would otherwise false-positive on the keyword list
        # above (e.g. "cut myself some slack" should never read as self-harm).
        # This only trims known-safe phrasing; it never suppresses a match
        # based on negation, since missing a real crisis is far worse than a
        # false alarm.
        self.crisis_safe_phrases = [
            "cut myself some slack", "kill it at", "kill two birds",
            "kill time", "killing it", "dying to see", "dying to know",
            "dead tired", "dead serious", "die laughing", "died laughing",
        ]

        self.emotion_keywords = {
            "happy": [
                "happy", "happiness", "joy", "joyful", "great", "amazing", "wonderful",
                "good mood", "feeling good", "feel good", "doing well", "doing great",
            ],
            "excited": [
                "excited", "excitement", "thrilled", "pumped", "stoked", "eager",
                "enthusiastic", "can't wait", "hyped", "fired up",
            ],
            "grateful": [
                "grateful", "gratitude", "thankful", "blessed", "appreciative",
                "appreciate", "fortunate", "lucky",
            ],
            "motivated": [
                "motivated", "motivation", "driven", "focused", "determined",
                "productive", "on fire", "inspired", "inspiration",
            ],
            "confident": [
                "confident", "confidence", "strong", "capable", "self-assured",
                "powerful", "unstoppable", "believe in myself",
            ],
            "calm": [
                "calm", "peaceful", "relaxed", "at peace", "serene", "tranquil",
                "centred", "centered", "settled", "content",
            ],
            "hopeful": [
                "hopeful", "hope", "optimistic", "optimism", "looking forward",
                "positive", "better days", "things will get better",
            ],
            "proud": [
                "proud", "pride", "accomplished", "achievement", "succeeded",
                "did it", "i did it", "achieved",
            ],
            "sad": [
                "sad", "sadness", "unhappy", "miserable", "down", "low", "blue",
                "upset", "heartbroken", "heartbreak", "crying", "cried", "tears",
                "feel awful", "feel terrible",
            ],
            "depressed": [
                "depressed", "depression", "hopeless", "worthless", "empty", "numb",
                "pointless", "nothing matters", "no energy", "no motivation", "unmotivated",
                "dark", "hollow", "broken", "giving up", "no hope",
                "cant feel anything", "can't feel", "low mood", "feel nothing",
                "joyless", "bleak", "what's the point",
            ],
            "anxious": [
                "anxious", "anxiety", "worried", "worrying", "panic", "panicking",
                "fear", "scared", "terrified", "dread", "dreading", "on edge",
                "cant relax", "can't relax", "racing thoughts", "overthinking",
                "tense", "restless", "uneasy", "freaking out", "heart racing",
            ],
            "nervous": [
                "nervous", "nervousness", "jitters", "butterflies", "nerve-wracking",
                "nerve wracking", "apprehensive", "dreading",
            ],
            "stressed": [
                "stressed", "stress", "pressure", "burnout", "burnt out", "burned out",
                "too much", "cant cope", "can't cope", "stretched thin", "no time",
                "frantic", "swamped", "drowning in", "deadline", "workload", "overloaded",
            ],
            "overwhelmed": [
                "overwhelmed", "too much going on", "cant handle", "can't handle",
                "falling apart", "breaking down", "crumbling", "cant keep up",
                "can't keep up", "everything at once", "all at once", "spinning",
                "scattered", "too much on my plate",
            ],
            "angry": [
                "angry", "anger", "mad", "furious", "frustrated", "frustration",
                "irritated", "irritable", "annoyed", "resentful", "resentment",
                "bitter", "rage", "raging", "livid", "pissed", "fed up", "had enough",
            ],
            "lonely": [
                "lonely", "loneliness", "alone", "isolated", "isolation", "abandoned",
                "no one cares", "nobody cares", "no friends", "disconnected", "left out",
                "invisible", "no one understands", "nobody understands", "by myself",
                "no one to talk to",
            ],
            "tired": [
                "tired", "exhausted", "drained", "fatigued", "worn out", "sleepy",
                "no energy", "burnt out", "burned out", "running on empty", "depleted",
            ],
            "lost": [
                "lost", "directionless", "no purpose", "no direction",
                "don't know what to do", "stuck", "no idea what i want",
                "no idea where i'm going",
            ],
            "confused": [
                "confused", "confusion", "unsure", "uncertain", "don't understand",
                "unclear", "can't figure out", "can't decide",
            ],
            "bored": [
                "bored", "boredom", "nothing to do", "dull", "uninterested",
                "uninspired", "restless",
            ],
            "grief": [
                "grief", "grieving", "grieve", "loss", "lost someone", "lost my",
                "died", "dying", "dead", "death", "passed away", "passed on",
                "put down", "put to sleep", "bereavement", "mourning",
                "miss them", "missing them", "miss him", "miss her", "miss my",
                "heartbroken", "devastating", "devastated",
                "my dog", "my cat", "my pet", "my mom", "my dad", "my father",
                "my mother", "my grandmother", "my grandfather", "my grandma",
                "my grandpa", "dealing with loss", "lost a loved",
            ],
        }

    def analyze_emotional_state(self, text):
        results = {
            'primary_emotion': 'neutral',
            'sentiment': 0.0,
            'key_phrases': [],
            'crisis_indicators': [],
        }

        vader = self.sentiment_analyzer.polarity_scores(text)
        results['sentiment'] = vader['compound']
        text_lower = text.lower()

        crisis_scan_text = text_lower
        for phrase in self.crisis_safe_phrases:
            crisis_scan_text = crisis_scan_text.replace(phrase, "")

        for kw in self.crisis_keywords:
            if re.search(r'\b' + re.escape(kw) + r'\b', crisis_scan_text):
                results['crisis_indicators'].append(kw)

        # Score each emotion by keyword matches
        scores = {emotion: 0 for emotion in self.emotion_keywords}
        for emotion, keywords in self.emotion_keywords.items():
            for kw in keywords:
                if kw in text_lower:
                    scores[emotion] += 1

        best_emotion = max(scores, key=scores.get)
        if scores[best_emotion] > 0:
            results['primary_emotion'] = best_emotion
        elif vader['compound'] >= 0.5:
            results['primary_emotion'] = 'happy'
        elif vader['compound'] <= -0.5:
            results['primary_emotion'] = 'depressed'
        elif vader['compound'] <= -0.2:
            results['primary_emotion'] = 'stressed'

        if nlp:
            try:
                doc = nlp(text)
                results['key_phrases'] = [
                    chunk.text for chunk in doc.noun_chunks
                    if len(chunk.text.split()) > 1
                ][:3]
            except Exception:
                pass

        return results

    def _pick(self, pool, exclude):
        import random
        exclude_set = set(exclude or [])
        candidates = [r for r in pool if r not in exclude_set]
        return random.choice(candidates if candidates else pool)

    def generate_therapeutic_response(self, analysis, user_message, recent_history=None, exclude=None):
        if analysis['crisis_indicators']:
            return self._pick(self.crisis_responses, exclude)

        emotion = analysis['primary_emotion']
        sentiment = analysis['sentiment']

        history_text = ""
        if recent_history:
            history_text = "Recent conversation:\n" + "\n".join(
                f"{'User' if m.sender == 'user' else 'Bot'}: {m.content}" for m in recent_history
            ) + "\n\n"

        prompt = (
            "You are a warm, supportive mental health companion chatbot. You are not a "
            "licensed therapist and must never claim to diagnose or treat any condition. "
            "Respond to the user's message with empathy: validate what they're feeling, "
            "reflect it back briefly, and offer one gentle, grounded thought or question "
            "that helps them keep talking. Keep it to 2-4 sentences, conversational, no "
            "bullet points, no em dashes, no clinical jargon. If appropriate, you may gently suggest "
            "professional support, but don't do this every time.\n\n"
            f"{history_text}"
            f"Detected emotion: {emotion} (sentiment score: {sentiment:.2f})\n"
            f"User's message: \"{user_message}\"\n\n"
            "Your response:"
        )

        try:
            result = self.gemini_client.models.generate_content(
                model=self.gemini_model,
                contents=prompt,
            )
            text = (result.text or "").strip()
            if text:
                return text
        except Exception as e:
            print(f"Gemini API error: {e}")

        return self._pick(self.fallback_responses, exclude)

# Initialize bot
bot = AITherapyBot()

# Routes
@app.route('/')
def home():
    if 'session_id' not in session:
        session['session_id'] = secrets.token_hex(16)
        session.permanent = True

        conversation = Conversation.query.filter_by(session_id=session['session_id']).first()
        if not conversation:
            conversation = Conversation(session_id=session['session_id'])
            db.session.add(conversation)
            db.session.commit()

    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message', '').strip()

    if not user_message:
        return jsonify({'error': 'Please enter a message'}), 400

    session_id = session.get('session_id')
    if not session_id:
        return jsonify({'error': 'Session not found'}), 400

    conversation = Conversation.query.filter_by(session_id=session_id).first()
    if not conversation:
        conversation = Conversation(session_id=session_id)
        db.session.add(conversation)
        db.session.commit()

    # Analyze user message
    analysis = bot.analyze_emotional_state(user_message)

    # NEW: avoid repeating the most recent bot replies (last 5)
    last_bot_msgs = (
        Message.query
        .filter_by(conversation_id=conversation.id, sender='bot')
        .order_by(Message.timestamp.desc())
        .limit(5)
        .all()
    )
    exclude = [m.content for m in last_bot_msgs]

    recent_history = (
        Message.query
        .filter_by(conversation_id=conversation.id)
        .order_by(Message.timestamp.desc())
        .limit(6)
        .all()
    )[::-1]

    # Generate therapeutic response
    bot_response = bot.generate_therapeutic_response(
        analysis, user_message, recent_history=recent_history, exclude=exclude
    )

    # Store user message
    user_msg = Message(
        conversation_id=conversation.id,
        sender='user',
        content=user_message,
        emotion=analysis['primary_emotion'],
        sentiment_score=analysis['sentiment'],
        key_phrases=json.dumps(analysis['key_phrases'])
    )
    db.session.add(user_msg)

    # Store bot response
    bot_msg = Message(
        conversation_id=conversation.id,
        sender='bot',
        content=bot_response,
        therapy_technique=analysis['primary_emotion']
    )
    db.session.add(bot_msg)

    db.session.commit()

    return jsonify({
        'response': bot_response,
        'timestamp': bot_msg.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
        'analysis': {
            'emotion': analysis['primary_emotion'],
            'sentiment': analysis['sentiment'],
            'confidence': 0.75,
            'key_phrases': analysis['key_phrases'][:3],
            'therapy_technique': analysis['primary_emotion'],
            'crisis_detected': len(analysis['crisis_indicators']) > 0
        }
    })

@app.route('/conversation')
def get_conversation():
    session_id = session.get('session_id')
    if not session_id:
        return jsonify([])

    conversation = Conversation.query.filter_by(session_id=session_id).first()
    if not conversation:
        return jsonify([])

    messages = Message.query.filter_by(conversation_id=conversation.id).order_by(Message.timestamp).all()

    return jsonify([{
        'sender': msg.sender,
        'content': msg.content,
        'timestamp': msg.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
        'emotion': msg.emotion,
        'sentiment_score': msg.sentiment_score,
        'therapy_technique': msg.therapy_technique
    } for msg in messages])

@app.route('/new-conversation', methods=['POST'])
def new_conversation():
    session['session_id'] = secrets.token_hex(16)
    session.permanent = True

    conversation = Conversation(session_id=session['session_id'])
    db.session.add(conversation)
    db.session.commit()

    return jsonify({'status': 'ok'})

# Initialize database
with app.app_context():
    db.create_all()

if __name__ == '__main__':
    debug_mode = os.environ.get('FLASK_DEBUG', '0') == '1'
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=debug_mode, host='0.0.0.0', port=port)

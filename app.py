from flask import Flask, render_template, redirect, url_for, flash, request, session, jsonify
import os
import torch
from transformers import MarianMTModel, MarianTokenizer, BertTokenizer, BertModel
import speech_recognition as sr
from moviepy.editor import AudioFileClip
from googletrans import Translator  # Using googletrans for fallback
from gtts import gTTS
import uuid
import time, random
from flask_bcrypt import Bcrypt
from flask_sqlalchemy import SQLAlchemy


app = Flask(__name__)
app.config["SECRET_KEY"] = "ABC"
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['UPLOAD_FOLDER'] = 'static/saved_audios'


db = SQLAlchemy(app)
bcrypt = Bcrypt(app)


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.String(1), nullable=False)
    mobile = db.Column(db.String(15), nullable=False)

class Quizes(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    language = db.Column(db.String(1000), nullable=False)
    question = db.Column(db.String(1000), nullable=False)

class Options(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    opt = db.Column(db.String(500), nullable=False)
    answer = db.Column(db.Boolean, nullable=False)
    question_id = db.Column(db.Integer, nullable=False)

# ======= ContextAwareTranslator Class with Retry and Caching =======
from deep_translator import GoogleTranslator  # Ensure this is at the top

class ContextAwareTranslator:
    def __init__(self, source_lang='en', target_lang='hi'):
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.translation_cache = {}

        # Initialize MarianMT only for Hindi
        if target_lang == 'hi':
            self.model_name = f'Helsinki-NLP/opus-mt-{source_lang}-{target_lang}'
            try:
                self.model = MarianMTModel.from_pretrained(self.model_name)
                self.tokenizer = MarianTokenizer.from_pretrained(self.model_name)
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.model.to(self.device)
                print(f"MarianMT model loaded for {source_lang}-{target_lang}")
            except Exception as e:
                print(f"Error loading MarianMT model for Hindi: {e}")
                self.model = None
                self.tokenizer = None
        else:
            self.model = None
            self.tokenizer = None

    def translate(self, text, retries=3, delay=1):
        # Check if translation is already in cache
        cache_key = (text, self.source_lang, self.target_lang)
        if cache_key in self.translation_cache:
            return self.translation_cache[cache_key]

        # Use MarianMT for Hindi
        if self.target_lang == 'hi' and self.model and self.tokenizer:
            try:
                tokens = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True).to(self.device)
                translated_tokens = self.model.generate(**tokens)
                translation = self.tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
                self.translation_cache[cache_key] = translation
                return translation
            except Exception as e:
                print(f"Error during translation with MarianMT for Hindi: {e}")

        # Use deep-translator for other languages
        for attempt in range(retries):
            try:
                print(f"Attempting translation using GoogleTranslator for {self.target_lang}")
                translation = GoogleTranslator(source=self.source_lang, target=self.target_lang).translate(text)
                self.translation_cache[cache_key] = translation
                return translation
            except Exception as e:
                print(f"deep-translator attempt {attempt + 1} failed for {self.target_lang}: {e}")
                time.sleep(delay)

        print(f"Failed to translate text after {retries} attempts for {self.target_lang}.")
        return "Translation failed"

    
# Define supported languages for gTTS
SUPPORTED_LANGUAGES = {'en', 'hi', 'ml', 'bn', 'ta', 'te', 'kn', 'mr'}  # Add other supported languages as needed

def text_to_speech(text, lang='en', retries=3):
    # Check if the language is supported by gTTS, if not, default to 'en'
    if lang not in SUPPORTED_LANGUAGES:
        print(f"Language '{lang}' not supported by gTTS. Defaulting to English.")
        lang = 'en'
    
    # Generate a unique filename for each audio file
    unique_filename = f'translated_audio_{uuid.uuid4().hex}.mp3'
    audio_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)

    # Retry mechanism
    for attempt in range(retries):
        try:
            tts = gTTS(text=text, lang=lang, slow=False)
            tts.save(audio_path)
            return audio_path
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(1)  # Wait a bit before retrying
    
    # If all attempts fail, log the error and return None
    print(f"Failed to convert text to audio after {retries} attempts.")
    return None

# Cleanup function to remove old audio files
def cleanup_old_files(folder, age_threshold_seconds=86400):  # Default is 24 hours
    current_time = time.time()
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isfile(file_path):
            file_age = current_time - os.path.getmtime(file_path)
            if file_age > age_threshold_seconds:
                os.remove(file_path)
                print(f"Deleted old file: {file_path}")


@app.route('/translation_text', methods=['GET', 'POST'])
def translation_text():
    if request.method == 'POST':
        original_lang = request.form['originalLanguage']
        translated_lang = request.form['translatedLanguage']
        user_text = request.form['userText']

        # Use the updated ContextAwareTranslator
        translator = ContextAwareTranslator(source_lang=original_lang, target_lang=translated_lang)

        translated_text = translator.translate(user_text)
        print(f"Translated text: {translated_text}")

        audio_path = text_to_speech(translated_text, translated_lang)
        if not audio_path:
            flash("Failed to convert text to audio. Please try again later.", "danger")

        cleanup_old_files(app.config['UPLOAD_FOLDER'])  # Clean old files

        return render_template('translation1.html', translated_text=translated_text, audio_file=audio_path)
    return render_template('translation1.html')

@app.route('/translation_audio', methods=['GET', 'POST'])
def translation_audio():
    return render_template('translation2.html')

@app.route('/translate', methods=['POST'])
def translate():
    try:
        data = request.get_json()
        from_lang = data.get('from_lang')
        to_lang = data.get('to_lang')

        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
            audio = recognizer.listen(source, timeout=5)

        # Language code mapping for speech recognition and gTTS
        LANG_CODE_MAPPING_SR = {
            'en': 'en-US',
            'hi': 'hi-IN',
            'ml': 'ml-IN',
            'bn': 'bn-IN',
            'ta': 'ta-IN',
            'te': 'te-IN',
            'kn': 'kn-IN',
            'mr': 'mr-IN',
        }
        LANG_CODE_MAPPING_TTS = {
            'en': 'en',
            'hi': 'hi',
            'ml': 'ml',
            'bn': 'bn',
            'ta': 'ta',
            'te': 'te',
            'kn': 'kn',
            'mr': 'mr',
        }

        recognized_text = recognizer.recognize_google(audio, language=LANG_CODE_MAPPING_SR.get(from_lang, 'en-US'))
        print(99999999999999, recognized_text)
        translator = ContextAwareTranslator(source_lang=from_lang, target_lang=to_lang)
        translated_text = translator.translate(recognized_text)
        audio_path = text_to_speech(translated_text, LANG_CODE_MAPPING_TTS.get(to_lang, 'en'))

        if not audio_path:
            return jsonify({'error': 'Failed to convert translated text to audio.'}), 500

        cleanup_old_files(app.config['UPLOAD_FOLDER'])  # Clean old files

        response = {
            'original_text': recognized_text,
            'translated_text': translated_text,
            'audio_file': audio_path
        }
        return jsonify(response)

    except sr.UnknownValueError:
        return jsonify({'error': 'Could not understand audio, please try again.'}), 400

    except sr.RequestError as e:
        return jsonify({'error': f'Speech Recognition service error: {e}. Check your internet connection.'}), 500

    except Exception as e:
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500
    






@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        if email == "admin@gmail.com" and password == "admin":
            return redirect('/admin_home')

        user = User.query.filter_by(email=email).first()
        if user and bcrypt.check_password_hash(user.password, password):
            session['user_id'] = user.id
            return redirect('/home')
        else:
            flash('Invalid email or password.', 'danger')
    return render_template('blog.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        age = request.form.get('age')
        gender = request.form.get('gender')
        mobile = request.form.get('mobile')

        if len(mobile) != 10 or not mobile.isdigit():
            flash('Mobile number must be exactly 10 digits.', 'danger')
            return render_template('blog.html')

        if User.query.filter_by(email=email).first():
            flash('Email address already in use. Please choose a different one.', 'danger')
            return render_template('blog.html')

        if User.query.filter_by(username=username).first():
            flash('Username is already taken. Please choose a different one.', 'danger')
            return render_template('blog.html')

        if password != confirm_password:
            flash('Passwords do not match.', 'danger')
            return render_template('blog.html')

        if len(password) < 8:
            flash('Password must be at least 8 characters long.', 'danger')
            return render_template('blog.html')

        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        new_user = User(username=username, email=email, password=hashed_password, age=age, gender=gender, mobile=mobile)

        db.session.add(new_user)
        db.session.commit()

        flash('Registration successful! You can now log in.', 'success')
        return redirect(url_for('login'))
    return render_template('blog.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/about_1')
def about_1():
    return render_template('about_1.html')



@app.route('/quiz/<lang>', methods=["GET", "POST"])
def quiz(lang):
    from translate import Translator
    # Initialize the translator with the correct target language
    translator = Translator(from_lang="en", to_lang=lang)

    if request.method == "POST":
        total_questions = 0
        correct_answers = 0
        feedback_list = []

        for key, value in request.form.items():
            if key.startswith("answer_"):  # Keys that start with 'answer_' are the submitted answers
                quiz_id = key.split('_')[1]
                opt_id = value  # opt_id is the selected option's ID

                # Get the correct answer from the Options table
                correct_option = Options.query.filter_by(question_id=quiz_id, answer=True).first()
                user_answer = Options.query.filter_by(id=opt_id).first()  # Get the user's selected answer

                if correct_option and user_answer:
                    is_correct = (int(opt_id) == correct_option.id)  # Compare the option IDs
                    feedback_list.append({
                        'question_number': total_questions + 1,
                        'question': Quizes.query.get(quiz_id).question,
                        'user_answer': translator.translate(user_answer.opt),  # Access the user's selected option text
                        'correct_answer': translator.translate(correct_option.opt),  # Access the correct option text
                        'is_correct': is_correct
                    })
                    if is_correct:
                        correct_answers += 1
                
                total_questions += 1

        # Calculate the percentage of correct answers
        if total_questions > 0:
            score_percentage = (correct_answers / total_questions) * 100
        else:
            score_percentage = 0

        return render_template('quiz.html', quizzes_list=[], lang=lang, feedback_list=feedback_list, score_percentage=score_percentage)
    

    # Handling GET request - Displaying the quizzes
    quizzes = Quizes.query.filter_by(language=lang).all()
    quizzes_list = []

    for quiz in quizzes:
        # Fetch options for the current quiz
        options = Options.query.filter_by(question_id=quiz.id).all()

        # Translate each option and shuffle them
        translated_options = []
        for option in options:
            translated_option = translator.translate(option.opt)  # The result is already a string
            translated_options.append({
                'name': f'answer_{quiz.id}',
                'value': option.id,  # Use the option ID as the value
                'label': translated_option  # Store the translated option text as the label
            })

        random.shuffle(translated_options)

        quizzes_list.append({
            'id': quiz.id,
            'language': quiz.language,
            'question': quiz.question,
            'options': translated_options
        })

    return render_template('quiz.html', quizzes_list=quizzes_list, lang=lang)

@app.route('/admin_home')
def admin_home():
    return render_template('admin_home.html')


@app.route('/add_quiz', methods=["GET", "POST"])
def add_quiz():
    if request.method == "POST":
        language = request.form['language']
        question = request.form['question']
        answer = request.form['answer']
        opt1 = request.form['opt1']
        opt2 = request.form['opt2']
        opt3 = request.form['opt3']

        # Create and add the new quiz to the database
        new_quiz = Quizes(
            language=language,
            question=question
        )
        db.session.add(new_quiz)
        db.session.commit()  # Commit to assign an ID to the quiz

        # Now, retrieve the ID of the newly created quiz
        latest_quiz_id = new_quiz.id

        # Create and add the correct answer option
        correct_option = Options(
            opt=answer,
            answer=True,
            question_id=latest_quiz_id
        )
        db.session.add(correct_option)

        # Create and add the incorrect options
        option1 = Options(
            opt=opt1,
            answer=False,
            question_id=latest_quiz_id
        )
        db.session.add(option1)

        option2 = Options(
            opt=opt2,
            answer=False,
            question_id=latest_quiz_id
        )
        db.session.add(option2)

        option3 = Options(
            opt=opt3,
            answer=False,
            question_id=latest_quiz_id
        )
        db.session.add(option3)

        # Commit all options to the database at once
        db.session.commit()

        flash('Quiz added successfully!', 'success')
        return render_template('add_quiz.html', message="Quiz added successfully!")
    
    return render_template('add_quiz.html')



@app.route('/quiz_list/<lang>', methods=["GET", "POST"])
def quiz_list(lang):
    if request.method == "POST":
        # Get the quiz ID and updated data from the form
        quiz_id = request.form['id']
        question = request.form['question']
        language = request.form['language']
        answer_id = request.form['answer_id']
        
        # Get the answer text
        answer_text = request.form['answer']

        # Update the quiz question and language
        quiz = Quizes.query.get(quiz_id)
        if quiz:
            quiz.language = language
            quiz.question = question
            db.session.commit()

            # Update the answer option
            answer_option = Options.query.get(answer_id)
            if answer_option:
                answer_option.opt = answer_text
                db.session.commit()

            # Update each option (but not the answer)
            for i in range(1, 5):
                option_id = request.form.get(f'option_id_{i}')
                option_text = request.form.get(f'option_text_{i}')
                
                if option_id and option_text:
                    option = Options.query.get(option_id)
                    if option:
                        option.opt = option_text
                        # Check if this is the correct answer
                        option.answer = (option.id == int(answer_id))  # Mark as answer if matched
                        db.session.commit()

            flash("Quiz updated successfully!", 'success')
        else:
            flash("Quiz not found!", 'danger')

        return redirect(url_for('quiz_list', lang=lang))

    # For GET request: render the page with quiz data
    quizzes = Quizes.query.filter_by(language=lang).all()

    quizzes_list = []
    for quiz in quizzes:
        options = Options.query.filter_by(question_id=quiz.id).all()

        quiz_data = {
            'id': quiz.id,
            'language': quiz.language,
            'question': quiz.question,
            'options': []  # Use a list to store options
        }

        for opt in options:
            quiz_data['options'].append({
                'id': opt.id,
                'option': opt.opt,
                'is_answer': opt.answer
            })

        quizzes_list.append(quiz_data)

    return render_template('quiz_list.html', quizzes_list=quizzes_list, lang=lang)



@app.route('/edit_quiz/<lang>', methods=["POST"])
def edit_quiz(lang):
    # Retrieve the quiz ID and the updated fields from the form
    quiz_id = request.form['id']
    language = request.form['language']
    question = request.form['question']
    answer_text = request.form['answer']
    answer_id = int(request.form['answer_id'])

    # Collect options from the form
    options = []
    for i in range(1, 4):  # Adjust range based on the number of options expected
        option_id = request.form.get(f'option_id_{i}')
        option_text = request.form.get(f'option_text_{i}')
        
        if option_id and option_text:
            options.append({
                'option_id': int(option_id),
                'option_text': option_text,
            })

    # Update the quiz record in the database
    quiz = Quizes.query.get(quiz_id)
    if quiz:
        quiz.language = language
        quiz.question = question
        db.session.commit()

        # Update the options
        for opt in options:
            option = Options.query.get(opt['option_id'])
            if option:
                option.opt = opt['option_text']
                option.answer = (option.id == answer_id)  # Mark this option as the correct answer if it matches answer_id
                db.session.commit()

        flash("Quiz updated successfully!", 'success')
    else:
        flash("Quiz not found!", 'danger')

    return redirect(url_for('quiz_list', lang=lang))



@app.route('/remove_quiz/<id>/<lang>')
def remove_quiz(id, lang):
    # Find the quiz by its ID
    quiz = Quizes.query.filter_by(id=id).first()
    if quiz:
        # Delete the quiz
        db.session.delete(quiz)
        db.session.commit()

        # Find all options related to the quiz and delete them
        options = Options.query.filter_by(question_id=id).all()
        for option in options:
            db.session.delete(option)
        db.session.commit()

        flash("Quiz and its options deleted successfully!", 'success')
    else:
        flash("Quiz not found!", 'danger')
    
    return redirect(url_for('quiz_list', lang=lang))




if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

        # Create the database tables
    with app.app_context():
        db.create_all()

    app.run(debug=True)

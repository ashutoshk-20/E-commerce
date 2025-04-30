from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from pymongo import MongoClient
from flask_paginate import Pagination, get_page_args
from flask_cors import CORS
from bson.objectid import ObjectId
from openai import OpenAI
import threading
import pyotp
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = "your_secret_key"  

MONGO_URI = "mongodb://localhost:27017/" 
OPENROUTER_API_KEY = "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"  
OPENROUTER_MODEL = "deepseek/deepseek-r1-zero:free"  

mongo_client = MongoClient(MONGO_URI)
db = mongo_client["product_data"]
collection = db["products"]
users_collection = db["users"]

openai_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)
openrouter_model = OPENROUTER_MODEL

CORS(app)
shutdown_flag = threading.Event()

@app.route('/')
def index():
    query = request.args.get('q', '').strip('+')
    page, per_page, offset = get_page_args(page_parameter='page', per_page_parameter='per_page', per_page=6)

    search_filter = {}
    if query:
        search_filter = {
            "$or": [
                {"Name": {"$regex": query, "$options": "i"}},
                {"Brand": {"$regex": query, "$options": "i"}}
            ]
        }

    total = collection.count_documents(search_filter)
    products = list(collection.find(search_filter).skip(offset).limit(per_page))
    pagination = Pagination(
        page=page,
        per_page=per_page,
        total=total,
        css_framework='bootstrap4',
        outer_class='inline-flex space-x-2 items-center',
        prev_label='«',
        next_label='»',
        format_total=True,
        format_number=True,
        page_parameter='page',
        link_class='px-3 py-2 bg-blue-500 text-white rounded-full hover:bg-blue-600 transition-colors duration-200',
        active_class='bg-blue-600 text-white',
        disabled_class='opacity-50 cursor-not-allowed',
        dotdot_label='...'
    )

    user_email = session.get('user_email', None)
    return render_template('products.html', products=products, pagination=pagination, query=query, user_email=user_email)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        if 'step' not in session or session.get('step') == 'initial':
            try:
                email = request.form['email']
                password = request.form['password']
            except KeyError as e:
                logger.error(f"Missing form field in login: {e}")
                return render_template('login.html', error=f"Missing field: {e}")
            
            user = users_collection.find_one({"email": email, "password": password})
            if user:
                totp = pyotp.TOTP(pyotp.random_base32(), interval=300)
                otp = totp.now()
                session['otp'] = otp
                session['email'] = email
                session['step'] = 'otp'
                logger.debug(f"Login OTP for {email}: {otp}")
                return render_template('login.html', step='otp', email=email, otp=otp)
            return render_template('login.html', error="Invalid email or password")

        elif session.get('step') == 'otp':
            try:
                user_otp = request.form['otp']
            except KeyError:
                logger.error("Missing OTP field in login OTP step")
                return render_template('login.html', step='otp', error="Missing OTP field", email=session.get('email'), otp=session.get('otp'))
            
            stored_otp = session.get('otp')
            if stored_otp and user_otp == stored_otp:
                session['user_email'] = session.get('email')
                session.pop('otp', None)
                session.pop('email', None)
                session.pop('step', None)
                return redirect(url_for('index'))
            return render_template('login.html', step='otp', error="Invalid OTP", email=session.get('email'), otp=session.get('otp'))

    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        try:
            email = request.form['email']
            password = request.form['password']
        except KeyError as e:
            logger.error(f"Missing form field in register: {e}")
            return render_template('register.html', error=f"Missing field: {e}")

        if users_collection.find_one({"email": email}):
            return render_template('register.html', error="Email already registered")

        totp = pyotp.TOTP(pyotp.random_base32(), interval=300)
        otp = totp.now()
        session['otp'] = otp
        session['email'] = email
        session['password'] = password

        logger.debug(f"Registration OTP for {email}: {otp}")
        return render_template('verify_otp.html', email=email, otp=otp)

    return render_template('register.html')

@app.route('/verify_otp', methods=['GET', 'POST'])
def verify_otp():
    if request.method == 'POST':
        try:
            user_otp = request.form['otp']
        except KeyError:
            logger.error("Missing OTP field in verify_otp")
            return render_template('verify_otp.html', error="Missing OTP field", email=session.get('email'), otp=session.get('otp'))

        stored_otp = session.get('otp')
        email = session.get('email')
        password = session.get('password')

        if stored_otp and user_otp == stored_otp:
            users_collection.insert_one({"email": email, "password": password})
            session['user_email'] = email
            session.pop('otp', None)
            session.pop('email', None)
            session.pop('password', None)
            return redirect(url_for('index'))
        return render_template('verify_otp.html', error="Invalid OTP", email=session.get('email'), otp=session.get('otp'))

    return render_template('verify_otp.html')

@app.route('/logout')
def logout():
    session.pop('user_email', None)
    return redirect(url_for('index'))

@app.route('/product/<product_id>')
def product_detail(product_id):
    try:
        product = collection.find_one({"_id": ObjectId(product_id)})
        if not product:
            return "Product not found", 404

        recommendations = collection.find({
            "$or": [
                {"Brand": product.get("Brand", "")},
                {"Category": product.get("Category", "")}
            ],
            "_id": {"$ne": product["_id"]}
        }).limit(4)

        user_email = session.get('user_email', None)
        return render_template('product_detail.html', product=product, recommendations=list(recommendations), user_email=user_email)
    except Exception as e:
        return f"Invalid product ID. Error: {e}", 400

@app.route('/suggest')
def suggest():
    query = request.args.get('q', '').strip('+')
    if not query:
        return jsonify([])

    results = collection.find(
        {"Name": {"$regex": f'^{query}', "$options": "i"}},
        {"Name": 1, "_id": 0}
    ).limit(10)

    suggestions = [item['Name'] for item in results if 'Name' in item]
    return jsonify(suggestions)

@app.route('/chat', methods=['POST'])
def chat():
    if 'user_email' not in session:
        return jsonify({'error': 'Please log in first'}), 401
    user_input = request.json.get("message")

    if not user_input:
        return jsonify({'error': 'No message received'}), 400

    try:
        response = openai_client.chat.completions.create(
            model=openrouter_model,
            messages=[
                {"role": "user", "content": user_input}
            ],
            extra_headers={
                "HTTP-Referer": "http://localhost:5001/",
                "X-Title": "ProductBot"
            }
        )

        message = response.choices[0].message.content
        return jsonify({'response': message})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/chat')
def chat_ui():
    if 'user_email' not in session:
        return redirect(url_for('login'))
    user_email = session.get('user_email', None)
    return render_template('chat.html', user_email=user_email)

@app.route('/api/check-session')
def check_session():
    if 'user_email' in session:
        return jsonify({"loggedIn": True, "email": session['user_email']})
    return jsonify({"loggedIn": False})

@app.route('/shutdown', methods=['POST'])
def shutdown():
    shutdown_flag.set()
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()
    return jsonify({"message": "Server is shutting down..."})

def run_server():
    app.run(debug=True, port=5001, use_reloader=False)

if __name__ == '__main__':
    server_thread = threading.Thread(target=run_server)
    server_thread.start()
    try:
        while not shutdown_flag.is_set():
            shutdown_flag.wait(timeout=1)
    except KeyboardInterrupt:
        shutdown_flag.set()
    finally:
        server_thread.join()
    print("Server stopped gracefully.")
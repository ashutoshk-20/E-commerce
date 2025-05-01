from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from pymongo import MongoClient
from flask_paginate import Pagination, get_page_args
from flask_cors import CORS
from bson.objectid import ObjectId
from openai import OpenAI
import threading
import pyotp
import logging
import json
import re,os
from dotenv import load_dotenv

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "a_fallback_secret_key_if_env_missing")

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-r1-zero:free")

try:
    mongo_client = MongoClient(MONGO_URI)
    mongo_client.admin.command('ping')
    db = mongo_client["product_data"]
    collection = db["tech_data"] 
    users_collection = db["users"]
    logger.info("MongoDB connection successful.")
except Exception as e:
    logger.error(f"Failed to connect to MongoDB: {e}")
    mongo_client = None 

if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "your_valid_openrouter_api_key_here":
     logger.error("OPENROUTER_API_KEY is not set or is the default placeholder. Chat functionality will fail.")
     openai_client = None
else:
    openai_client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY
    )
    logger.info("OpenAI client (via OpenRouter) initialized.")

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
                {"titles": {"$regex": query, "$options": "i"}},
                {"Brand": {"$regex": query, "$options": "i"}}
            ]
        }

    try:
        total = collection.count_documents(search_filter)
        products = list(collection.find(search_filter).skip(offset).limit(per_page))
    except Exception as e:
         logger.error(f"Error accessing MongoDB for index page: {e}")
         total = 0
         products = []

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

            try:
                user = users_collection.find_one({"email": email, "password": password})
            except Exception as e:
                 logger.error(f"Error accessing MongoDB during login: {e}")
                 return render_template('login.html', error="An error occurred during login.")

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
                logger.info(f"User {session['user_email']} logged in successfully.")
                return redirect(url_for('index'))
            return render_template('login.html', step='otp', error="Invalid OTP", email=session.get('email'), otp=session.get('otp'))

    session['step'] = 'initial'
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

        try:
            if users_collection.find_one({"email": email}):
                return render_template('register.html', error="Email already registered")
        except Exception as e:
             logger.error(f"Error accessing MongoDB during registration check: {e}")
             return render_template('register.html', error="An error occurred during registration.")

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
    if 'email' not in session or 'password' not in session or 'otp' not in session:
        logger.warning("Attempted to access verify_otp without necessary session data.")
        return redirect(url_for('register'))

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
            try:
                users_collection.insert_one({"email": email, "password": password})
                logger.info(f"New user registered: {email}")
                session['user_email'] = email
                session.pop('otp', None)
                session.pop('email', None)
                session.pop('password', None)
                return redirect(url_for('index'))
            except Exception as e:
                 logger.error(f"Error saving user to MongoDB after OTP verification: {e}")
                 return render_template('verify_otp.html', error="An error occurred while finalizing registration.")

        return render_template('verify_otp.html', email=session.get('email'), otp=session.get('otp'))

    return render_template('verify_otp.html', email=session.get('email'), otp=session.get('otp'))

@app.route('/logout')
def logout():
    email = session.get('user_email')
    if email:
        logger.info(f"User {email} logged out.")
    session.pop('user_email', None)
    session.pop('otp', None)
    session.pop('email', None)
    session.pop('password', None)
    session.pop('step', None)
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
        logger.error(f"Error fetching product detail for ID {product_id}: {e}")
        if "is not a valid ObjectId" in str(e):
             return "Invalid product ID format.", 400
        return f"An error occurred while loading the product details. Error: {e}", 500

@app.route('/suggest')
def suggest():
    query = request.args.get('q', '').strip('+')
    if not query:
        return jsonify([])

    try:
        results = collection.find(
            {"titles": {"$regex": f'^{re.escape(query)}', "$options": "i"}},
            {"titles": 1, "_id": 0}
        ).limit(10)

        suggestions = [item.get('titles') for item in results if item.get('titles')]
        return jsonify(suggestions)
    except Exception as e:
        logger.error(f"Error fetching suggestions for query '{query}': {e}")
        return jsonify([]), 500


@app.route('/chat', methods=['POST'])
def chat():
    if openai_client is None:
         return jsonify({'error': 'Chat service is not available due to configuration errors.'}), 503

    if 'user_email' not in session:
        return jsonify({'error': 'Please log in first'}), 401

    user_input = request.json.get("message")

    if not user_input:
        return jsonify({'error': 'No message received'}), 400

    messages = [
        {"role": "system", "content": "You are a helpful product support assistant. Use the provided product information to answer the user's question. Respond concisely and use plain text or standard markdown. Do not use LaTeX formatting like \\boxed{}. If you cannot find relevant product information provided, state that you don't have details for that specific item or that it's outside your current knowledge."}
    ]

    product_data_for_ai = None
    product_keywords = ["product", "item", "about", "details", "info", "title", "name"]
    if any(keyword in user_input.lower() for keyword in product_keywords):
        logger.debug(f"Potential product query detected: {user_input}")
        name_match = re.search(r'\b([A-Z][a-zA-Z0-9\s]+)\b', user_input)
        product_title_guess = name_match.group(1).strip() if name_match else None

        if not product_title_guess:
             quoted_match = re.search(r'"(.*?)"|\'(.*?)\'', user_input)
             if quoted_match:
                 product_title_guess = quoted_match.group(1) or quoted_match.group(2)
                 if product_title_guess:
                      product_title_guess = product_title_guess.strip()

        if product_title_guess:
            logger.debug(f"Attempting to find product with title guess: '{product_title_guess}'")
            try:
                if ' ' in product_title_guess:
                     product = collection.find_one({"titles": {"$regex": r"\b" + re.escape(product_title_guess) + r"\b", "$options": "i"}})
                else:
                    product = collection.find_one({"titles": {"$regex": f"^{re.escape(product_title_guess)}", "$options": "i"}})

                if product:
                    logger.debug(f"Product found: {product.get('titles')}")
                    product_dict = {k: (str(v) if isinstance(v, ObjectId) else v) for k, v in product.items()}
                    product_data_for_ai = json.dumps(product_dict)

                    messages.append({"role": "user", "content": f"Database Context (Product Info): {product_data_for_ai}"})
                else:
                    logger.debug(f"Product not found for title guess: '{product_title_guess}'")
                    messages.append({"role": "user", "content": f"Database Context: No specific product found matching '{product_title_guess}'."})

            except Exception as e:
                logger.error(f"Error accessing MongoDB for product lookup: {e}")
                messages.append({"role": "user", "content": f"Database Context: An error occurred while looking up product information: {e}"})
        else:
             logger.debug("No potential product title extracted from user input.")
             messages.append({"role": "user", "content": "Database Context: Could not identify a specific product title to look up."})

    messages.append({"role": "user", "content": f"User's original question: {user_input}"})

    try:
        logger.debug(f"Sending messages to AI: {messages}")
        response = openai_client.chat.completions.create(
            model=openrouter_model,
            messages=messages,
            extra_headers={
                "HTTP-Referer": "http://localhost:5001/",
                "X-Title": "ProductBot"
            }
        )

        message = response.choices[0].message.content
        logger.debug(f"Received AI response: {message}")
        return jsonify({'response': message})

    except Exception as e:
        logger.error(f"Error during OpenAI chat completion: {e}", exc_info=True)
        return jsonify({'error': 'An error occurred while processing your request. Please try again later.'}), 500


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
    if request.remote_addr != '127.0.0.1':
        return jsonify({"message": "Remote shutdown not allowed."}), 403

    logger.info("Shutdown requested...")
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        logger.error("Not running with the Werkzeug Server, shutdown not possible this way.")
        return jsonify({"message": "Server not running with Werkzeug server."}), 500
    try:
        func()
        logger.info("Werkzeug server shutdown function called.")
        return jsonify({"message": "Server is shutting down..."})
    except Exception as e:
        logger.error(f"Error during Werkzeug server shutdown: {e}")
        return jsonify({"message": f"Error during shutdown: {e}"}), 500


def run_server():
    app.run(debug=True, port=5001, use_reloader=False)

if __name__ == '__main__':
    logger.info("Starting Flask server in a separate thread.")
    server_thread = threading.Thread(target=run_server)
    server_thread.daemon = True 
    server_thread.start()
    logger.info("Server thread started. Press Ctrl+C to stop.")

    try:
        while not shutdown_flag.is_set():
            shutdown_flag.wait(timeout=1)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Setting shutdown flag.")
        shutdown_flag.set() 
    finally:
        logger.info("Main thread waiting for server thread to join.")
        if server_thread.is_alive():
             server_thread.join(timeout=5)
             if server_thread.is_alive():
                 logger.warning("Server thread did not terminate gracefully.")
             else:
                 logger.info("Server thread terminated gracefully.")
from flask import Flask, render_template, request, jsonify
from pymongo import MongoClient
from flask_paginate import Pagination, get_page_args
from bson.objectid import ObjectId
from openai import OpenAI
from dotenv import load_dotenv
import os
import threading

load_dotenv()
app = Flask(__name__)

mongo_client = MongoClient(os.getenv("MONGO_URI"))
db = mongo_client["product_data"]
collection = db["products"]

openai_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)
openrouter_model = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-r1-zero:free")

# Flag for graceful shutdown
shutdown_flag = threading.Event()

@app.route('/')
def index():
    query = request.args.get('q', '')
    # Decode and clean the query to remove excessive padding
    query = query.strip('+') if query else ''
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

    return render_template('products.html', products=products, pagination=pagination, query=query)

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

        return render_template('product_detail.html', product=product, recommendations=list(recommendations))
    except Exception as e:
        return f"Invalid product ID. Error: {e}", 400

@app.route('/suggest')
def suggest():
    query = request.args.get('q', '')
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
    return render_template('chat.html')

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
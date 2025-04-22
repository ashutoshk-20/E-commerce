from flask import Flask, render_template, request, jsonify
from pymongo import MongoClient
from flask_paginate import Pagination, get_page_args
from bson.objectid import ObjectId

app = Flask(__name__)

client = MongoClient("mongodb://localhost:27017/")
db = client["product_data"]
collection = db["products"]

@app.route('/')
def index():
    query = request.args.get('q', '')
    page, per_page, offset = get_page_args(page_parameter='page', per_page_parameter='per_page')

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

    pagination = Pagination(page=page, per_page=per_page, total=total, css_framework='bootstrap4')
    return render_template('products.html', products=products, pagination=pagination)

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
        {"Name": 1}
    ).limit(10)

    suggestions = [item['Name'] for item in results if 'Name' in item]
    return jsonify(suggestions)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, threaded=False, port=5001)

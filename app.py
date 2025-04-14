from flask import Flask, render_template, request
from pymongo import MongoClient
from flask_paginate import Pagination, get_page_args
from bson.objectid import ObjectId

app = Flask(__name__)

# MongoDB Connection
client = MongoClient("mongodb://localhost:27017/")
db = client["product_data"]
collection = db["products"]

# Helper function for pagination
def get_products(offset=0, per_page=10):
    return list(collection.find().skip(offset).limit(per_page))

@app.route('/')
def index():
    page, per_page, offset = get_page_args(page_parameter='page', per_page_parameter='per_page')
    total = collection.count_documents({})
    products = get_products(offset=offset, per_page=per_page)
    pagination = Pagination(page=page, per_page=per_page, total=total, css_framework='bootstrap4')
    return render_template('products.html', products=products, pagination=pagination)

# Product detail route
@app.route('/product/<product_id>')
def product_detail(product_id):
    try:
        product = collection.find_one({"_id": ObjectId(product_id)})
        if not product:
            return "Product not found", 404
        return render_template('product_detail.html', product=product)
    except Exception as e:
        return f"Invalid product ID. Error: {e}", 400

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, threaded=False, port=5001)
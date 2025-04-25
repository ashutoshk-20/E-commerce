import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import joblib
import os
from datetime import datetime
import re

class CategoryBasedRecommender:
    def __init__(self):
        """Initialize the recommender system."""
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.scaler = StandardScaler()
        self.train_data = None
        self.tfidf_matrix = None
        self.feature_matrix = None
        
    def extract_category(self, title):
        """Improved category extraction from product title based on actual data patterns."""
        # Common brand names and their variations based on actual product data
        brand_mapping = {
            'boat': [
                'boat', 'bo at', 'bo-at', 'boatairdopes', 'boatbassheads', 'boatrockerz',
                'boat rockerz', 'boat bassheads', 'boat airdopes', 'boat rockerz 235v2',
                'boat rockerz 255', 'boat bassheads 100', 'boat airdopes 131',
                'boat rockerz 235v2 with asap charging version 5.0'
            ],
            'oneplus': [
                'oneplus', 'one plus', 'one-plus', 'oneplusbullets', 'oneplusbuds',
                'oneplus bullets', 'oneplus bullets wireless z', 'oneplus bullets wireless z bass edition',
                'oneplus bullet 2', 'oneplus bullets wireless z bass edition bluetooth headset'
            ],
            'realme': [
                'realme', 'real me', 'real-me', 'realme buds', 'realme buds wireless',
                'realme buds 2', 'realme buds q', 'realme buds wireless bluetooth headset',
                'realme buds 2 wired headset', 'realme buds q bluetooth headset'
            ],
            'u&i': [
                'u&i', 'u&i buds', 'u&i titanic', 'u&i titanic series',
                'u&i titanic series bluetooth neckband bluetooth headset',
                'u&i titanic series - low price bluetooth neckband bluetooth headset'
            ],
            'titanic': [
                'titanic', 'titanic buds', 'titanic series', 'titanic series bluetooth neckband',
                'titanic series - low price bluetooth neckband bluetooth headset'
            ],
            'samsung': [
                'samsung', 'samsung galaxy buds', 'samsung galaxy'
            ],
            'apple': [
                'apple', 'airpods', 'apple airpods'
            ],
            'mi': [
                'mi', 'xiaomi', 'redmi', 'mi buds', 'mi wireless'
            ]
        }
        
        title = title.lower()
        
        # Special handling for Titanic series products
        if 'titanic' in title:
            return 'u&i'  # Titanic series products are from U&I
            
        for category, variations in brand_mapping.items():
            for variation in variations:
                if variation in title:
                    return category
        return 'other'
        
    def load_data(self, train_path):
        """Load and preprocess training data with improved feature engineering."""
        print("Loading training data...")
        self.train_data = pd.read_csv(train_path)
        
        # Create combined text feature with brand emphasis
        self.train_data['combined_text'] = (
            self.train_data['product_title'].fillna('') + ' ' +
            self.train_data['review'].fillna('') + ' ' + 
            self.train_data['summary'].fillna('')
        )
        
        # Extract category using improved method
        self.train_data['category'] = self.train_data['product_title'].apply(
            self.extract_category
        )
        
        # Create brand-specific features
        self.train_data['is_boat'] = self.train_data['category'] == 'boat'
        self.train_data['is_oneplus'] = self.train_data['category'] == 'oneplus'
        self.train_data['is_realme'] = self.train_data['category'] == 'realme'
        self.train_data['is_u&i'] = self.train_data['category'] == 'u&i'
        self.train_data['is_titanic'] = self.train_data['category'] == 'titanic'
        
        # Create numerical features
        self.train_data['review_length'] = self.train_data['review'].apply(
            lambda x: len(str(x).split())
        )
        
        # Create TF-IDF matrix
        print("Creating TF-IDF matrix...")
        self.tfidf_matrix = self.vectorizer.fit_transform(self.train_data['combined_text'])
        
        # Create feature matrix with weighted features
        print("Creating feature matrix...")
        numerical_features = self.train_data[[
            'rating', 'review_length', 'is_boat', 'is_oneplus', 
            'is_realme', 'is_u&i', 'is_titanic'
        ]].values
        scaled_features = self.scaler.fit_transform(numerical_features)
        
        # Apply weights to features
        rating_weight = 0.25
        review_weight = 0.15
        brand_weight = 0.45  # Increased weight for brand-specific features
        text_weight = 0.15
        
        scaled_features[:, 0] *= rating_weight  # Rating weight
        scaled_features[:, 1] *= review_weight  # Review length weight
        scaled_features[:, 2:] *= brand_weight  # Brand weights
        
        # Combine TF-IDF and numerical features
        self.feature_matrix = np.hstack([
            self.tfidf_matrix.toarray() * text_weight,
            scaled_features
        ])
        
        print("Data preprocessing completed.")
        
    def train(self):
        """Train the recommendation model."""
        print("Training recommendation model...")
        # Calculate similarity matrix with proper normalization
        self.similarity_matrix = cosine_similarity(self.feature_matrix)
        
        # Normalize similarity scores to [0, 1] range
        self.similarity_matrix = (self.similarity_matrix + 1) / 2
        
        print("Model training completed.")
        
    def get_recommendations(self, category, n_recommendations=10):
        """Get recommendations for a specific category."""
        # Convert category to lowercase
        category = category.lower()
        
        # Special handling for Titanic series
        if category == 'titanic':
            category = 'u&i'  # Titanic series products are from U&I
        
        # Filter products by category
        category_mask = self.train_data['category'] == category
        if not category_mask.any():
            return []
        
        # Get indices of products in the category
        category_indices = self.train_data[category_mask].index
        
        # Calculate similarity scores for products in the category
        category_similarities = self.similarity_matrix[category_indices].mean(axis=0)
        
        # Get top recommendations
        top_indices = np.argsort(category_similarities)[::-1]
        
        # Get unique product titles with brand-specific filtering
        unique_products = {}
        for idx in top_indices:
            product = self.train_data.iloc[idx]
            title = product['product_title']
            
            # For Titanic series, only include products with 'titanic' in the title
            if category == 'u&i' and 'titanic' not in title.lower():
                continue
                
            if title not in unique_products:
                unique_products[title] = {
                    'product_id': product['product_id'],
                    'product_title': title,
                    'rating': product['rating'],
                    'category': product['category'],
                    'similarity_score': float(category_similarities[idx])  # Convert to float
                }
            if len(unique_products) >= n_recommendations:
                break
        
        # If we don't have enough unique products, try to get more
        if len(unique_products) < n_recommendations:
            # Get all products in the category
            category_products = self.train_data[category_mask]
            
            # Sort by rating and review length
            category_products = category_products.sort_values(
                by=['rating', 'review_length'],
                ascending=[False, False]
            )
            
            # Add more products until we reach n_recommendations
            for _, product in category_products.iterrows():
                title = product['product_title']
                if title not in unique_products:
                    unique_products[title] = {
                        'product_id': product['product_id'],
                        'product_title': title,
                        'rating': product['rating'],
                        'category': product['category'],
                        'similarity_score': 0.5  # Default similarity score for additional products
                    }
                if len(unique_products) >= n_recommendations:
                    break
        
        # Sort recommendations by similarity score and rating
        recommendations = sorted(
            unique_products.values(),
            key=lambda x: (x['similarity_score'], x['rating']),
            reverse=True
        )
        
        return recommendations
    
    def save_model(self, model_dir='models'):
        """Save the trained model."""
        os.makedirs(model_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save model components
        joblib.dump(self.vectorizer, f'{model_dir}/vectorizer_{timestamp}.joblib')
        joblib.dump(self.scaler, f'{model_dir}/scaler_{timestamp}.joblib')
        joblib.dump(self.similarity_matrix, f'{model_dir}/similarity_matrix_{timestamp}.joblib')
        
        # Save training data
        self.train_data.to_csv(f'{model_dir}/train_data_{timestamp}.csv', index=False)
        
        print(f"Model saved to {model_dir}")
    
    def load_model(self, model_dir='models', timestamp=None):
        """Load a trained model."""
        if timestamp is None:
            # Get the latest model files
            files = os.listdir(model_dir)
            timestamps = [f.split('_')[1].split('.')[0] for f in files if f.startswith('vectorizer_')]
            if not timestamps:
                raise ValueError("No trained models found")
            timestamp = max(timestamps)
        
        # Load model components
        self.vectorizer = joblib.load(f'{model_dir}/vectorizer_{timestamp}.joblib')
        self.scaler = joblib.load(f'{model_dir}/scaler_{timestamp}.joblib')
        self.similarity_matrix = joblib.load(f'{model_dir}/similarity_matrix_{timestamp}.joblib')
        
        # Load training data
        self.train_data = pd.read_csv(f'{model_dir}/train_data_{timestamp}.csv')
        
        print(f"Model loaded from {model_dir}")

def main():
    # Initialize recommender
    recommender = CategoryBasedRecommender()
    
    # Load and train model
    train_path = 'data/processed/train.csv'  # Update with your train file
    recommender.load_data(train_path)
    recommender.train()
    
    # Save the model
    recommender.save_model()
    
    # Test recommendations
    test_categories = ['boat', 'oneplus','realme','u&i','titanic']
    for category in test_categories:
        print(f"\nRecommendations for category: {category}")
        recommendations = recommender.get_recommendations(category)
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. {rec['product_title']}")
            print(f"   Rating: {rec['rating']:.1f}")
            print(f"   Similarity Score: {rec['similarity_score']:.3f}")

if __name__ == "__main__":
    main() 
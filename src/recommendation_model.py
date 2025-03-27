import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import joblib
import os
from datetime import datetime

class CategoryBasedRecommender:
    def __init__(self):
        """Initialize the recommender system."""
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.scaler = StandardScaler()
        self.train_data = None
        self.tfidf_matrix = None
        self.feature_matrix = None
        
    def load_data(self, train_path):
        """Load and preprocess training data."""
        print("Loading training data...")
        self.train_data = pd.read_csv(train_path)
        
        # Create combined text feature
        self.train_data['combined_text'] = (
            self.train_data['review'].fillna('') + ' ' + 
            self.train_data['summary'].fillna('')
        )
        
        # Extract category from product title
        self.train_data['category'] = self.train_data['product_title'].apply(
            lambda x: x.split()[0].lower()
        )
        
        # Create numerical features
        self.train_data['review_length'] = self.train_data['review'].apply(
            lambda x: len(str(x).split())
        )
        
        # Create TF-IDF matrix
        print("Creating TF-IDF matrix...")
        self.tfidf_matrix = self.vectorizer.fit_transform(self.train_data['combined_text'])
        
        # Create feature matrix
        print("Creating feature matrix...")
        numerical_features = self.train_data[['rating', 'review_length']].values
        scaled_features = self.scaler.fit_transform(numerical_features)
        
        # Combine TF-IDF and numerical features
        self.feature_matrix = np.hstack([
            self.tfidf_matrix.toarray(),
            scaled_features
        ])
        
        print("Data preprocessing completed.")
        
    def train(self):
        """Train the recommendation model."""
        print("Training recommendation model...")
        # Calculate similarity matrix
        self.similarity_matrix = cosine_similarity(self.feature_matrix)
        print("Model training completed.")
        
    def get_recommendations(self, category, n_recommendations=5):
        """Get recommendations for a specific category."""
        # Convert category to lowercase
        category = category.lower()
        
        # Filter products by category
        category_mask = self.train_data['category'] == category
        if not category_mask.any():
            return []
        
        # Get indices of products in the category
        category_indices = self.train_data[category_mask].index
        
        # Calculate average similarity scores for products in the category
        category_similarities = self.similarity_matrix[category_indices].mean(axis=0)
        
        # Get top recommendations (excluding products in the same category)
        category_similarities[category_indices] = -1  # Exclude products in the same category
        top_indices = np.argsort(category_similarities)[-n_recommendations:][::-1]
        
        # Get recommended products
        recommendations = []
        for idx in top_indices:
            product = self.train_data.iloc[idx]
            recommendations.append({
                'product_id': product['product_id'],
                'product_title': product['product_title'],
                'rating': product['rating'],
                'category': product['category'],
                'similarity_score': category_similarities[idx]
            })
        
        return recommendations
    
    def save_model(self, model_dir='../models'):
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
    
    def load_model(self, model_dir='../models', timestamp=None):
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
    train_path = '../data/processed/train.csv'  # Update with your train file
    recommender.load_data(train_path)
    recommender.train()
    
    # Save the model
    recommender.save_model()
    
    # Test recommendations
    test_categories = ['boat', 'Oneplus']
    for category in test_categories:
        print(f"\nRecommendations for category: {category}")
        recommendations = recommender.get_recommendations(category)
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. {rec['product_title']}")
            print(f"   Rating: {rec['rating']:.1f}")
            print(f"   Similarity Score: {rec['similarity_score']:.3f}")

if __name__ == "__main__":
    main() 
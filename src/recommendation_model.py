import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import joblib
import os
from datetime import datetime

class MobileRecommender:
    def __init__(self):
        """Initialize with smaller TF-IDF dimensions and NN model"""
        self.vectorizer = TfidfVectorizer(max_features=2000, stop_words='english')
        self.scaler = StandardScaler()
        self.nn_model = None
        self.train_data = None
        self.feature_matrix = None
        self._reset_indices = False  # Track if we need to reset indices
        
    def extract_brand(self, title):
        """Improved brand extraction for mobile devices"""
        brand_mapping = {
            'apple': ['iphone', 'apple'],
            'samsung': ['samsung', 'galaxy'],
            'redmi': ['redmi', 'xiaomi', 'mi'],
            'realme': ['realme', 'narzo'],
            'oneplus': ['oneplus', 'one plus'],
            'other': []
        }
        
        title = str(title).lower()
        for brand, keywords in brand_mapping.items():
            if any(keyword in title for keyword in keywords):
                return brand
        return 'other'
        
    def load_data(self, data_path, sample_size=5000):
        """Load and preprocess data with memory efficiency"""
        print("Loading and sampling data...")
        raw_data = pd.read_csv(data_path)
        
        # Sample the data to reduce size and reset indices
        self.train_data = raw_data.sample(min(sample_size, len(raw_data)), random_state=42).reset_index(drop=True)
        self._reset_indices = True
        
        # Rename columns
        self.train_data = self.train_data.rename(columns={
            'Product Titles': 'product_title',
            'Review Titles': 'summary',
            'Review Texts': 'review',
            'Ratings': 'rating'
        })
        
        # Extract brand
        self.train_data['brand'] = self.train_data['product_title'].apply(self.extract_brand)
        
        # Create brand features
        for brand in ['apple', 'samsung', 'redmi', 'realme', 'oneplus']:
            self.train_data[f'is_{brand}'] = (self.train_data['brand'] == brand).astype(int)
        
        # Combine text features
        self.train_data['combined_text'] = (
            self.train_data['product_title'].fillna('') + ' ' +
            self.train_data['summary'].fillna('') + ' ' + 
            self.train_data['review'].fillna('')
        )
        
        # Create numerical features
        self.train_data['review_length'] = self.train_data['review'].apply(
            lambda x: len(str(x).split()))
        
        # Create TF-IDF matrix
        print("Creating TF-IDF matrix...")
        self.tfidf_matrix = self.vectorizer.fit_transform(self.train_data['combined_text'])
        
        # Create feature matrix
        print("Creating feature matrix...")
        numerical_features = self.train_data[[
            'rating', 'review_length', 'is_apple', 'is_samsung', 
            'is_redmi', 'is_realme', 'is_oneplus'
        ]].values
        
        # Scale features
        scaled_features = self.scaler.fit_transform(numerical_features)
        
        # Apply weights
        rating_weight = 0.3
        review_weight = 0.1
        brand_weight = 0.4
        text_weight = 0.2
        
        scaled_features[:, 0] *= rating_weight
        scaled_features[:, 1] *= review_weight
        scaled_features[:, 2:] *= brand_weight
        
        # Combine features
        self.feature_matrix = np.hstack([
            self.tfidf_matrix.toarray() * text_weight,
            scaled_features
        ])
        
        print(f"Preprocessing complete. Final dataset size: {len(self.train_data)}")
        
    def train(self):
        """Train approximate nearest neighbors model"""
        print("Training NearestNeighbors model...")
        self.nn_model = NearestNeighbors(
            n_neighbors=min(50, len(self.feature_matrix)),  # Ensure we don't ask for more neighbors than samples
            metric='cosine',
            algorithm='brute',  # Most memory-efficient option
            n_jobs=-1  # Use all cores
        )
        self.nn_model.fit(self.feature_matrix)
        print("Training completed.")
        
    def get_recommendations(self, brand, n=10):
        """Get recommendations using approximate NN search"""
        brand = brand.lower()
        brand_mask = self.train_data['brand'] == brand
        
        if not brand_mask.any():
            print(f"No products found for brand: {brand}")
            return []
        
        # Get all products from the brand
        brand_indices = self.train_data[brand_mask].index
        
        # If we have enough brand products, use them as queries
        if len(brand_indices) >= 5:
            query_indices = brand_indices[:5]  # Use first 5 as queries
        else:
            query_indices = brand_indices
        
        # Find similar items
        recommendations = []
        seen_titles = set()
        
        for idx in query_indices:
            distances, indices = self.nn_model.kneighbors(
                [self.feature_matrix[idx]], 
                n_neighbors=min(3*n, len(self.feature_matrix))  # Get extra to account for duplicates
            )
            
            for i, neighbor_idx in enumerate(indices[0]):
                product = self.train_data.iloc[neighbor_idx]
                title = product['product_title']
                
                if (title not in seen_titles and 
                    product['brand'] == brand and 
                    len(recommendations) < n):
                    recommendations.append({
                        'product_title': title,
                        'rating': product['rating'],
                        'similarity_score': float(1 - distances[0][i])  # Convert cosine distance to similarity
                    })
                    seen_titles.add(title)
                
                if len(recommendations) >= n:
                    break
                    
        # Sort by similarity and rating
        return sorted(recommendations, key=lambda x: (-x['similarity_score'], -x['rating']))
    
    def save_model(self, model_dir='models'):
        """Save model components"""
        os.makedirs(model_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        joblib.dump(self.vectorizer, f'{model_dir}/vectorizer_{timestamp}.joblib')
        joblib.dump(self.scaler, f'{model_dir}/scaler_{timestamp}.joblib')
        joblib.dump(self.nn_model, f'{model_dir}/nn_model_{timestamp}.joblib')
        self.train_data.to_csv(f'{model_dir}/train_data_{timestamp}.csv', index=False)
        
        print(f"Model saved to {model_dir} with timestamp {timestamp}")
    
    def load_model(self, model_dir='models', timestamp=None):
        """Load saved model"""
        if timestamp is None:
            files = os.listdir(model_dir)
            timestamps = [f.split('_')[1].split('.')[0] for f in files if f.startswith('vectorizer_')]
            if not timestamps:
                raise ValueError("No trained models found")
            timestamp = max(timestamps)
        
        self.vectorizer = joblib.load(f'{model_dir}/vectorizer_{timestamp}.joblib')
        self.scaler = joblib.load(f'{model_dir}/scaler_{timestamp}.joblib')
        self.nn_model = joblib.load(f'{model_dir}/nn_model_{timestamp}.joblib')
        self.train_data = pd.read_csv(f'{model_dir}/train_data_{timestamp}.csv')
        
        # Rebuild feature matrix if needed
        if not hasattr(self, 'feature_matrix'):
            self._rebuild_feature_matrix()
        
        print(f"Model loaded from {model_dir} (timestamp: {timestamp})")
    
    def _rebuild_feature_matrix(self):
        """Rebuild feature matrix after loading model"""
        if self.train_data is None or self.vectorizer is None or self.scaler is None:
            raise ValueError("Required components not loaded")
            
        print("Rebuilding feature matrix...")
        # Recreate TF-IDF matrix
        self.tfidf_matrix = self.vectorizer.transform(self.train_data['combined_text'])
        
        # Recreate numerical features
        numerical_features = self.train_data[[
            'rating', 'review_length', 'is_apple', 'is_samsung', 
            'is_redmi', 'is_realme', 'is_oneplus'
        ]].values
        
        # Scale features
        scaled_features = self.scaler.transform(numerical_features)
        
        # Apply weights (same as in load_data)
        rating_weight = 0.3
        review_weight = 0.1
        brand_weight = 0.4
        text_weight = 0.2
        
        scaled_features[:, 0] *= rating_weight
        scaled_features[:, 1] *= review_weight
        scaled_features[:, 2:] *= brand_weight
        
        # Combine features
        self.feature_matrix = np.hstack([
            self.tfidf_matrix.toarray() * text_weight,
            scaled_features
        ])

def main():
    # Initialize recommender
    recommender = MobileRecommender()
    
    # Load and train
    recommender.load_data('data/raw/amazon_mobiles_reviews.csv', sample_size=5000)
    recommender.train()
    recommender.save_model()
    
    # Test recommendations
    test_brands = ['apple', 'samsung', 'redmi', 'realme', 'oneplus']
    
    for brand in test_brands:
        print(f"\n=== Top {brand.upper()} Recommendations ===")
        recs = recommender.get_recommendations(brand)
        
        if not recs:
            print(f"No recommendations available for {brand}")
            continue
            
        for i, rec in enumerate(recs, 1):
            print(f"{i}. {rec['product_title']}")
            print(f"   Rating: {rec['rating']:.1f} | Similarity: {rec['similarity_score']:.2f}")
            print("-" * 80)

if __name__ == "__main__":
    main()
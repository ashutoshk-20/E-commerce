import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer #type: ignore
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import joblib
import os
from datetime import datetime
from tqdm import tqdm
import faiss #type: ignore
import torch #type: ignore
import sys

class MobileRecommender:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """Initialize with version-compatible components"""
        # Version check
        if sys.version_info >= (3, 12):
            print("Python 3.12 detected - using compatible configurations")
            
        self.device = device
        self.vectorizer = SentenceTransformer('all-mpnet-base-v2', device=device)
        self.scaler = StandardScaler()
        self.nn_model = None
        self.faiss_index = None
        self.train_data = None
        self.feature_matrix = None
        self._reset_indices = False
        
    def _initialize_faiss(self, dimensions):
        """FAISS initialization compatible with Python 3.12"""
        self.faiss_index = faiss.IndexFlatIP(dimensions)
        if torch.cuda.is_available():
            res = faiss.StandardGpuResources()
            self.faiss_index = faiss.index_cpu_to_gpu(res, 0, self.faiss_index)
        
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
        
    def load_data(self, data_path, sample_size=10000):
        """Load and preprocess data with memory efficiency"""
        print("Loading and sampling data...")
        
        # Use pandas with PyArrow backend for better 3.12 compatibility
        try:
            raw_data = pd.read_csv(data_path, engine='pyarrow')
        except:
            raw_data = pd.read_csv(data_path)
            
        self.train_data = raw_data.sample(min(sample_size, len(raw_data)), random_state=42).reset_index(drop=True)
        self._reset_indices = True
        
        # Rename columns with safer string operations
        column_map = {
            col: col.lower().replace(' ', '_') for col in raw_data.columns
        }
        self.train_data = self.train_data.rename(columns=column_map)
        
        # Ensure required columns exist
        required_cols = ['product_titles', 'review_titles', 'review_texts', 'ratings']
        for col in required_cols:
            if col not in self.train_data.columns:
                raise ValueError(f"Missing required column: {col}")
                
        self.train_data = self.train_data.rename(columns={
            'product_titles': 'product_title',
            'review_titles': 'summary',
            'review_texts': 'review',
            'ratings': 'rating'
        })
        
        # Extract brand
        self.train_data['brand'] = self.train_data['product_title'].apply(self.extract_brand)
        
        # Create brand features
        for brand in ['apple', 'samsung', 'redmi', 'realme', 'oneplus']:
            self.train_data[f'is_{brand}'] = (self.train_data['brand'] == brand).astype('int8')  # Save memory
            
        # Combine text features with null checks
        self.train_data['combined_text'] = (
            self.train_data['product_title'].fillna('') + ' ' +
            self.train_data['summary'].fillna('') + ' ' + 
            self.train_data['review'].fillna('')
        ).str.strip()
        
        # Create numerical features
        self.train_data['review_length'] = self.train_data['review'].apply(
            lambda x: len(str(x).split())).astype('int16')
        
        # Create text embeddings
        print("Creating text embeddings...")
        text_embeddings = self.vectorizer.encode(
            self.train_data['combined_text'].tolist(),
            show_progress_bar=True,
            batch_size=32,
            convert_to_numpy=True,
            device=self.device
        )
        
        # Create numerical features with memory efficiency
        numerical_features = self.train_data[
            ['rating', 'review_length', 'is_apple', 'is_samsung', 'is_redmi', 'is_realme', 'is_oneplus']
        ].to_numpy(dtype='float32')
        
        # Scale features
        scaled_features = self.scaler.fit_transform(numerical_features)
        
        # Apply weights
        rating_weight = 0.25
        review_weight = 0.05
        brand_weight = 0.25
        text_weight = 0.45
        
        scaled_features[:, 0] *= rating_weight
        scaled_features[:, 1] *= review_weight
        scaled_features[:, 2:] *= brand_weight
        
        # Combine features
        self.feature_matrix = np.hstack([
            text_embeddings * text_weight,
            scaled_features
        ]).astype('float32')  # FAISS requires float32
        
        print(f"Preprocessing complete. Final dataset size: {len(self.train_data)}")
        
    def train(self, use_faiss=True):
        """Train the recommendation model"""
        if use_faiss:
            print("Training FAISS index...")
            self._initialize_faiss(self.feature_matrix.shape[1])
            self.faiss_index.add(self.feature_matrix)
        else:
            print("Training NearestNeighbors model...")
            self.nn_model = NearestNeighbors(
                n_neighbors=min(100, len(self.feature_matrix)),
                metric='cosine',
                algorithm='brute',
                n_jobs=-1
            )
            self.nn_model.fit(self.feature_matrix.astype('float64'))  # sklearn works better with float64
        print("Training completed.")
        
    def get_recommendations(self, brand, n=10, min_rating=3.5):
        """Get recommendations with 3.12 compatible operations"""
        brand = str(brand).lower()  # Ensure string type
        brand_mask = self.train_data['brand'] == brand
        
        if not brand_mask.any():
            print(f"No products found for brand: {brand}")
            return []
            
        brand_products = self.train_data.loc[brand_mask]
        
        if min_rating is not None:
            brand_products = brand_products[brand_products['rating'] >= min_rating]
            if len(brand_products) == 0:
                print(f"No products with rating >= {min_rating}")
                return []
                
        recommendations = []
        seen_titles = set()
        query_indices = brand_products.sample(min(5, len(brand_products))).index
        
        for idx in query_indices:
            if hasattr(self, 'faiss_index') and self.faiss_index is not None:
                query = self.feature_matrix[idx:idx+1]
                distances, indices = self.faiss_index.search(query, k=min(3*n, len(self.feature_matrix)))
                similarities = distances[0]  # FAISS returns similarities directly
            else:
                query = self.feature_matrix[idx:idx+1].astype('float64')
                distances, indices = self.nn_model.kneighbors(query, n_neighbors=min(3*n, len(self.feature_matrix)))
                similarities = 1 - distances[0]
                
            for i, neighbor_idx in enumerate(indices[0]):
                product = self.train_data.iloc[neighbor_idx]
                title = product['product_title']
                
                if (title not in seen_titles and product['brand'] == brand and 
                    (min_rating is None or product['rating'] >= min_rating)):
                    recommendations.append({
                        'product_title': title,
                        'rating': float(product['rating']),
                        'similarity_score': float(similarities[i])
                    })
                    seen_titles.add(title)
                    
                if len(recommendations) >= n:
                    break
                    
        return sorted(recommendations, key=lambda x: (-x['similarity_score'], -x['rating']))[:n]
    
    def save_model(self, model_dir='models'):
        """Save model with 3.12 compatible operations"""
        os.makedirs(model_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save components with version-aware serialization
        self.vectorizer.save(f'{model_dir}/sentence_transformer_{timestamp}')
        joblib.dump(self.scaler, f'{model_dir}/scaler_{timestamp}.joblib')
        
        if self.faiss_index is not None:
            if torch.cuda.is_available():
                cpu_index = faiss.index_gpu_to_cpu(self.faiss_index)
                faiss.write_index(cpu_index, f'{model_dir}/faiss_index_{timestamp}.index')
            else:
                faiss.write_index(self.faiss_index, f'{model_dir}/faiss_index_{timestamp}.index')
                
        if self.nn_model is not None:
            joblib.dump(self.nn_model, f'{model_dir}/nn_model_{timestamp}.joblib')
            
        self.train_data.to_parquet(f'{model_dir}/train_data_{timestamp}.parquet')  # More efficient than CSV
        
        print(f"Model saved to {model_dir} with timestamp {timestamp}")
    
    def load_model(self, model_dir='models', timestamp=None):
        """Load model with 3.12 compatibility"""
        if timestamp is None:
            files = os.listdir(model_dir)
            timestamps = [f.split('_')[-1].split('.')[0] for f in files if f.startswith('scaler_')]
            if not timestamps:
                raise ValueError("No trained models found")
            timestamp = max(timestamps)
            
        # Load components
        self.vectorizer = SentenceTransformer(f'{model_dir}/sentence_transformer_{timestamp}', device=self.device)
        self.scaler = joblib.load(f'{model_dir}/scaler_{timestamp}.joblib')
        
        # Check for FAISS or sklearn model
        faiss_file = f'{model_dir}/faiss_index_{timestamp}.index'
        if os.path.exists(faiss_file):
            self.faiss_index = faiss.read_index(faiss_file)
            if torch.cuda.is_available():
                res = faiss.StandardGpuResources()
                self.faiss_index = faiss.index_cpu_to_gpu(res, 0, self.faiss_index)
        else:
            self.nn_model = joblib.load(f'{model_dir}/nn_model_{timestamp}.joblib')
            
        self.train_data = pd.read_parquet(f'{model_dir}/train_data_{timestamp}.parquet')
        self._rebuild_feature_matrix()
        
        print(f"Model loaded from {model_dir} (timestamp: {timestamp})")
    
    def _rebuild_feature_matrix(self):
        """Rebuild feature matrix with 3.12 compatible operations"""
        print("Rebuilding feature matrix...")
        
        text_embeddings = self.vectorizer.encode(
            self.train_data['combined_text'].tolist(),
            show_progress_bar=True,
            batch_size=32,
            device=self.device
        )
        
        numerical_features = self.train_data[
            ['rating', 'review_length', 'is_apple', 'is_samsung', 'is_redmi', 'is_realme', 'is_oneplus']
        ].to_numpy(dtype='float32')
        
        scaled_features = self.scaler.transform(numerical_features)
        scaled_features[:, 0] *= 0.25
        scaled_features[:, 1] *= 0.05
        scaled_features[:, 2:] *= 0.25
        
        self.feature_matrix = np.hstack([
            text_embeddings * 0.45,
            scaled_features
        ]).astype('float32')

def main():
    print("Initializing recommender system...")
    print(f"Python version: {sys.version}")
    
    recommender = MobileRecommender()
    
    try:
        print("\nLoading data...")
        recommender.load_data('data/raw/amazon_mobiles_reviews.csv', sample_size=10000)
        
        print("\nTraining model...")
        recommender.train(use_faiss=True)
        
        print("\nSaving model...")
        recommender.save_model()
        
        print("\nTesting recommendations:")
        test_brands = ['apple', 'samsung', 'redmi', 'realme', 'oneplus']
        
        for brand in test_brands:
            print(f"\n=== Top {brand.upper()} Recommendations ===")
            recs = recommender.get_recommendations(brand, n=3, min_rating=4.0)
            
            for i, rec in enumerate(recs, 1):
                print(f"{i}. {rec['product_title']}")
                print(f"   Rating: {rec['rating']:.1f} | Similarity: {rec['similarity_score']:.3f}")
                
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("Ensure you have all required packages installed:")
        print("pip install sentence-transformers pandas numpy scikit-learn joblib tqdm torch transformers faiss-cpu")

if __name__ == "__main__":
    main()
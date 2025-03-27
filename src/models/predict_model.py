import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model #type:ignore
import joblib
import os
from datetime import datetime
import argparse
from typing import Dict, List, Union, Optional

class RNNPredictor:
    def __init__(self, model_dir: str = '../models'):
        """Initialize the RNN predictor with trained model and preprocessing components."""
        self.model_dir = model_dir
        self.model = None
        self.tokenizer = None
        self.scaler = None
        self.load_latest_model()
    
    def load_latest_model(self) -> None:
        """Load the latest trained model and preprocessing components."""
        # Get the latest model files
        files = os.listdir(self.model_dir)
        timestamps = [f.split('_')[1].split('.')[0] for f in files if f.startswith('rnn_model_')]
        if not timestamps:
            raise ValueError("No trained models found in the specified directory")
        
        latest_timestamp = max(timestamps)
        print(f"Loading model from timestamp: {latest_timestamp}")
        
        # Load model
        self.model = load_model(f'{self.model_dir}/rnn_model_best.h5')
        
        # Load preprocessing components
        self.tokenizer = joblib.load(f'{self.model_dir}/tokenizer_{latest_timestamp}.joblib')
        self.scaler = joblib.load(f'{self.model_dir}/scaler_{latest_timestamp}.joblib')
        
        print("Model and components loaded successfully")
    
    def preprocess_input(self, text: str, rating: float, review_length: int) -> tuple:
        """Preprocess input data for prediction."""
        # Preprocess text
        sequence = self.tokenizer.texts_to_sequences([text])
        padded_sequence = pad_sequences(sequence, maxlen=200)  #type:ignore # Using default max_len
        
        # Scale numerical features
        numerical_features = np.array([[rating, review_length]])
        scaled_features = self.scaler.transform(numerical_features)
        
        return padded_sequence, scaled_features
    
    def predict_single(self, text: str, rating: float, review_length: int) -> float:
        """Make prediction for a single input."""
        padded_sequence, scaled_features = self.preprocess_input(text, rating, review_length)
        prediction = self.model.predict([padded_sequence, scaled_features])
        return prediction[0][0]
    
    def predict_batch(self, data: List[Dict]) -> List[float]:
        """Make predictions for a batch of inputs."""
        predictions = []
        for item in data:
            pred = self.predict_single(
                item['text'],
                item['rating'],
                item['review_length']
            )
            predictions.append(pred)
        return predictions
    
    def analyze_review_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze the sentiment of a review text."""
        # Get prediction for the review
        rating = self.predict_single(text, 0, len(text.split()))
        
        # Calculate sentiment score (-1 to 1)
        sentiment_score = (rating - 3) / 2  # Convert 1-5 rating to -1 to 1 sentiment
        
        return {
            'predicted_rating': rating,
            'sentiment_score': sentiment_score,
            'confidence': 1.0 - abs(sentiment_score)  # Higher confidence for neutral sentiment
        }
    
    def get_recommendations(self, 
                          user_reviews: List[Dict], 
                          product_catalog: pd.DataFrame,
                          n_recommendations: int = 5) -> List[Dict]:
        """Get personalized product recommendations based on user reviews."""
        # Calculate user's average sentiment
        user_sentiments = [self.analyze_review_sentiment(review['text'])['sentiment_score'] 
                         for review in user_reviews]
        avg_sentiment = np.mean(user_sentiments) if user_sentiments else 0
        
        # Predict ratings for all products in catalog
        product_predictions = []
        for _, product in product_catalog.iterrows():
            pred_rating = self.predict_single(
                product['combined_text'],
                product['rating'],
                product['review_length']
            )
            product_predictions.append({
                'product_id': product['product_id'],
                'title': product['product_title'],
                'predicted_rating': pred_rating,
                'actual_rating': product['rating'],
                'sentiment_score': (pred_rating - 3) / 2
            })
        
        # Sort by predicted rating and sentiment alignment
        product_predictions.sort(
            key=lambda x: (x['predicted_rating'], 
                         1 - abs(x['sentiment_score'] - avg_sentiment)),
            reverse=True
        )
        
        return product_predictions[:n_recommendations]

def main():
    parser = argparse.ArgumentParser(description='RNN-based Product Rating Predictor')
    parser.add_argument('--mode', choices=['single', 'batch', 'recommend'], 
                      default='single', help='Prediction mode')
    parser.add_argument('--input_file', type=str, help='Path to input CSV file for batch predictions')
    parser.add_argument('--catalog_file', type=str, help='Path to product catalog CSV file')
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = RNNPredictor()
    
    if args.mode == 'single':
        # Interactive single prediction
        print("\nEnter review details (or 'q' to quit):")
        while True:
            text = input("\nReview text: ")
            if text.lower() == 'q':
                break
            
            try:
                rating = float(input("Rating (1-5): "))
                review_length = len(text.split())
                
                prediction = predictor.predict_single(text, rating, review_length)
                sentiment = predictor.analyze_review_sentiment(text)
                
                print("\nPrediction Results:")
                print(f"Predicted Rating: {prediction:.2f}")
                print(f"Sentiment Score: {sentiment['sentiment_score']:.2f}")
                print(f"Confidence: {sentiment['confidence']:.2f}")
                
            except ValueError:
                print("Invalid input. Please try again.")
    
    elif args.mode == 'batch':
        # Batch prediction from file
        if not args.input_file:
            print("Please provide an input file path using --input_file")
            return
        
        df = pd.read_csv(args.input_file)
        predictions = predictor.predict_batch(df.to_dict('records'))
        
        # Add predictions to dataframe
        df['predicted_rating'] = predictions
        output_file = f'predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        df.to_csv(output_file, index=False)
        print(f"\nPredictions saved to {output_file}")
    
    elif args.mode == 'recommend':
        # Get personalized recommendations
        if not args.catalog_file:
            print("Please provide a catalog file path using --catalog_file")
            return
        
        catalog = pd.read_csv(args.catalog_file)
        
        # Sample user reviews (in practice, these would come from user history)
        user_reviews = [
            {'text': catalog['combined_text'].iloc[0]},
            {'text': catalog['combined_text'].iloc[1]}
        ]
        
        recommendations = predictor.get_recommendations(user_reviews, catalog)
        
        print("\nRecommended Products:")
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. {rec['title']}")
            print(f"   Predicted Rating: {rec['predicted_rating']:.2f}")
            print(f"   Sentiment Score: {rec['sentiment_score']:.2f}")

if __name__ == "__main__":
    main()

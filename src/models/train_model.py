import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.text import Tokenizer #type:ignore 
from tensorflow.keras.preprocessing.sequence import pad_sequences #type:ignore
from tensorflow.keras.models import Sequential, load_model #type:ignore
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional #type:ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint #type:ignore
import tensorflow as tf
import os
from datetime import datetime
import joblib

class RNNRecommender:
    def __init__(self, max_words=10000, max_len=200, embedding_dim=100):
        """Initialize the RNN recommender system."""
        self.max_words = max_words
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.tokenizer = Tokenizer(num_words=max_words)
        self.scaler = StandardScaler()  
        self.model = None
        self.history = None
        
    def preprocess_data(self, df):
        """Preprocess the data for RNN model."""
        print("Preprocessing data...")
        
        # Combine review and summary
        df['combined_text'] = (
            df['review'].fillna('') + ' ' + 
            df['summary'].fillna('')
        )
        
        # Tokenize text
        self.tokenizer.fit_on_texts(df['combined_text'])
        sequences = self.tokenizer.texts_to_sequences(df['combined_text'])
        X_text = pad_sequences(sequences, maxlen=self.max_len)
        
        # Scale numerical features
        numerical_features = df[['rating', 'review_length']].values
        X_numerical = self.scaler.fit_transform(numerical_features)
        
        # Combine features
        X = np.hstack([X_text, X_numerical])
        
        # Create target (rating)
        y = df['rating'].values
        
        return X, y
    
    def build_model(self):
        """Build the RNN model architecture."""
        print("Building model...")
        
        # Text input branch
        text_input = tf.keras.Input(shape=(self.max_len,))
        x = Embedding(self.max_words, self.embedding_dim)(text_input)
        x = Bidirectional(LSTM(64, return_sequences=True))(x)
        x = Bidirectional(LSTM(32))(x)
        x = Dropout(0.5)(x)
        
        # Numerical input branch
        numerical_input = tf.keras.Input(shape=(2,))
        y = Dense(32, activation='relu')(numerical_input)
        y = Dropout(0.3)(y)
        
        # Combine branches
        combined = tf.keras.layers.concatenate([x, y])
        z = Dense(64, activation='relu')(combined)
        z = Dropout(0.3)(z)
        z = Dense(32, activation='relu')(z)
        output = Dense(1)(z)
        
        # Create model
        self.model = tf.keras.Model(inputs=[text_input, numerical_input], outputs=output)
        
        # Compile model
        self.model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        print("Model built successfully.")
    
    def train(self, X, y, batch_size=32, epochs=50, validation_split=0.2):
        """Train the RNN model."""
        print("Training model...")
        
        # Split features
        X_text = X[:, :self.max_len]
        X_numerical = X[:, self.max_len:]
        
        # Create callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            ModelCheckpoint(
                '../models/rnn_model_best.h5',
                monitor='val_loss',
                save_best_only=True
            )
        ]
        
        # Train model
        self.history = self.model.fit(
            [X_text, X_numerical],
            y,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        print("Model training completed.")
    
    def save_model(self, model_dir='../models'):
        """Save the trained model and preprocessing components."""
        os.makedirs(model_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save model
        self.model.save(f'{model_dir}/rnn_model_{timestamp}.h5')
        
        # Save preprocessing components
        joblib.dump(self.tokenizer, f'{model_dir}/tokenizer_{timestamp}.joblib')
        joblib.dump(self.scaler, f'{model_dir}/scaler_{timestamp}.joblib')
        
        # Save training history
        history_df = pd.DataFrame(self.history.history)
        history_df.to_csv(f'{model_dir}/training_history_{timestamp}.csv')
        
        print(f"Model and components saved to {model_dir}")
    
    def load_model(self, model_dir='../models', timestamp=None):
        """Load a trained model and preprocessing components."""
        if timestamp is None:
            # Get the latest model files
            files = os.listdir(model_dir)
            timestamps = [f.split('_')[1].split('.')[0] for f in files if f.startswith('rnn_model_')]
            if not timestamps:
                raise ValueError("No trained models found")
            timestamp = max(timestamps)
        
        # Load model
        self.model = load_model(f'{model_dir}/rnn_model_{timestamp}.h5')
        
        # Load preprocessing components
        self.tokenizer = joblib.load(f'{model_dir}/tokenizer_{timestamp}.joblib')
        self.scaler = joblib.load(f'{model_dir}/scaler_{timestamp}.joblib')
        
        print(f"Model loaded from {model_dir}")
    
    def predict(self, text, rating, review_length):
        """Make predictions for new data."""
        # Preprocess text
        sequence = self.tokenizer.texts_to_sequences([text])
        padded_sequence = pad_sequences(sequence, maxlen=self.max_len)
        
        # Scale numerical features
        numerical_features = np.array([[rating, review_length]])
        scaled_features = self.scaler.transform(numerical_features)
        
        # Make prediction
        prediction = self.model.predict([padded_sequence, scaled_features])
        return prediction[0][0]

def main():
    # Load training data
    train_path = '../data/processed/train.csv'
    df = pd.read_csv(train_path)
    
    # Initialize recommender
    recommender = RNNRecommender()
    
    # Preprocess data
    X, y = recommender.preprocess_data(df)
    
    # Build and train model
    recommender.build_model()
    recommender.train(X, y)
    
    # Save model
    recommender.save_model()
    
    # Test predictions
    test_samples = [
        {
            'text': df['combined_text'].iloc[0],
            'rating': df['rating'].iloc[0],
            'review_length': df['review_length'].iloc[0]
        },
        {
            'text': df['combined_text'].iloc[1],
            'rating': df['rating'].iloc[1],
            'review_length': df['review_length'].iloc[1]
        }
    ]
    
    print("\nTesting predictions:")
    for i, sample in enumerate(test_samples, 1):
        prediction = recommender.predict(
            sample['text'],
            sample['rating'],
            sample['review_length']
        )
        print(f"\nSample {i}:")
        print(f"Actual Rating: {sample['rating']:.1f}")
        print(f"Predicted Rating: {prediction:.1f}")

if __name__ == "__main__":
    main() 
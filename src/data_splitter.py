import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
from datetime import datetime

def load_data():
    """Load the product review dataset."""
    return pd.read_csv('../data/raw/flipkart_product_review.csv')

def preprocess_data(df):
    """Preprocess the data for splitting."""
    # Create a copy to avoid modifying the original data
    df = df.copy()
    
    # Convert rating to float if it's not already
    df['rating'] = df['rating'].astype(float)
    
    # Create a combined text feature
    df['combined_text'] = df['review'].fillna('') + ' ' + df['summary'].fillna('')
    
    # Calculate review length
    df['review_length'] = df['review'].apply(lambda x: len(str(x).split()))
    
    # Extract category from product title
    df['category'] = df['product_title'].apply(lambda x: x.split()[0])
    
    return df

def split_data(df, test_size=0.2, random_state=42):
    """Split the data into training and test sets."""
    # Split the data
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df['rating']  # Stratify by rating to maintain distribution
    )
    
    return train_df, test_df

def save_split_data(train_df, test_df):
    """Save the split datasets to CSV files."""
    # Create processed directory if it doesn't exist
    os.makedirs('../data/processed', exist_ok=True)
    
    
    # Save training data
    train_path = f'../data/processed/train.csv'
    train_df.to_csv(train_path, index=False)
    
    # Save test data
    test_path = f'../data/processed/test.csv'
    test_df.to_csv(test_path, index=False)
    
    return train_path, test_path

def print_split_statistics(train_df, test_df):
    """Print statistics about the split datasets."""
    print("\nDataset Split Statistics:")
    print("-" * 50)
    
    # Print total counts
    print(f"Total samples: {len(train_df) + len(test_df)}")
    print(f"Training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # Print rating distribution
    print("\nRating Distribution:")
    print("-" * 50)
    print("Training Set:")
    print(train_df['rating'].value_counts().sort_index())
    print("\nTest Set:")
    print(test_df['rating'].value_counts().sort_index())
    
    # Print category distribution
    print("\nTop 5 Categories Distribution:")
    print("-" * 50)
    print("Training Set:")
    print(train_df['category'].value_counts().head())
    print("\nTest Set:")
    print(test_df['category'].value_counts().head())
    
    # Print review length statistics
    print("\nReview Length Statistics:")
    print("-" * 50)
    print("Training Set:")
    print(train_df['review_length'].describe())
    print("\nTest Set:")
    print(test_df['review_length'].describe())

def main():
    # Load data
    print("Loading data...")
    df = load_data()
    
    # Preprocess data
    print("Preprocessing data...")
    df = preprocess_data(df)
    
    # Split data
    print("Splitting data into train and test sets...")
    train_df, test_df = split_data(df)
    
    # Save split data
    print("Saving split datasets...")
    train_path, test_path = save_split_data(train_df, test_df)
    
    # Print statistics
    print_split_statistics(train_df, test_df)
    
    print("\nSplit datasets have been saved to:")
    print(f"Training data: {train_path}")
    print(f"Test data: {test_path}")

if __name__ == "__main__":
    main() 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
import os
import json

# Create processed directory if it doesn't exist
os.makedirs('../data/processed', exist_ok=True)

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

def load_data():
    """Load the product review dataset."""
    return pd.read_csv('../data/raw/flipkart_product_review.csv')

def analyze_ratings(df):
    """Analyze rating distribution and create visualizations."""
    # Rating distribution
    rating_counts = df['rating'].value_counts().sort_index()
    
    # Create bar plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x=rating_counts.index, y=rating_counts.values)
    plt.title('Distribution of Product Ratings')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.savefig('../data/processed/rating_distribution.png')
    plt.close()
    
    # Calculate rating statistics
    rating_stats = {
        'mean_rating': float(df['rating'].mean()),
        'median_rating': float(df['rating'].median()),
        'rating_std': float(df['rating'].std()),
        'rating_counts': rating_counts.to_dict()
    }
    
    return rating_stats

def analyze_sentiment(df):
    """Analyze sentiment of reviews."""
    # Calculate sentiment scores
    df['sentiment'] = df['review'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    
    # Create sentiment distribution plot
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='sentiment', bins=50)
    plt.title('Distribution of Review Sentiment')
    plt.xlabel('Sentiment Score')
    plt.ylabel('Count')
    plt.savefig('../data/processed/sentiment_distribution.png')
    plt.close()
    
    # Calculate sentiment statistics
    sentiment_stats = {
        'mean_sentiment': float(df['sentiment'].mean()),
        'median_sentiment': float(df['sentiment'].median()),
        'sentiment_std': float(df['sentiment'].std()),
        'positive_reviews': int(len(df[df['sentiment'] > 0])),
        'negative_reviews': int(len(df[df['sentiment'] < 0])),
        'neutral_reviews': int(len(df[df['sentiment'] == 0]))
    }
    
    return sentiment_stats

def analyze_product_categories(df):
    """Analyze product categories and their distributions."""
    # Extract main categories from product titles
    df['category'] = df['product_title'].apply(lambda x: x.split()[0])
    
    # Get top categories
    top_categories = df['category'].value_counts().head(10)
    
    # Create category distribution plot
    plt.figure(figsize=(12, 6))
    sns.barplot(x=top_categories.values, y=top_categories.index)
    plt.title('Top 10 Product Categories')
    plt.xlabel('Count')
    plt.ylabel('Category')
    plt.savefig('../data/processed/category_distribution.png')
    plt.close()
    
    # Calculate category statistics
    category_stats = {
        'total_categories': int(len(df['category'].unique())),
        'top_categories': top_categories.to_dict()
    }
    
    return category_stats

def analyze_review_length(df):
    """Analyze review length distribution."""
    # Calculate review lengths
    df['review_length'] = df['review'].apply(lambda x: len(str(x).split()))
    
    # Create review length distribution plot
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='review_length', bins=50)
    plt.title('Distribution of Review Lengths')
    plt.xlabel('Number of Words')
    plt.ylabel('Count')
    plt.savefig('../data/processed/review_length_distribution.png')
    plt.close()
    
    # Calculate review length statistics
    length_stats = {
        'mean_length': float(df['review_length'].mean()),
        'median_length': float(df['review_length'].median()),
        'max_length': int(df['review_length'].max()),
        'min_length': int(df['review_length'].min())
    }
    
    return length_stats

def main():
    # Load data
    print("Loading data...")
    df = load_data()
    
    # Perform analysis
    print("Analyzing ratings...")
    rating_stats = analyze_ratings(df)
    
    print("Analyzing sentiment...")
    sentiment_stats = analyze_sentiment(df)
    
    print("Analyzing product categories...")
    category_stats = analyze_product_categories(df)
    
    print("Analyzing review lengths...")
    length_stats = analyze_review_length(df)
    
    # Combine all statistics
    analysis_results = {
        'rating_stats': rating_stats,
        'sentiment_stats': sentiment_stats,
        'category_stats': category_stats,
        'length_stats': length_stats
    }
    
    # Convert numpy types to Python native types
    analysis_results = convert_numpy_types(analysis_results)
    
    # Save results to JSON
    with open('../data/processed/analysis_results.json', 'w') as f:
        json.dump(analysis_results, f, indent=4)
    
    print("\nEDA Results Summary:")
    print(f"Total number of reviews: {len(df)}")
    print(f"Average rating: {rating_stats['mean_rating']:.2f}")
    print(f"Average sentiment: {sentiment_stats['mean_sentiment']:.2f}")
    print(f"Number of categories: {category_stats['total_categories']}")
    print(f"Average review length: {length_stats['mean_length']:.2f} words")
    
    print("\nResults have been saved to ../data/processed/")

if __name__ == "__main__":
    main() 
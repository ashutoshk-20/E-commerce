# Product Review Data Analysis

This directory contains the results of Exploratory Data Analysis (EDA) performed on the Flipkart product review dataset.

## Analysis Components

1. **Rating Analysis**
   - Distribution of product ratings
   - Mean, median, and standard deviation of ratings
   - Rating counts by value

2. **Sentiment Analysis**
   - Distribution of review sentiment scores
   - Count of positive, negative, and neutral reviews
   - Mean and median sentiment scores

3. **Product Category Analysis**
   - Distribution of product categories
   - Top 10 most common categories
   - Total number of unique categories

4. **Review Length Analysis**
   - Distribution of review lengths
   - Statistics about review lengths (mean, median, max, min)

## Files

- `eda.py`: The main script that performs the analysis
- `analysis_results.json`: Contains all statistical results
- `rating_distribution.png`: Visualization of rating distribution
- `sentiment_distribution.png`: Visualization of sentiment distribution
- `category_distribution.png`: Visualization of category distribution
- `review_length_distribution.png`: Visualization of review length distribution

## How to Run

1. Make sure you have all required dependencies installed:
```bash
pip install -r requirements.txt
```

2. Run the EDA script:
```bash
python data/processed/eda.py
```

## Results Structure

The `analysis_results.json` file contains the following sections:

```json
{
    "rating_stats": {
        "mean_rating": float,
        "median_rating": float,
        "rating_std": float,
        "rating_counts": dict
    },
    "sentiment_stats": {
        "mean_sentiment": float,
        "median_sentiment": float,
        "sentiment_std": float,
        "positive_reviews": int,
        "negative_reviews": int,
        "neutral_reviews": int
    },
    "category_stats": {
        "total_categories": int,
        "top_categories": dict
    },
    "length_stats": {
        "mean_length": float,
        "median_length": float,
        "max_length": int,
        "min_length": int
    }
}
```

## Visualizations

The generated visualizations provide insights into:
- How ratings are distributed across products
- The overall sentiment of reviews
- The most common product categories
- The typical length of reviews

These insights can be used to:
- Understand customer satisfaction levels
- Identify popular product categories
- Analyze review patterns
- Improve product recommendations 
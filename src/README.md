# Source Code

This directory contains the source code for data processing and model training.

## Data Splitting

The `data_splitter.py` script splits the product review dataset into training and test sets.

### Features

1. **Data Preprocessing**
   - Converts ratings to float type
   - Creates combined text feature from review and summary
   - Calculates review lengths
   - Extracts categories from product titles

2. **Data Splitting**
   - Splits data into 80% training and 20% test sets
   - Maintains rating distribution through stratification
   - Uses random seed for reproducibility

3. **Output**
   - Saves split datasets with timestamps
   - Generates detailed statistics about the split
   - Creates visualizations of the distributions

### Usage

1. Make sure you have the required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the data splitter:
```bash
python data_splitter.py
```

### Output Files

The script generates:
- `train_YYYYMMDD_HHMMSS.csv`: Training dataset
- `test_YYYYMMDD_HHMMSS.csv`: Test dataset

Both files contain:
- Original columns from the raw data
- Additional processed features:
  - `combined_text`: Combined review and summary
  - `review_length`: Number of words in review
  - `category`: Extracted from product title

### Statistics Generated

The script prints:
- Total number of samples
- Training/test split sizes
- Rating distribution in both sets
- Category distribution in both sets
- Review length statistics

### Example Output

```
Dataset Split Statistics:
--------------------------------------------------
Total samples: 1000
Training samples: 800
Test samples: 200

Rating Distribution:
--------------------------------------------------
Training Set:
1    50
2    30
3    100
4    200
5    420

Test Set:
1    12
2    8
3    25
4    50
5    105

...
```

### Notes

- The script uses stratification to maintain the rating distribution
- Timestamps in filenames prevent overwriting previous splits
- All preprocessing steps are documented in the code 
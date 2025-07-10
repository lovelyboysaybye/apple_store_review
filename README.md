
# App Review Analyzer

This script fetches App Store reviews for a given app, processes them, analyzes sentiment, extracts keywords from negative reviews, and optionally clusters review summaries.

## Features

1. **Fetch Reviews**: Uses the SerpApi Apple Reviews API to retrieve reviews.
2. **Save & Load**: Saves fetched reviews to a JSON file; can also load existing review files.
3. **Rating Distribution**: Categorizes reviews by star ratings (negative, neutral, positive).
4. **Sentiment Analysis**: Runs an NLP sentiment model to classify review texts.
5. **Keyword Extraction**: Extracts top keywords from negative reviews using KeyBERT and YAKE.
6. **Clustering (Optional)**: Summarizes negative reviews and clusters the summaries.

## Requirements

- Python 3.7+
- Install dependencies:
  ```bash
  pip install serpapi transformers keybert yake sentence-transformers scikit-learn
  ```

## Usage

```bash
python app_review_analyzer.py [options]
```

### Arguments

- `--max_reviews`: Maximum number of reviews to fetch (integer > 1). Default: `100`
- `--product_id`: App Store product ID (integer). Default: `1459969523`
- `--reviews_file`: Path to JSON file for reviews. Default: `app_{product_id}_review.json`
- `--rewrite_review_file`: `0` to load existing file, `1` to force rewrite. Default: `0`
- `--log_file`: Path to log file (`.log` or `.txt`). Default: `app_{product_id}.log`
- `--top_n`: Number of top keywords to extract per document. Default: `15`
- `--most_common`: Number of most common keywords to display. Default: `50`
- `--clustering`: `0` to skip clustering, `1` to perform clustering. Default: `0`
- `--n_clusters`: Number of clusters for summary clustering. Default: `5`
- `--clusters_file`: Path to save full clusters. Default: `app_{product_id}_clusters.txt`

## Examples

1. **Default run for Nebula: Horoscope & Astrology** (uses default product ID `1459969523`):

   ```bash
   python app_review_analyzer.py --product_id 1459969523 --max_reviews 100 --log_file nebula_reviews.log --reviews_file nebula_reviews.json
   ```

2. **Run for Co–Star Personalized Astrology** (product ID `1264782561`):

   ```bash
   python app_review_analyzer.py --product_id 1264782561 --max_reviews 100 --rewrite_review_file 1 --clustering 1 --n_clusters 5 --log_file costar_reviews.log --reviews_file costar_reviews.json --clusters_file costar_clusters.txt
   ```

In both examples:
- Reviews are fetched and saved.
- The script logs progress to console and log file.
- Sentiment and keyword analyses are performed.
- Clustering is enabled only in the second example.

import time
import json
import serpapi
import logging
import argparse
import os
import random
from collections import Counter, defaultdict
from transformers import pipeline, AutoTokenizer
from keybert import KeyBERT
import yake
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

# Constants
API_KEY_FILE = 'serpai_api_key.txt'  # API key file
COUNTRY = 'us'
SORT = 'mostrecent'
POSITIVE_KEY = 'POSITIVE'
NEUTRAL_KEY = 'NEUTRAL'
NEGATIVE_KEY = 'NEGATIVE'
DEFAULT_ID = 1459969523


def setup_logger(product_id: int, log_file: str) -> logging.Logger:
    logger = logging.getLogger(str(product_id))
    logger.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)

    # File handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


def load_api_key(filepath: str, logger: logging.Logger) -> str:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            key = f.read().strip()
        if not key:
            raise ValueError('API key file is empty.')
        logger.info(f'Loaded API key from {filepath}')
        return key
    except FileNotFoundError:
        logger.error(f'API key file not found: {filepath}')
        raise


def load_reviews_from_file(filepath: str, logger: logging.Logger) -> list:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reviews = json.load(f)
        logger.info(f'Loaded {len(reviews)} reviews from {filepath}')
        return reviews
    except FileNotFoundError:
        logger.warning(f'File {filepath} not found. Returning empty list.')
        return []


def fetch_reviews(api_key: str,
                  product_id: int,
                  country: str,
                  sort: str,
                  max_reviews: int,
                  delay: float,
                  logger: logging.Logger) -> list:
    all_reviews = []
    page = 1
    while len(all_reviews) < max_reviews:
        params = {
            'api_key': api_key,
            'engine': 'apple_reviews',
            'product_id': product_id,
            'country': country,
            'sort': sort,
            'page': page
        }
        logger.info(f'Requesting page {page} of reviews...')
        search = serpapi.search(params)
        results = search.as_dict()

        reviews = results.get('reviews', [])
        if not reviews:
            logger.info('No more reviews returned by API. Stopping fetch.')
            break

        all_reviews.extend(reviews)
        logger.info(f'Fetched {len(reviews)} reviews (total: {len(all_reviews)}) from page {page}')
        page += 1
        time.sleep(delay)

    return all_reviews[:max_reviews]


def save_reviews_to_file(reviews: list, filepath: str, logger: logging.Logger) -> None:
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(reviews, f, indent=4, ensure_ascii=False)
    logger.info(f'Saved {len(reviews)} reviews to {filepath}')


def process_ratings(reviews: list, logger: logging.Logger) -> None:
    """
    Categorize reviews by stars before sentiment analysis:
      - rating < 3: negative
      - rating = 3: neutral
      - rating > 3: positive
    """
    counts = Counter()
    for r in reviews:
        stars = r.get('rating')
        if stars is None:
            continue
        if stars < 3:
            counts[NEGATIVE_KEY] += 1
        elif stars == 3:
            counts[NEUTRAL_KEY] += 1
        else:
            counts[POSITIVE_KEY] += 1
    logger.info(f'Rating distribution by stars: {dict(counts)}')


def analyze_sentiment(texts: list, logger: logging.Logger) -> list:
    sa = pipeline(
        'sentiment-analysis',
        model='nlptown/bert-base-multilingual-uncased-sentiment'
    )
    raw = sa(
        texts,
        batch_size=16,
        truncation=True,
        padding=True
    )
    normalized = []
    for res in raw:
        stars = int(res['label'].split()[0])
        if stars >= 4:
            normalized.append(POSITIVE_KEY)
        elif stars == 3:
            normalized.append(NEUTRAL_KEY)
        else:
            normalized.append(NEGATIVE_KEY)
    logger.info(f'Sentiment distribution by model: {Counter(normalized)}')
    return normalized


def extract_keywords(docs: list, method: str, top_n: int, logger: logging.Logger) -> list:
    kws = []
    if method == 'keybert':
        model = KeyBERT()
        for doc in docs:
            kws += [kw for kw, _ in model.extract_keywords(doc, top_n=top_n)]
    elif method == 'yake':
        extractor = yake.KeywordExtractor(lan='en', top=top_n)
        for doc in docs:
            kws += [kw for kw, _ in sorted(extractor.extract_keywords(doc), key=lambda x: x[1])]
    logger.info(f'Extracted {len(kws)} keywords using {method} (top_n={top_n}).')
    return kws


def summarize_reviews(texts, min_len=20, max_len=50, batch_size=8, logger=None):
    """
    Summarize each review into a short sentence.
    """
    if logger:
        logger.info('Summarizing reviews...')
    summarizer = pipeline(
        'summarization',
        model='facebook/bart-large-cnn',
        tokenizer='facebook/bart-large-cnn',
        framework='pt'
    )
    summaries = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        docs = summarizer(
            batch,
            min_length=min_len,
            max_length=max_len,
            truncation=True
        )
        summaries.extend([doc['summary_text'] for doc in docs])
    return summaries


def cluster_summaries(summaries, n_clusters=5, logger=None):
    """
    Embed summaries and cluster them with KMeans.
    Returns:
      - clusters: dict cluster_id -> list of summaries
    """
    if logger:
        logger.info('Clustering summaries...')
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embedder.encode(summaries, show_progress_bar=False)

    km = KMeans(n_clusters=n_clusters, random_state=42)
    labels = km.fit_predict(embeddings)

    clusters = defaultdict(list)
    for lbl, summ in zip(labels, summaries):
        clusters[lbl].append(summ)
    return clusters


def parse_args():
    parser = argparse.ArgumentParser(description='Fetch and analyze App Store reviews')
    parser.add_argument('--max_reviews', type=int, default=100,
                        help='Maximum number of reviews to fetch (integer > 1)')
    parser.add_argument('--product_id', type=int, default=DEFAULT_ID,
                        help='Product ID of the app (integer)')
    parser.add_argument('--reviews_file', type=str,
                        default='',
                        help='Path to JSON file for reviews')
    parser.add_argument('--rewrite_review_file', type=int, choices=[0, 1], default=0,
                        help='0 to load if exists, 1 to rewrite file')
    parser.add_argument('--log_file', type=str,
                        default='',
                        help='Path to log file (.log or .txt)')
    parser.add_argument('--top_n', type=int, default=15,
                        help='Number of top keywords to extract per document')
    parser.add_argument('--most_common', type=int, default=50,
                        help='Number of most common keywords to display')
    parser.add_argument('--clustering', type=int, choices=[0, 1], default=0,
                        help='0 to skip clustering, 1 to perform clustering')
    parser.add_argument('--n_clusters', type=int, default=5,
                        help='Number of clusters for summarization')
    parser.add_argument('--clusters_file', type=str,
                        default='',
                        help='Path to save all clusters summaries')
    return parser.parse_args()


def main():
    args = parse_args()

    # Argument validation
    if args.max_reviews <= 1:
        raise ValueError('max_reviews must be greater than 1')
    if args.top_n < 1:
        raise ValueError('top_n must be greater than 0')
    if args.most_common < 1:
        raise ValueError('most_common must be greater than 0')
    if args.reviews_file and not args.reviews_file.endswith('.json'):
        raise ValueError('reviews_file must end with .json')
    if args.log_file and not (args.log_file.endswith('.log') or args.log_file.endswith('.txt')):
        raise ValueError('log_file must end with .log or .txt')

    log_file = args.log_file or f'app_{args.product_id}.log'
    review_file = args.reviews_file or f'app_{args.product_id}_review.json'
    clusters_file = args.clusters_file or f'app_{args.product_id}_clusters.txt'

    logger = setup_logger(args.product_id, log_file)

    # Load or fetch reviews
    reviews = []
    if args.rewrite_review_file == 0:
        reviews = load_reviews_from_file(review_file, logger)
    if not reviews:
        api_key = load_api_key(API_KEY_FILE, logger)
        reviews = fetch_reviews(
            api_key,
            args.product_id,
            country=COUNTRY,
            sort=SORT,
            max_reviews=args.max_reviews,
            delay=1.0,
            logger=logger
        )
        save_reviews_to_file(reviews, review_file, logger)

    reviews = reviews[:args.max_reviews]

    # Process ratings by stars
    process_ratings(reviews, logger)

    # Prepare texts
    titles = [r.get('title', '') for r in reviews]
    texts = [
        f"{title}. {text}" if title else text
        for title, text in zip(titles, [r.get('text', '') for r in reviews])
    ]

    # Sentiment analysis
    sentiments = analyze_sentiment(texts, logger)

    neg_texts = [txt for txt, lab in zip(texts, sentiments) if lab == NEGATIVE_KEY]
    logger.info(f'Found {len(neg_texts)} negative reviews.')

    # Keyword extraction
    keywords_kb = extract_keywords(neg_texts, method='keybert', top_n=args.top_n, logger=logger)
    keywords_yk = extract_keywords(neg_texts, method='yake', top_n=args.top_n, logger=logger)

    common_kb = Counter(keywords_kb).most_common(args.most_common)
    common_yk = Counter(keywords_yk).most_common(args.most_common)

    logger.info('Top keywords (KeyBERT):')
    for word, cnt in common_kb:
        logger.info(f'  {word}: {cnt}')

    logger.info('Top keywords (YAKE):')
    for word, cnt in common_yk:
        logger.info(f'  {word}: {cnt}')

    # Clustering flow
    if args.clustering == 1:
        summaries = summarize_reviews(neg_texts, logger=logger)
        clusters = cluster_summaries(summaries, n_clusters=args.n_clusters, logger=logger)

        # Log examples to main log
        for cid, items in clusters.items():
            logger.info(f"--- Cluster {cid} ({len(items)} items) ---")
            examples = random.sample(items, min(5, len(items)))
            for ex in examples:
                logger.info(f" • {ex}")

        # Write full clusters to file
        with open(clusters_file, 'w', encoding='utf-8') as cf:
            for cid, items in clusters.items():
                cf.write(f"--- Cluster {cid} ({len(items)} items) ---\n")
                for summ in items:
                    cf.write(f"{summ}\n")
                cf.write("\n")
        logger.info(f'Wrote full clusters to {clusters_file}')


if __name__ == '__main__':
    main()

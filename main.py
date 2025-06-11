#!/usr/bin/env python3
"""
Advanced Web Scraping & Data Analysis Tool
A comprehensive tool for web scraping, data analysis, machine learning,
and automated reporting with advanced visualization capabilities.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import asyncio
import aiohttp
import json
import csv
import re
import time
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict, field
from urllib.parse import urljoin, urlparse, parse_qs, quote
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import warnings
from collections import defaultdict, Counter, deque
import threading
from queue import Queue, PriorityQueue
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import base64
import pickle
import gzip
import random
import schedule
import socket
from http.cookiejar import CookieJar
import ssl
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import xml.etree.ElementTree as ET
from io import StringIO, BytesIO
import zipfile
import tarfile
import os
import sys
import subprocess
import multiprocessing
from functools import lru_cache, wraps
import inspect
from urllib.robotparser import RobotFileParser

# Machine Learning imports
try:
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    from sklearn.decomposition import PCA, LatentDirichletAllocation
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
    from sklearn.metrics import classification_report, mean_squared_error, r2_score, accuracy_score
    from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.svm import SVC
    from textblob import TextBlob
    import nltk
    from wordcloud import WordCloud
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. ML features will be disabled.")

# Additional data visualization
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: Plotly not available. Advanced visualizations will be disabled.")

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('web_analyzer.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Global configuration
MAX_WORKERS = 20
DEFAULT_DELAY = 1.0
DEFAULT_TIMEOUT = 30
MAX_RETRIES = 3
DEFAULT_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"

@dataclass
class ScrapingConfig:
    """Configuration for web scraping operations"""
    max_workers: int = MAX_WORKERS
    delay_between_requests: float = DEFAULT_DELAY
    timeout: int = DEFAULT_TIMEOUT
    max_retries: int = MAX_RETRIES
    user_agent: str = DEFAULT_USER_AGENT
    headers: Dict[str, str] = field(default_factory=dict)
    follow_redirects: bool = True
    verify_ssl: bool = True
    max_pages_per_domain: int = 100
    respect_robots_txt: bool = True
    max_depth: int = 3
    allowed_domains: List[str] = field(default_factory=list)
    blocked_domains: List[str] = field(default_factory=list)
    file_extensions: List[str] = field(default_factory=lambda: ['.html', '.htm', '.php', '.asp', '.aspx'])

    def __post_init__(self):
        if not self.headers:
            self.headers = {
                'User-Agent': self.user_agent,
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1'
            }

@dataclass
class WebPage:
    """Represents a scraped web page"""
    url: str
    title: str = ""
    content: str = ""
    status_code: int = 0
    response_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    links: List[str] = field(default_factory=list)
    images: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    text_content: str = ""
    word_count: int = 0
    language: str = ""
    encoding: str = ""
    headers: Dict[str, str] = field(default_factory=dict)
    cookies: Dict[str, str] = field(default_factory=dict)
    forms: List[Dict] = field(default_factory=list)
    scripts: List[str] = field(default_factory=list)
    stylesheets: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

@dataclass
class AnalysisResult:
    """Results from data analysis"""
    total_pages: int
    total_words: int
    unique_domains: int
    avg_page_size: float
    avg_response_time: float
    most_common_words: List[Tuple[str, int]]
    link_analysis: Dict[str, Any]
    content_analysis: Dict[str, Any]
    seo_analysis: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    sentiment_analysis: Dict[str, Any] = field(default_factory=dict)
    topic_modeling: Dict[str, Any] = field(default_factory=dict)
    clustering_results: Dict[str, Any] = field(default_factory=dict)

class RateLimiter:
    """Rate limiting for web requests"""

    def __init__(self, max_requests_per_second: float = 1.0):
        self.max_requests_per_second = max_requests_per_second
        self.min_interval = 1.0 / max_requests_per_second
        self.last_request_time = 0.0
        self.lock = threading.Lock()

    def wait_if_needed(self):
        """Wait if necessary to respect rate limit"""
        with self.lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time

            if time_since_last < self.min_interval:
                sleep_time = self.min_interval - time_since_last
                time.sleep(sleep_time)

            self.last_request_time = time.time()

class RobotsTxtChecker:
    """Check robots.txt compliance"""

    def __init__(self):
        self.cache = {}
        self.lock = threading.Lock()

    def can_fetch(self, url: str, user_agent: str = "*") -> bool:
        """Check if URL can be fetched according to robots.txt"""
        try:
            parsed_url = urlparse(url)
            base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"

            with self.lock:
                if base_url not in self.cache:
                    robots_url = urljoin(base_url, '/robots.txt')
                    rp = RobotFileParser()
                    rp.set_url(robots_url)
                    try:
                        rp.read()
                        self.cache[base_url] = rp
                    except:
                        # If robots.txt can't be read, assume allowed
                        return True

                rp = self.cache[base_url]
                return rp.can_fetch(user_agent, url)
        except:
            return True

class WebScraper:
    """Advanced web scraper with async support and intelligent crawling"""

    def __init__(self, config: ScrapingConfig):
        self.config = config
        self.session = self._create_session()
        self.rate_limiter = RateLimiter(1.0 / config.delay_between_requests)
        self.robots_checker = RobotsTxtChecker()
        self.visited_urls = set()
        self.failed_urls = set()
        self.url_queue = PriorityQueue()
        self.results = []
        self.lock = threading.Lock()

    def _create_session(self) -> requests.Session:
        """Create configured requests session"""
        session = requests.Session()
        session.headers.update(self.config.headers)

        # Add retry strategy
        retry_strategy = Retry(
            total=self.config.max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        return session

    def scrape_url(self, url: str, depth: int = 0) -> Optional[WebPage]:
        """Scrape a single URL"""
        try:
            # Check robots.txt if enabled
            if self.config.respect_robots_txt and not self.robots_checker.can_fetch(url, self.config.user_agent):
                logger.info(f"Skipping {url} due to robots.txt")
                return None

            # Rate limiting
            self.rate_limiter.wait_if_needed()

            # Make request
            start_time = time.time()
            response = self.session.get(
                url,
                timeout=self.config.timeout,
                verify=self.config.verify_ssl,
                allow_redirects=self.config.follow_redirects
            )
            response_time = time.time() - start_time

            if response.status_code != 200:
                logger.warning(f"HTTP {response.status_code} for {url}")
                return None

            # Parse content
            soup = BeautifulSoup(response.content, 'html.parser')

            # Extract data
            page = WebPage(
                url=url,
                title=self._extract_title(soup),
                content=response.text,
                status_code=response.status_code,
                response_time=response_time,
                links=self._extract_links(soup, url),
                images=self._extract_images(soup, url),
                metadata=self._extract_metadata(soup),
                text_content=self._extract_text(soup),
                encoding=response.encoding or 'utf-8',
                headers=dict(response.headers),
                cookies=dict(response.cookies),
                forms=self._extract_forms(soup),
                scripts=self._extract_scripts(soup),
                stylesheets=self._extract_stylesheets(soup, url)
            )

            page.word_count = len(page.text_content.split())
            page.language = self._detect_language(page.text_content)

            return page

        except Exception as e:
            logger.error(f"Error scraping {url}: {str(e)}")
            with self.lock:
                self.failed_urls.add(url)
            return None

    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract page title"""
        title_tag = soup.find('title')
        return title_tag.get_text().strip() if title_tag else ""

    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract all links from page"""
        links = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            absolute_url = urljoin(base_url, href)
            links.append(absolute_url)
        return links

    def _extract_images(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract all image URLs"""
        images = []
        for img in soup.find_all('img', src=True):
            src = img['src']
            absolute_url = urljoin(base_url, src)
            images.append(absolute_url)
        return images

    def _extract_metadata(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract metadata from page"""
        metadata = {}

        # Meta tags
        for meta in soup.find_all('meta'):
            name = meta.get('name') or meta.get('property') or meta.get('http-equiv')
            content = meta.get('content')
            if name and content:
                metadata[name] = content

        # Headings
        headings = {}
        for i in range(1, 7):
            tags = soup.find_all(f'h{i}')
            if tags:
                headings[f'h{i}'] = [tag.get_text().strip() for tag in tags]
        metadata['headings'] = headings

        return metadata

    def _extract_text(self, soup: BeautifulSoup) -> str:
        """Extract clean text content"""
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # Get text
        text = soup.get_text()

        # Clean up
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)

        return text

    def _extract_forms(self, soup: BeautifulSoup) -> List[Dict]:
        """Extract form information"""
        forms = []
        for form in soup.find_all('form'):
            form_data = {
                'action': form.get('action', ''),
                'method': form.get('method', 'get').lower(),
                'inputs': []
            }

            for input_tag in form.find_all(['input', 'textarea', 'select']):
                input_data = {
                    'type': input_tag.get('type', 'text'),
                    'name': input_tag.get('name', ''),
                    'value': input_tag.get('value', ''),
                    'required': input_tag.has_attr('required')
                }
                form_data['inputs'].append(input_data)

            forms.append(form_data)

        return forms

    def _extract_scripts(self, soup: BeautifulSoup) -> List[str]:
        """Extract script sources"""
        scripts = []
        for script in soup.find_all('script', src=True):
            scripts.append(script['src'])
        return scripts

    def _extract_stylesheets(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract stylesheet URLs"""
        stylesheets = []
        for link in soup.find_all('link', rel='stylesheet'):
            href = link.get('href')
            if href:
                absolute_url = urljoin(base_url, href)
                stylesheets.append(absolute_url)
        return stylesheets

    def _detect_language(self, text: str) -> str:
        """Detect language of text content"""
        try:
            if SKLEARN_AVAILABLE:
                blob = TextBlob(text[:1000])  # Use first 1000 characters
                return blob.detect_language()
        except:
            pass
        return "unknown"

    def crawl_website(self, start_url: str, max_pages: int = None) -> List[WebPage]:
        """Crawl entire website starting from URL"""
        if max_pages is None:
            max_pages = self.config.max_pages_per_domain

        # Initialize queue
        self.url_queue.put((0, start_url))  # (depth, url)
        self.visited_urls.clear()
        self.failed_urls.clear()
        self.results.clear()

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_url = {}

            while not self.url_queue.empty() and len(self.results) < max_pages:
                # Get next URL
                try:
                    depth, url = self.url_queue.get(timeout=1)
                except:
                    break

                if url in self.visited_urls or depth > self.config.max_depth:
                    continue

                # Check domain restrictions
                if self.config.allowed_domains:
                    domain = urlparse(url).netloc
                    if not any(allowed in domain for allowed in self.config.allowed_domains):
                        continue

                if self.config.blocked_domains:
                    domain = urlparse(url).netloc
                    if any(blocked in domain for blocked in self.config.blocked_domains):
                        continue

                # Submit scraping task
                with self.lock:
                    self.visited_urls.add(url)

                future = executor.submit(self.scrape_url, url, depth)
                future_to_url[future] = (url, depth)

                # Process completed futures
                for future in list(future_to_url.keys()):
                    if future.done():
                        url, depth = future_to_url.pop(future)
                        try:
                            page = future.result()
                            if page:
                                with self.lock:
                                    self.results.append(page)

                                # Add new URLs to queue
                                if depth < self.config.max_depth:
                                    for link in page.links:
                                        if link not in self.visited_urls:
                                            # Check file extension
                                            parsed = urlparse(link)
                                            path = parsed.path.lower()
                                            if any(path.endswith(ext) for ext in self.config.file_extensions) or not path or path.endswith('/'):
                                                self.url_queue.put((depth + 1, link))

                        except Exception as e:
                            logger.error(f"Error processing {url}: {str(e)}")

            # Wait for remaining futures
            for future in future_to_url:
                try:
                    page = future.result(timeout=30)
                    if page:
                        self.results.append(page)
                except:
                    pass

        logger.info(f"Crawled {len(self.results)} pages, {len(self.failed_urls)} failed")
        return self.results

class DataAnalyzer:
    """Advanced data analysis with machine learning capabilities"""

    def __init__(self):
        self.pages = []
        self.df = None

    def load_data(self, pages: List[WebPage]):
        """Load scraped data for analysis"""
        self.pages = pages
        self.df = pd.DataFrame([page.to_dict() for page in pages])

        # Convert timestamp
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])

        # Add derived columns
        self.df['domain'] = self.df['url'].apply(lambda x: urlparse(x).netloc)
        self.df['path_depth'] = self.df['url'].apply(lambda x: len([p for p in urlparse(x).path.split('/') if p]))
        self.df['has_forms'] = self.df['forms'].apply(lambda x: len(x) > 0)
        self.df['external_links'] = self.df.apply(self._count_external_links, axis=1)
        self.df['internal_links'] = self.df.apply(self._count_internal_links, axis=1)

    def _count_external_links(self, row) -> int:
        """Count external links on page"""
        page_domain = urlparse(row['url']).netloc
        external_count = 0
        for link in row['links']:
            link_domain = urlparse(link).netloc
            if link_domain and link_domain != page_domain:
                external_count += 1
        return external_count

    def _count_internal_links(self, row) -> int:
        """Count internal links on page"""
        page_domain = urlparse(row['url']).netloc
        internal_count = 0
        for link in row['links']:
            link_domain = urlparse(link).netloc
            if link_domain == page_domain or not link_domain:
                internal_count += 1
        return internal_count

    def basic_analysis(self) -> AnalysisResult:
        """Perform basic statistical analysis"""
        if self.df is None or self.df.empty:
            raise ValueError("No data loaded for analysis")

        # Basic statistics
        total_pages = len(self.df)
        total_words = self.df['word_count'].sum()
        unique_domains = self.df['domain'].nunique()
        avg_page_size = self.df['content'].apply(len).mean()
        avg_response_time = self.df['response_time'].mean()

        # Text analysis
        all_text = ' '.join(self.df['text_content'].fillna(''))
        words = re.findall(r'\b\w+\b', all_text.lower())
        word_freq = Counter(words)
        most_common_words = word_freq.most_common(50)

        # Link analysis
        link_analysis = {
            'total_links': self.df['links'].apply(len).sum(),
            'avg_links_per_page': self.df['links'].apply(len).mean(),
            'external_links_total': self.df['external_links'].sum(),
            'internal_links_total': self.df['internal_links'].sum(),
            'pages_with_forms': self.df['has_forms'].sum(),
            'avg_page_depth': self.df['path_depth'].mean()
        }

        # Content analysis
        content_analysis = {
            'avg_title_length': self.df['title'].apply(len).mean(),
            'pages_with_images': self.df['images'].apply(lambda x: len(x) > 0).sum(),
            'total_images': self.df['images'].apply(len).sum(),
            'pages_with_scripts': self.df['scripts'].apply(lambda x: len(x) > 0).sum(),
            'total_scripts': self.df['scripts'].apply(len).sum(),
            'languages': self.df['language'].value_counts().to_dict()
        }

        # SEO analysis
        seo_analysis = self._analyze_seo()

        # Performance metrics
        performance_metrics = {
            'avg_response_time': avg_response_time,
            'slow_pages': len(self.df[self.df['response_time'] > 3.0]),
            'fast_pages': len(self.df[self.df['response_time'] < 1.0]),
            'response_time_percentiles': {
                '50th': self.df['response_time'].quantile(0.5),
                '90th': self.df['response_time'].quantile(0.9),
                '95th': self.df['response_time'].quantile(0.95)
            }
        }

        return AnalysisResult(
            total_pages=total_pages,
            total_words=total_words,
            unique_domains=unique_domains,
            avg_page_size=avg_page_size,
            avg_response_time=avg_response_time,
            most_common_words=most_common_words,
            link_analysis=link_analysis,
            content_analysis=content_analysis,
            seo_analysis=seo_analysis,
            performance_metrics=performance_metrics
        )

    def _analyze_seo(self) -> Dict[str, Any]:
        """Analyze SEO factors"""
        seo_issues = []

        # Check for missing titles
        missing_titles = len(self.df[self.df['title'].str.len() == 0])
        if missing_titles > 0:
            seo_issues.append(f"{missing_titles} pages missing titles")

        # Check title lengths
        long_titles = len(self.df[self.df['title'].str.len() > 60])
        if long_titles > 0:
            seo_issues.append(f"{long_titles} pages with titles longer than 60 characters")

        # Check for meta descriptions
        pages_with_meta_desc = 0
        for idx, row in self.df.iterrows():
            metadata = row.get('metadata', {})
            if 'description' in metadata:
                pages_with_meta_desc += 1

        missing_meta_desc = len(self.df) - pages_with_meta_desc
        if missing_meta_desc > 0:
            seo_issues.append(f"{missing_meta_desc} pages missing meta descriptions")

        return {
            'issues': seo_issues,
            'pages_with_meta_desc': pages_with_meta_desc,
            'avg_title_length': self.df['title'].str.len().mean(),
            'duplicate_titles': len(self.df) - self.df['title'].nunique()
        }

    def sentiment_analysis(self) -> Dict[str, Any]:
        """Perform sentiment analysis on text content"""
        if not SKLEARN_AVAILABLE:
            return {"error": "TextBlob not available"}

        sentiments = []
        polarities = []

        for text in self.df['text_content'].fillna(''):
            if text.strip():
                try:
                    blob = TextBlob(text)
                    sentiment = blob.sentiment

                    if sentiment.polarity > 0.1:
                        sentiment_label = 'positive'
                    elif sentiment.polarity < -0.1:
                        sentiment_label = 'negative'
                    else:
                        sentiment_label = 'neutral'

                    sentiments.append(sentiment_label)
                    polarities.append(sentiment.polarity)
                except:
                    sentiments.append('neutral')
                    polarities.append(0.0)
            else:
                sentiments.append('neutral')
                polarities.append(0.0)

        self.df['sentiment'] = sentiments
        self.df['polarity'] = polarities

        sentiment_counts = pd.Series(sentiments).value_counts().to_dict()
        avg_polarity = np.mean(polarities)

        return {
            'sentiment_distribution': sentiment_counts,
            'average_polarity': avg_polarity,
            'most_positive_pages': self.df.nlargest(5, 'polarity')[['url', 'title', 'polarity']].to_dict('records'),
            'most_negative_pages': self.df.nsmallest(5, 'polarity')[['url', 'title', 'polarity']].to_dict('records')
        }

    def topic_modeling(self, num_topics: int = 5) -> Dict[str, Any]:
        """Perform topic modeling using LDA"""
        if not SKLEARN_AVAILABLE:
            return {"error": "scikit-learn not available"}

        # Prepare text data
        texts = [text for text in self.df['text_content'].fillna('') if text.strip()]

        if len(texts) < 2:
            return {"error": "Not enough text data for topic modeling"}

        # Vectorize text
        vectorizer = CountVectorizer(
            max_df=0.95,
            min_df=2,
            max_features=1000,
            stop_words='english'
        )

        doc_term_matrix = vectorizer.fit_transform(texts)

        # Perform LDA
        lda = LatentDirichletAllocation(
            n_components=num_topics,
            random_state=42,
            max_iter=10
        )

        lda.fit(doc_term_matrix)

        # Extract topics
        feature_names = vectorizer.get_feature_names_out()
        topics = []

        for topic_idx, topic in enumerate(lda.components_):
            top_words = [feature_names[i] for i in topic.argsort()[-10:]]
            topics.append({
                'topic_id': topic_idx,
                'words': top_words,
                'weight': topic.max()
            })

        # Assign topics to documents
        doc_topic_probs = lda.transform(doc_term_matrix)
        dominant_topics = np.argmax(doc_topic_probs, axis=1)

        topic_distribution = pd.Series(dominant_topics).value_counts().to_dict()

        return {
            'topics': topics,
            'topic_distribution': topic_distribution,
            'num_topics': num_topics,
            'perplexity': lda.perplexity(doc_term_matrix)
        }

    def clustering_analysis(self, num_clusters: int = 5) -> Dict[str, Any]:
        """Perform clustering analysis on pages"""
        if not SKLEARN_AVAILABLE:
            return {"error": "scikit-learn not available"}

        # Prepare features for clustering
        features = []
        valid_indices = []

        for idx, row in self.df.iterrows():
            text = row['text_content']
            if isinstance(text, str) and text.strip():
                features.append(text)
                valid_indices.append(idx)

        if len(features) < 2:
            return {"error": "Not enough data for clustering"}

        # Vectorize text
        vectorizer = TfidfVectorizer(
            max_df=0.95,
            min_df=2,
            max_features=1000,
            stop_words='english'
        )

        tfidf_matrix = vectorizer.fit_transform(features)

        # Perform clustering
        kmeans = KMeans(n_clusters=min(num_clusters, len(features)), random_state=42)
        cluster_labels = kmeans.fit_predict(tfidf_matrix)

        # Analyze clusters
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []

            original_idx = valid_indices[i]
            clusters[label].append({
                'url': self.df.iloc[original_idx]['url'],
                'title': self.df.iloc[original_idx]['title'],
                'word_count': self.df.iloc[original_idx]['word_count']
            })

        cluster_sizes = {str(k): len(v) for k, v in clusters.items()}

        return {
            'clusters': clusters,
            'cluster_sizes': cluster_sizes,
            'num_clusters': len(clusters),
            'silhouette_score': 0  # Would need additional calculation
        }

class Visualizer:
    """Create various visualizations for the analysis results"""

    def __init__(self, analysis_result: AnalysisResult, df: pd.DataFrame = None):
        self.analysis_result = analysis_result
        self.df = df

        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

    def create_summary_dashboard(self, output_dir: str = "visualizations"):
        """Create comprehensive dashboard"""
        Path(output_dir).mkdir(exist_ok=True)

        # Create multiple plots
        self._create_word_frequency_plot(output_dir)
        self._create_response_time_distribution(output_dir)
        self._create_page_size_distribution(output_dir)
        self._create_domain_analysis(output_dir)
        self._create_seo_analysis_plot(output_dir)

        if PLOTLY_AVAILABLE:
            self._create_interactive_dashboard(output_dir)

        logger.info(f"Visualizations saved to {output_dir}")

    def _create_word_frequency_plot(self, output_dir: str):
        """Create word frequency visualization"""
        words, counts = zip(*self.analysis_result.most_common_words[:20])

        plt.figure(figsize=(12, 8))
        bars = plt.bar(range(len(words)), counts)
        plt.xlabel('Words')
        plt.ylabel('Frequency')
        plt.title('Top 20 Most Common Words')
        plt.xticks(range(len(words)), words, rotation=45, ha='right')

        # Add value labels on bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     str(count), ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(f"{output_dir}/word_frequency.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Create word cloud if available
        try:
            if 'WordCloud' in globals():
                word_freq = dict(self.analysis_result.most_common_words)
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)

                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                plt.title('Word Cloud')
                plt.savefig(f"{output_dir}/word_cloud.png", dpi=300, bbox_inches='tight')
                plt.close()
        except:
            pass

    def _create_response_time_distribution(self, output_dir: str):
        """Create response time distribution plot"""
        if self.df is None:
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Histogram
        ax1.hist(self.df['response_time'], bins=30, edgecolor='black', alpha=0.7)
        ax1.set_xlabel('Response Time (seconds)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Response Time Distribution')
        ax1.axvline(self.df['response_time'].mean(), color='red', linestyle='--', label=f'Mean: {self.df["response_time"].mean():.2f}s')
        ax1.legend()

        # Box plot
        ax2.boxplot(self.df['response_time'])
        ax2.set_ylabel('Response Time (seconds)')
        ax2.set_title('Response Time Box Plot')

        plt.tight_layout()
        plt.savefig(f"{output_dir}/response_time_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _create_page_size_distribution(self, output_dir: str):
        """Create page size distribution plot"""
        if self.df is None:
            return

        page_sizes = self.df['content'].apply(len) / 1024  # Convert to KB

        plt.figure(figsize=(12, 6))
        plt.hist(page_sizes, bins=30, edgecolor='black', alpha=0.7)
        plt.xlabel('Page Size (KB)')
        plt.ylabel('Frequency')
        plt.title('Page Size Distribution')
        plt.axvline(page_sizes.mean(), color='red', linestyle='--', label=f'Mean: {page_sizes.mean():.1f} KB')
        plt.legend()

        plt.tight_layout()
        plt.savefig(f"{output_dir}/page_size_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _create_domain_analysis(self, output_dir: str):
        """Create domain analysis visualization"""
        if self.df is None:
            return

        domain_counts = self.df['domain'].value_counts().head(10)

        plt.figure(figsize=(12, 8))
        bars = plt.bar(range(len(domain_counts)), domain_counts.values)
        plt.xlabel('Domains')
        plt.ylabel('Number of Pages')
        plt.title('Top 10 Domains by Page Count')
        plt.xticks(range(len(domain_counts)), domain_counts.index, rotation=45, ha='right')

        # Add value labels
        for bar, count in zip(bars, domain_counts.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     str(count), ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(f"{output_dir}/domain_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _create_seo_analysis_plot(self, output_dir: str):
        """Create SEO analysis visualization"""
        seo_data = self.analysis_result.seo_analysis

        # Create metrics
        metrics = [
            'Pages with Meta Desc',
            'Pages without Meta Desc',
            'Duplicate Titles',
            'Unique Titles'
        ]

        values = [
            seo_data.get('pages_with_meta_desc', 0),
            self.analysis_result.total_pages - seo_data.get('pages_with_meta_desc', 0),
            seo_data.get('duplicate_titles', 0),
            self.analysis_result.total_pages - seo_data.get('duplicate_titles', 0)
        ]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Meta description pie chart
        meta_labels = ['With Meta Desc', 'Without Meta Desc']
        meta_values = values[:2]
        ax1.pie(meta_values, labels=meta_labels, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Meta Description Coverage')

        # Title uniqueness pie chart
        title_labels = ['Unique Titles', 'Duplicate Titles']
        title_values = [values[3], values[2]]
        ax2.pie(title_values, labels=title_labels, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Title Uniqueness')

        plt.tight_layout()
        plt.savefig(f"{output_dir}/seo_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _create_interactive_dashboard(self, output_dir: str):
        """Create interactive dashboard using Plotly"""
        if not PLOTLY_AVAILABLE or self.df is None:
            return

        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Word Frequency', 'Response Time vs Page Size',
                            'Pages by Domain', 'Performance Metrics'),
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "indicator"}]]
        )

        # Word frequency
        words, counts = zip(*self.analysis_result.most_common_words[:15])
        fig.add_trace(
            go.Bar(x=list(words), y=list(counts), name="Word Frequency"),
            row=1, col=1
        )

        # Response time vs page size scatter
        page_sizes = self.df['content'].apply(len) / 1024
        fig.add_trace(
            go.Scatter(
                x=page_sizes,
                y=self.df['response_time'],
                mode='markers',
                name='Pages',
                text=self.df['title'],
                hovertemplate='<b>%{text}</b><br>Size: %{x:.1f} KB<br>Response: %{y:.2f}s'
            ),
            row=1, col=2
        )

        # Domain distribution
        domain_counts = self.df['domain'].value_counts().head(10)
        fig.add_trace(
            go.Bar(x=domain_counts.index, y=domain_counts.values, name="Pages by Domain"),
            row=2, col=1
        )

        # Performance indicator
        avg_response_time = self.df['response_time'].mean()
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=avg_response_time,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Avg Response Time (s)"},
                gauge={
                    'axis': {'range': [None, 5]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 1], 'color': "lightgreen"},
                        {'range': [1, 3], 'color': "yellow"},
                        {'range': [3, 5], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 3
                    }
                }
            ),
            row=2, col=2
        )

        # Update layout
        fig.update_layout(
            height=800,
            title_text="Web Scraping Analysis Dashboard",
            showlegend=False
        )

        # Save interactive HTML
        fig.write_html(f"{output_dir}/interactive_dashboard.html")

class DatabaseManager:
    """Manage SQLite database for storing scraped data"""

    def __init__(self, db_path: str = "web_scraping.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize database tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Create pages table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS pages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    url TEXT UNIQUE,
                    title TEXT,
                    content TEXT,
                    status_code INTEGER,
                    response_time REAL,
                    timestamp TEXT,
                    word_count INTEGER,
                    language TEXT,
                    domain TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Create links table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS links (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_url TEXT,
                    target_url TEXT,
                    link_text TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (source_url) REFERENCES pages (url)
                )
            ''')

            # Create metadata table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    url TEXT,
                    meta_key TEXT,
                    meta_value TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (url) REFERENCES pages (url)
                )
            ''')

            # Create indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_pages_url ON pages (url)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_pages_domain ON pages (domain)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_links_source ON links (source_url)')

            conn.commit()

    def save_page(self, page: WebPage):
        """Save page to database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Insert page
            cursor.execute('''
                INSERT OR REPLACE INTO pages 
                (url, title, content, status_code, response_time, timestamp, 
                 word_count, language, domain)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                page.url, page.title, page.content, page.status_code,
                page.response_time, page.timestamp.isoformat(),
                page.word_count, page.language, urlparse(page.url).netloc
            ))

            # Insert links
            for link in page.links:
                cursor.execute('''
                    INSERT OR IGNORE INTO links (source_url, target_url)
                    VALUES (?, ?)
                ''', (page.url, link))

            # Insert metadata
            for key, value in page.metadata.items():
                if isinstance(value, (str, int, float)):
                    cursor.execute('''
                        INSERT OR REPLACE INTO metadata (url, meta_key, meta_value)
                        VALUES (?, ?, ?)
                    ''', (page.url, key, str(value)))

            conn.commit()

    def save_pages(self, pages: List[WebPage]):
        """Save multiple pages to database"""
        for page in pages:
            self.save_page(page)

    def load_pages(self, domain: str = None, limit: int = None) -> List[WebPage]:
        """Load pages from database"""
        with sqlite3.connect(self.db_path) as conn:
            query = "SELECT * FROM pages"
            params = []

            if domain:
                query += " WHERE domain = ?"
                params.append(domain)

            if limit:
                query += " LIMIT ?"
                params.append(limit)

            df = pd.read_sql_query(query, conn, params=params)

            pages = []
            for _, row in df.iterrows():
                page = WebPage(
                    url=row['url'],
                    title=row['title'],
                    content=row['content'],
                    status_code=row['status_code'],
                    response_time=row['response_time'],
                    word_count=row['word_count'],
                    language=row['language']
                )
                page.timestamp = datetime.fromisoformat(row['timestamp'])
                pages.append(page)

            return pages

    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Basic counts
            cursor.execute("SELECT COUNT(*) FROM pages")
            total_pages = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(DISTINCT domain) FROM pages")
            unique_domains = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM links")
            total_links = cursor.fetchone()[0]

            # Domain distribution
            cursor.execute("""
                SELECT domain, COUNT(*) as count 
                FROM pages 
                GROUP BY domain 
                ORDER BY count DESC 
                LIMIT 10
            """)
            top_domains = dict(cursor.fetchall())

            return {
                'total_pages': total_pages,
                'unique_domains': unique_domains,
                'total_links': total_links,
                'top_domains': top_domains
            }

class ReportGenerator:
    """Generate comprehensive reports in various formats"""

    def __init__(self, analysis_result: AnalysisResult, pages: List[WebPage] = None):
        self.analysis_result = analysis_result
        self.pages = pages or []

    def generate_html_report(self, output_path: str = "report.html"):
        """Generate comprehensive HTML report"""
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Web Scraping Analysis Report</title>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 20px rgba(0,0,0,0.1); }}
                h1, h2, h3 {{ color: #333; }}
                h1 {{ text-align: center; color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
                .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }}
                .summary-card {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; text-align: center; }}
                .summary-card h3 {{ margin: 0; font-size: 2em; }}
                .summary-card p {{ margin: 5px 0 0 0; opacity: 0.9; }}
                .section {{ margin: 30px 0; padding: 20px; background: #f8f9fa; border-radius: 8px; border-left: 4px solid #3498db; }}
                table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #3498db; color: white; }}
                tr:hover {{ background-color: #f5f5f5; }}
                .issue {{ background-color: #ffe6e6; padding: 10px; border-left: 4px solid #e74c3c; margin: 10px 0; border-radius: 4px; }}
                .recommendation {{ background-color: #e6f7ff; padding: 10px; border-left: 4px solid #3498db; margin: 10px 0; border-radius: 4px; }}
                .chart-placeholder {{ background: #ecf0f1; height: 200px; display: flex; align-items: center; justify-content: center; border-radius: 8px; margin: 15px 0; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🌐 Web Scraping Analysis Report</h1>
                <p style="text-align: center; color: #666; font-size: 1.1em;">Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <div class="summary">
                    <div class="summary-card">
                        <h3>{self.analysis_result.total_pages:,}</h3>
                        <p>Total Pages Scraped</p>
                    </div>
                    <div class="summary-card">
                        <h3>{self.analysis_result.unique_domains:,}</h3>
                        <p>Unique Domains</p>
                    </div>
                    <div class="summary-card">
                        <h3>{self.analysis_result.total_words:,}</h3>
                        <p>Total Words</p>
                    </div>
                    <div class="summary-card">
                        <h3>{self.analysis_result.avg_response_time:.2f}s</h3>
                        <p>Avg Response Time</p>
                    </div>
                </div>
                
                <div class="section">
                    <h2>📊 Content Analysis</h2>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                        <div>
                            <h3>Link Analysis</h3>
                            <p><strong>Total Links:</strong> {self.analysis_result.link_analysis.get('total_links', 0):,}</p>
                            <p><strong>Avg Links per Page:</strong> {self.analysis_result.link_analysis.get('avg_links_per_page', 0):.1f}</p>
                            <p><strong>External Links:</strong> {self.analysis_result.link_analysis.get('external_links_total', 0):,}</p>
                            <p><strong>Internal Links:</strong> {self.analysis_result.link_analysis.get('internal_links_total', 0):,}</p>
                        </div>
                        <div>
                            <h3>Media Content</h3>
                            <p><strong>Pages with Images:</strong> {self.analysis_result.content_analysis.get('pages_with_images', 0):,}</p>
                            <p><strong>Total Images:</strong> {self.analysis_result.content_analysis.get('total_images', 0):,}</p>
                            <p><strong>Pages with Scripts:</strong> {self.analysis_result.content_analysis.get('pages_with_scripts', 0):,}</p>
                            <p><strong>Total Scripts:</strong> {self.analysis_result.content_analysis.get('total_scripts', 0):,}</p>
                        </div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>🔍 SEO Analysis</h2>
                    <div>
                        <p><strong>Pages with Meta Descriptions:</strong> {self.analysis_result.seo_analysis.get('pages_with_meta_desc', 0):,}</p>
                        <p><strong>Average Title Length:</strong> {self.analysis_result.seo_analysis.get('avg_title_length', 0):.1f} characters</p>
                        <p><strong>Duplicate Titles:</strong> {self.analysis_result.seo_analysis.get('duplicate_titles', 0):,}</p>
                    </div>
                    
                    {''.join([f'<div class="issue">⚠️ {issue}</div>' for issue in self.analysis_result.seo_analysis.get('issues', [])])}
                </div>
                
                <div class="section">
                    <h2>⚡ Performance Metrics</h2>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;">
                        <div>
                            <h4>Response Time Percentiles</h4>
                            <p><strong>50th:</strong> {self.analysis_result.performance_metrics.get('response_time_percentiles', {}).get('50th', 0):.2f}s</p>
                            <p><strong>90th:</strong> {self.analysis_result.performance_metrics.get('response_time_percentiles', {}).get('90th', 0):.2f}s</p>
                            <p><strong>95th:</strong> {self.analysis_result.performance_metrics.get('response_time_percentiles', {}).get('95th', 0):.2f}s</p>
                        </div>
                        <div>
                            <h4>Page Speed</h4>
                            <p><strong>Fast Pages (&lt;1s):</strong> {self.analysis_result.performance_metrics.get('fast_pages', 0):,}</p>
                            <p><strong>Slow Pages (&gt;3s):</strong> {self.analysis_result.performance_metrics.get('slow_pages', 0):,}</p>
                        </div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>📈 Most Common Words</h2>
                    <table>
                        <thead>
                            <tr><th>Word</th><th>Frequency</th><th>Percentage</th></tr>
                        </thead>
                        <tbody>
                            {''.join([f'<tr><td>{word}</td><td>{freq:,}</td><td>{(freq/self.analysis_result.total_words*100):.2f}%</td></tr>' for word, freq in self.analysis_result.most_common_words[:20]])}
                        </tbody>
                    </table>
                </div>
                
                <div class="section">
                    <h2>🌍 Language Distribution</h2>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px;">
                        {''.join([f'<div style="background: #ecf0f1; padding: 10px; border-radius: 5px; text-align: center;"><strong>{lang}</strong><br>{count} pages</div>' for lang, count in self.analysis_result.content_analysis.get('languages', {}).items()])}
                    </div>
                </div>
                
                <div class="section">
                    <h2>💡 Recommendations</h2>
                    <div class="recommendation">
                        <h4>Performance Optimization</h4>
                        <ul>
                            <li>Optimize {self.analysis_result.performance_metrics.get('slow_pages', 0)} slow-loading pages (&gt;3s response time)</li>
                            <li>Consider implementing caching for frequently accessed content</li>
                            <li>Monitor and optimize database queries for better performance</li>
                        </ul>
                    </div>
                    
                    <div class="recommendation">
                        <h4>SEO Improvements</h4>
                        <ul>
                            <li>Add meta descriptions to pages missing them</li>
                            <li>Review and optimize page titles (recommended length: 50-60 characters)</li>
                            <li>Fix duplicate titles across {self.analysis_result.seo_analysis.get('duplicate_titles', 0)} pages</li>
                        </ul>
                    </div>
                    
                    <div class="recommendation">
                        <h4>Content Strategy</h4>
                        <ul>
                            <li>Focus on content around top keywords: {', '.join([word for word, _ in self.analysis_result.most_common_words[:5]])}</li>
                            <li>Ensure consistent internal linking structure</li>
                            <li>Consider adding more visual content to pages with low image counts</li>
                        </ul>
                    </div>
                </div>
                
                <div style="text-align: center; margin-top: 40px; padding: 20px; background: #2c3e50; color: white; border-radius: 8px;">
                    <p>Report generated by Advanced Web Scraping & Analysis Tool</p>
                    <p style="opacity: 0.8; font-size: 0.9em;">For more detailed analysis, check the generated visualizations and CSV exports</p>
                </div>
            </div>
        </body>
        </html>
        """

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        logger.info(f"HTML report saved to {output_path}")

    def generate_csv_export(self, output_dir: str = "exports"):
        """Export data to CSV files"""
        Path(output_dir).mkdir(exist_ok=True)

        if not self.pages:
            logger.warning("No pages data available for CSV export")
            return

        # Convert pages to DataFrame
        df = pd.DataFrame([page.to_dict() for page in self.pages])

        # Main pages export
        main_columns = ['url', 'title', 'status_code', 'response_time', 'word_count', 'language', 'timestamp']
        df[main_columns].to_csv(f"{output_dir}/pages_summary.csv", index=False)

        # Links export
        links_data = []
        for page in self.pages:
            for link in page.links:
                links_data.append({
                    'source_url': page.url,
                    'target_url': link,
                    'source_title': page.title
                })

        if links_data:
            pd.DataFrame(links_data).to_csv(f"{output_dir}/links.csv", index=False)

        # Images export
        images_data = []
        for page in self.pages:
            for image in page.images:
                images_data.append({
                    'page_url': page.url,
                    'image_url': image,
                    'page_title': page.title
                })

        if images_data:
            pd.DataFrame(images_data).to_csv(f"{output_dir}/images.csv", index=False)

        # Word frequency export
        word_freq_df = pd.DataFrame(
            self.analysis_result.most_common_words,
            columns=['word', 'frequency']
        )
        word_freq_df.to_csv(f"{output_dir}/word_frequency.csv", index=False)

        logger.info(f"CSV exports saved to {output_dir}")

class WebScrapingTool:
    """Main application class"""

    def __init__(self):
        self.config = ScrapingConfig()
        self.scraper = None
        self.analyzer = DataAnalyzer()
        self.db = DatabaseManager()
        self.pages = []
        self.analysis_result = None

    def run_interactive_mode(self):
        """Run interactive command-line interface"""
        print("🌐 Advanced Web Scraping & Data Analysis Tool")
        print("=" * 50)

        while True:
            print("\nChoose an option:")
            print("1. 🔍 Scrape single URL")
            print("2. 🕷️  Crawl entire website")
            print("3. 📊 Analyze scraped data")
            print("4. 📈 Generate visualizations")
            print("5. 📋 Generate reports")
            print("6. 💾 Database operations")
            print("7. ⚙️  Configure settings")
            print("8. 🧪 Run ML analysis")
            print("9. 📤 Export data")
            print("0. ❌ Exit")

            choice = input("\nEnter your choice (0-9): ").strip()

            try:
                if choice == '1':
                    self._scrape_single_url()
                elif choice == '2':
                    self._crawl_website()
                elif choice == '3':
                    self._analyze_data()
                elif choice == '4':
                    self._generate_visualizations()
                elif choice == '5':
                    self._generate_reports()
                elif choice == '6':
                    self._database_operations()
                elif choice == '7':
                    self._configure_settings()
                elif choice == '8':
                    self._run_ml_analysis()
                elif choice == '9':
                    self._export_data()
                elif choice == '0':
                    print("👋 Goodbye!")
                    break
                else:
                    print("❌ Invalid choice. Please try again.")

            except KeyboardInterrupt:
                print("\n⏸️  Operation interrupted by user")
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                logger.error(f"Error in interactive mode: {str(e)}")

    def _scrape_single_url(self):
        """Scrape a single URL"""
        url = input("Enter URL to scrape: ").strip()

        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url

        print(f"🔍 Scraping {url}...")

        self.scraper = WebScraper(self.config)
        page = self.scraper.scrape_url(url)

        if page:
            self.pages = [page]
            self.db.save_page(page)

            print("✅ Scraping completed!")
            print(f"📄 Title: {page.title}")
            print(f"📊 Words: {page.word_count:,}")
            print(f"🔗 Links: {len(page.links)}")
            print(f"🖼️  Images: {len(page.images)}")
            print(f"⏱️  Response time: {page.response_time:.2f}s")
        else:
            print("❌ Failed to scrape URL")

    def _crawl_website(self):
        """Crawl entire website"""
        start_url = input("Enter starting URL: ").strip()

        if not start_url.startswith(('http://', 'https://')):
            start_url = 'https://' + start_url

        max_pages = int(input(f"Maximum pages to crawl (default {self.config.max_pages_per_domain}): ") or self.config.max_pages_per_domain)
        max_depth = int(input(f"Maximum crawl depth (default {self.config.max_depth}): ") or self.config.max_depth)

        self.config.max_pages_per_domain = max_pages
        self.config.max_depth = max_depth

        print(f"🕷️  Crawling {start_url} (max {max_pages} pages, depth {max_depth})...")

        self.scraper = WebScraper(self.config)

        start_time = time.time()
        self.pages = self.scraper.crawl_website(start_url, max_pages)
        crawl_time = time.time() - start_time

        if self.pages:
            # Save to database
            self.db.save_pages(self.pages)

            print("✅ Crawling completed!")
            print(f"📄 Pages scraped: {len(self.pages)}")
            print(f"⏱️  Total time: {crawl_time:.1f} seconds")
            print(f"🔗 Total links found: {sum(len(page.links) for page in self.pages):,}")
            print(f"📊 Total words: {sum(page.word_count for page in self.pages):,}")
        else:
            print("❌ No pages were successfully scraped")

    def _analyze_data(self):
        """Analyze scraped data"""
        if not self.pages:
            # Try to load from database
            self.pages = self.db.load_pages(limit=1000)

        if not self.pages:
            print("❌ No data available for analysis. Please scrape some websites first.")
            return

        print("📊 Analyzing data...")

        self.analyzer.load_data(self.pages)
        self.analysis_result = self.analyzer.basic_analysis()

        print("✅ Analysis completed!")
        print(f"📄 Total pages: {self.analysis_result.total_pages:,}")
        print(f"🌍 Unique domains: {self.analysis_result.unique_domains}")
        print(f"📊 Total words: {self.analysis_result.total_words:,}")
        print(f"⏱️  Avg response time: {self.analysis_result.avg_response_time:.2f}s")
        print(f"🔗 Total links: {self.analysis_result.link_analysis['total_links']:,}")

        # Show top words
        print("\n🔤 Top 10 most common words:")
        for word, freq in self.analysis_result.most_common_words[:10]:
            print(f"  {word}: {freq:,}")

    def _generate_visualizations(self):
        """Generate visualizations"""
        if not self.analysis_result:
            print("❌ No analysis data available. Please run analysis first.")
            return

        print("📈 Generating visualizations...")

        # Create DataFrame for visualizations
        df = pd.DataFrame([page.to_dict() for page in self.pages])
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        visualizer = Visualizer(self.analysis_result, df)
        visualizer.create_summary_dashboard()

        print("✅ Visualizations generated and saved to 'visualizations' folder!")

    def _generate_reports(self):
        """Generate comprehensive reports"""
        if not self.analysis_result:
            print("❌ No analysis data available. Please run analysis first.")
            return

        print("📋 Generating reports...")

        report_generator = ReportGenerator(self.analysis_result, self.pages)

        # Generate HTML report
        report_generator.generate_html_report("comprehensive_report.html")

        print("✅ Reports generated!")
        print("📄 HTML report: comprehensive_report.html")

    def _database_operations(self):
        """Database operations menu"""
        print("\n💾 Database Operations:")
        print("1. View statistics")
        print("2. Load data from database")
        print("3. Clear database")
        print("4. Export database to CSV")

        choice = input("Choose option (1-4): ").strip()

        if choice == '1':
            stats = self.db.get_statistics()
            print("\n📊 Database Statistics:")
            print(f"Total pages: {stats['total_pages']:,}")
            print(f"Unique domains: {stats['unique_domains']}")
            print(f"Total links: {stats['total_links']:,}")
            print("\nTop domains:")
            for domain, count in list(stats['top_domains'].items())[:10]:
                print(f"  {domain}: {count} pages")

        elif choice == '2':
            domain = input("Enter domain filter (optional): ").strip() or None
            limit = int(input("Enter limit (default 1000): ") or 1000)

            self.pages = self.db.load_pages(domain, limit)
            print(f"✅ Loaded {len(self.pages)} pages from database")

        elif choice == '3':
            confirm = input("⚠️  Are you sure you want to clear the database? (yes/no): ")
            if confirm.lower() == 'yes':
                os.remove(self.db.db_path)
                self.db.init_database()
                print("✅ Database cleared")

        elif choice == '4':
            # Export from database
            pages = self.db.load_pages()
            if pages:
                report_generator = ReportGenerator(None, pages)
                report_generator.generate_csv_export()
                print("✅ Database exported to CSV files")
            else:
                print("❌ No data in database to export")

    def _configure_settings(self):
        """Configure scraping settings"""
        print("\n⚙️  Current Configuration:")
        print(f"Max workers: {self.config.max_workers}")
        print(f"Delay between requests: {self.config.delay_between_requests}s")
        print(f"Timeout: {self.config.timeout}s")
        print(f"Max retries: {self.config.max_retries}")
        print(f"Respect robots.txt: {self.config.respect_robots_txt}")
        print(f"Max pages per domain: {self.config.max_pages_per_domain}")
        print(f"Max crawl depth: {self.config.max_depth}")

        if input("\nModify settings? (y/n): ").lower() == 'y':
            self.config.max_workers = int(input(f"Max workers ({self.config.max_workers}): ") or self.config.max_workers)
            self.config.delay_between_requests = float(input(f"Delay between requests ({self.config.delay_between_requests}): ") or self.config.delay_between_requests)
            self.config.timeout = int(input(f"Timeout ({self.config.timeout}): ") or self.config.timeout)
            self.config.max_retries = int(input(f"Max retries ({self.config.max_retries}): ") or self.config.max_retries)
            self.config.max_pages_per_domain = int(input(f"Max pages per domain ({self.config.max_pages_per_domain}): ") or self.config.max_pages_per_domain)
            self.config.max_depth = int(input(f"Max crawl depth ({self.config.max_depth}): ") or self.config.max_depth)

            print("✅ Configuration updated!")

    def _run_ml_analysis(self):
        """Run machine learning analysis"""
        if not SKLEARN_AVAILABLE:
            print("❌ Machine learning features not available. Please install scikit-learn and textblob.")
            return

        if not self.analysis_result:
            print("❌ No analysis data available. Please run basic analysis first.")
            return

        print("🧪 Running ML analysis...")

        # Sentiment analysis
        print("📊 Performing sentiment analysis...")
        sentiment_results = self.analyzer.sentiment_analysis()
        if 'error' not in sentiment_results:
            print("✅ Sentiment analysis completed!")
            print(f"Sentiment distribution: {sentiment_results['sentiment_distribution']}")
            print(f"Average polarity: {sentiment_results['average_polarity']:.3f}")

        # Topic modeling
        print("📚 Performing topic modeling...")
        topic_results = self.analyzer.topic_modeling(num_topics=5)
        if 'error' not in topic_results:
            print("✅ Topic modeling completed!")
            print("Topics found:")
            for topic in topic_results['topics']:
                print(f"  Topic {topic['topic_id']}: {', '.join(topic['words'][:5])}")

        # Clustering
        print("🎯 Performing clustering analysis...")
        cluster_results = self.analyzer.clustering_analysis(num_clusters=5)
        if 'error' not in cluster_results:
            print("✅ Clustering completed!")
            print(f"Found {cluster_results['num_clusters']} clusters")
            print(f"Cluster sizes: {cluster_results['cluster_sizes']}")

    def _export_data(self):
        """Export data in various formats"""
        if not self.pages:
            print("❌ No data available for export. Please scrape some websites first.")
            return

        print("📤 Exporting data...")

        if self.analysis_result:
            report_generator = ReportGenerator(self.analysis_result, self.pages)
            report_generator.generate_csv_export()

        # Additional exports
        output_dir = Path("exports")
        output_dir.mkdir(exist_ok=True)

        # JSON export
        json_data = [page.to_dict() for page in self.pages]
        with open(output_dir / "pages_data.json", 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        # Pickle export for Python objects
        with open(output_dir / "pages_data.pkl", 'wb') as f:
            pickle.dump(self.pages, f)

        print("✅ Data exported to 'exports' folder!")
        print("📄 Available formats: CSV, JSON, Pickle")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Advanced Web Scraping & Data Analysis Tool")
    parser.add_argument('--url', help='URL to scrape')
    parser.add_argument('--crawl', help='Starting URL for crawling')
    parser.add_argument('--max-pages', type=int, default=50, help='Maximum pages to crawl')
    parser.add_argument('--output', default='results', help='Output directory')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')

    args = parser.parse_args()

    tool = WebScrapingTool()

    if args.interactive or (not args.url and not args.crawl):
        # Run interactive mode
        tool.run_interactive_mode()
    else:
        # Command line mode
        if args.url:
            print(f"🔍 Scraping {args.url}...")
            tool.scraper = WebScraper(tool.config)
            page = tool.scraper.scrape_url(args.url)
            if page:
                tool.pages = [page]
                tool.db.save_page(page)
                print("✅ Scraping completed!")

        elif args.crawl:
            print(f"🕷️  Crawling {args.crawl}...")
            tool.scraper = WebScraper(tool.config)
            tool.pages = tool.scraper.crawl_website(args.crawl, args.max_pages)
            if tool.pages:
                tool.db.save_pages(tool.pages)
                print("✅ Crawling completed!")

        # Perform analysis
        if tool.pages:
            print("📊 Analyzing data...")
            tool.analyzer.load_data(tool.pages)
            tool.analysis_result = tool.analyzer.basic_analysis()

            # Generate reports
            output_dir = Path(args.output)
            output_dir.mkdir(exist_ok=True)

            report_generator = ReportGenerator(tool.analysis_result, tool.pages)
            report_generator.generate_html_report(output_dir / "report.html")
            report_generator.generate_csv_export(output_dir)

            # Generate visualizations
            df = pd.DataFrame([page.to_dict() for page in tool.pages])
            visualizer = Visualizer(tool.analysis_result, df)
            visualizer.create_summary_dashboard(str(output_dir))

            print(f"✅ Analysis completed! Results saved to {output_dir}")

if __name__ == "__main__":
    main()
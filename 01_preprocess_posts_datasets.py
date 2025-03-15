import json
import pandas as pd
import re
import unicodedata
import os
import datetime
from urllib.parse import urlparse
from nltk.stem.snowball import SnowballStemmer

# Initialize Albanian-compatible stemmer (fallback to English)
stemmer = SnowballStemmer("english")

INITIAL_POSTS_DATASETS = [
    "scraped_datasets/fb_posts/KQZ_POSTS_INITIAL_DATASET.json",
    "scraped_datasets/fb_posts/DEBAT_PLUS_POSTS_INITIAL_DATASET.json",
    "scraped_datasets/fb_posts/KLAN_KOSOVA_POSTS_INITIAL_DATASET.json",
    "scraped_datasets/fb_posts/NACIONALE_POSTS_INITIAL_DATASET.json",
    "scraped_datasets/fb_posts/INDEKS_ONLINE_POSTS_INITIAL_DATASET.json",
    "scraped_datasets/fb_posts/KANAL_10_POSTS_INITIAL_DATASET.json"
]

OUTPUT_FILES = [
    "scraped_datasets/fb_posts/KQZ_POSTS_PREPROCESSED_DATASET.csv",
    "scraped_datasets/fb_posts/DEBAT_PLUS_POSTS_PREPROCESSED_DATASET.csv",
    "scraped_datasets/fb_posts/KLAN_KOSOVA_POSTS_PREPROCESSED_DATASET.csv",
    "scraped_datasets/fb_posts/NACIONALE_POSTS_PREPROCESSED_DATASET.csv",
    "scraped_datasets/fb_posts/INDEKS_ONLINE_POSTS_PREPROCESSED_DATASET.csv",
    "scraped_datasets/fb_posts/KANAL_10_POSTS_PREPROCESSED_DATASET.csv"
]

def extract_text_from_url(url):
    """Extracts meaningful text from the URL slug."""
    if not url:
        return ""
    parsed_url = urlparse(url)
    slug = parsed_url.path.strip("/").split("/")[-1]
    clean_text = re.sub(r"[-_]", " ", slug)
    clean_text = re.sub(r"\.html?$", "", clean_text)
    return clean_text.strip()

def normalize_text(text):
    """Lowercases and removes accents (ë → e)."""
    if pd.isna(text):
        return ""
    text = text.lower()
    text = ''.join(c for c in unicodedata.normalize('NFKD', text) if not unicodedata.combining(c))
    return text

def stem_word(word):
    """Returns the stemmed root of a word."""
    return stemmer.stem(word)

def preprocess():
    """Processes all datasets and saves the preprocessed files."""
    for json_file, output_csv in zip(INITIAL_POSTS_DATASETS, OUTPUT_FILES):
        print(f"Processing {json_file}...")
        
        with open(json_file, "r", encoding="utf-8") as file:
            data = json.load(file)
        
        extracted_data = []
        empty_text_filled_count = 0
        empty_text_removed_count = 0
        
        for post in data:
            text = post.get("text", "").strip()
            link = post.get("link", "").strip()
            
            if not text and link:
                text = extract_text_from_url(link)
                if text:
                    empty_text_filled_count += 1
                else:
                    empty_text_removed_count += 1
            
            if text:
                extracted_data.append({
                    "url": post.get("url"),
                    "time": post.get("time"),
                    "text": text,
                    "link": link,
                    "likes": post.get("likes", 0),
                    "comments": post.get("comments", 0),
                    "shares": post.get("shares", 0)
                })
        
        df = pd.DataFrame(extracted_data)
        df["text"] = df["text"].astype(str).apply(normalize_text)
        
        base_keywords = [
            "zgjedhje", "vota", "kqz", "komision", "kryeminister", "pdk", "ldk", "vv", "vetevendosje", "aak",
            "nisma", "9 shkurt", "numerim", "preleminare", "debat", "kandidat", "opozit", "koalicion",
            "elektorat", "parti", "qeveri", "mandat", "kuvend", "deputet", "parlament", "fushat", "kurti",
            "abdixhiku", "haradinaj", "hamza"
        ]
        
        if "DEBAT_PLUS" in json_file:
            base_keywords.remove("debat")

        keyword_patterns = [stem_word(word) for word in base_keywords]
        keyword_pattern = r'\b(' + '|'.join(keyword_patterns) + r')\w*\b'
        
        df = df[df["text"].str.contains(keyword_pattern, regex=True, na=False)]
        
        zero_comments_count = (df["comments"] == 0).sum()
        df = df[df["comments"] > 0]
        
        df.to_csv(output_csv, index=False, encoding="utf-8")
        
        print(f"Dataset saved to: {output_csv}")
        print(f"Removed {zero_comments_count} records with 0 comments.")
        print(f"Entries where text was empty and is filled by link: {empty_text_filled_count}")
        print(f"Entries removed where text remained empty despite having a link: {empty_text_removed_count}")

if __name__ == "__main__":
    preprocess()
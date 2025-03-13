import json
import pandas as pd
import re
import unicodedata
import os
import datetime
import argparse
from urllib.parse import urlparse
from nltk.stem.snowball import SnowballStemmer

# Initialize Albanian-compatible stemmer (fallback to English)
stemmer = SnowballStemmer("english")  

def extract_text_from_url(url):
    """Extracts meaningful text from the URL slug."""
    if not url:
        return ""  # Return empty if no URL is provided
    
    parsed_url = urlparse(url)
    slug = parsed_url.path.strip("/").split("/")[-1]  # Get the last part of the URL path
    clean_text = re.sub(r"[-_]", " ", slug)  # Replace hyphens/underscores with spaces
    clean_text = re.sub(r"\.html?$", "", clean_text)  # Remove file extensions like .html
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

def extract_facebook_data(json_file):
    project_dir = os.getcwd() 

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(project_dir, "tmp", timestamp)

    os.makedirs(output_dir, exist_ok=True)

    output_csv = os.path.join(output_dir, "filtered_election_posts.csv")
    
    print(f"Saving dataset to: {output_csv}")

    # Load JSON data
    with open(json_file, "r", encoding="utf-8") as file:
        data = json.load(file)
    
    extracted_data = []
    empty_text_filled_count = 0  # Count entries where text was successfully filled
    empty_text_removed_count = 0  # Count entries removed because text remained empty

    for post in data:
        text = post.get("text", "").strip()
        link = post.get("link", "").strip()
        
        # If text is empty, extract from link
        if not text and link:
            text = extract_text_from_url(link)
            if text:
                empty_text_filled_count += 1
            else:
                empty_text_removed_count += 1  # Count for removal

        # Only append records where text is not empty
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
    
    # Create DataFrame
    df = pd.DataFrame(extracted_data)

    # Normalize text
    df["text"] = df["text"].astype(str).apply(normalize_text)

    # Define base election-related keywords
    base_keywords = [
        "zgjedhje", "vota", "kqz", "komision", "kryeminister", "pdk", "ldk", "vv", "aak", "nisma",
        "familje", "shkurt", "numerim", "preleminare", "debat", "kandidat", "opozit", "koalicion",
        "elektorat", "parti", "qeveri", "mandat", "parlament", "fushat", "kryetar", "albin", "kurti",
        "lumir", "abdixhiku", "ramush", "haradinaj", "bedri", "hamza"
    ]

    # Generate regex pattern to match keyword variations
    keyword_patterns = [stem_word(word) for word in base_keywords]
    keyword_pattern = r'\b(' + '|'.join(keyword_patterns) + r')\w*\b'  # Match root + any suffix

    # Filter posts that contain election-related keywords
    df = df[df["text"].str.contains(keyword_pattern, regex=True, na=False)]

    # Count records with 0 comments
    zero_comments_count = (df["comments"] == 0).sum()

    # Remove rows where comments are 0
    df = df[df["comments"] > 0]

    # Save the final dataset
    df.to_csv(output_csv, index=False, encoding="utf-8")

    print(f"Dataset saved to: {output_csv}")
    print(f"Removed {zero_comments_count} records with 0 comments.")
    print(f"Entries where text was empty and is filled by link: {empty_text_filled_count}")
    print(f"Entries removed where text remained empty despite having a link: {empty_text_removed_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Facebook post data and filter election-related content.")
    parser.add_argument("--fileName", required=True, help="Path to the Facebook posts JSON file")
    
    args = parser.parse_args()
    extract_facebook_data(args.fileName)

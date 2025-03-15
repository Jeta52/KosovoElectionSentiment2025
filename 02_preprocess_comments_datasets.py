import pandas as pd
import ulid
from datetime import datetime

INITIAL_COMMENTS_DATASETS = [
    "scraped_datasets/fb_comments/KQZ_COMMENTS_INITIAL_DATASET.xlsx",
    "scraped_datasets/fb_comments/DEBAT_PLUS_COMMENTS_INITIAL_DATASET.xlsx",
    "scraped_datasets/fb_comments/KLAN_KOSOVA_COMMENTS_INITIAL_DATASET.xlsx",
    "scraped_datasets/fb_comments/NACIONALE_COMMENTS_INITIAL_DATASET.xlsx",
    "scraped_datasets/fb_comments/INDEKS_ONLINE_COMMENTS_INITIAL_DATASET.xlsx",
    "scraped_datasets/fb_comments/KANAL_10_COMMENTS_INITIAL_DATASET.xlsx"
]

OUTPUT_FILES = [
    "scraped_datasets/fb_comments/KQZ_COMMENTS_PREPROCESSED_DATASET.csv",
    "scraped_datasets/fb_comments/DEBAT_PLUS_COMMENTS_PREPROCESSED_DATASET.csv",
    "scraped_datasets/fb_comments/KLAN_KOSOVA_COMMENTS_PREPROCESSED_DATASET.csv",
    "scraped_datasets/fb_comments/NACIONALE_COMMENTS_PREPROCESSED_DATASET.csv",
    "scraped_datasets/fb_comments/INDEKS_ONLINE_COMMENTS_PREPROCESSED_DATASET.csv",
    "scraped_datasets/fb_comments/KANAL_10_COMMENTS_PREPROCESSED_DATASET.csv"
]

columns_to_keep = [
    "text",
    "likesCount",
    "date",
    "postTitle",
    "commentUrl",
    "facebookUrl"
]

def preprocess():
    for input_file, output_file in zip(INITIAL_COMMENTS_DATASETS, OUTPUT_FILES):
        print(f"Processing {input_file}...")
        
        # Load the dataset
        df = pd.read_excel(input_file)
        
        # Keep only the specified columns
        df_cleaned = df[columns_to_keep]
        
        # Remove rows where text is null
        df_cleaned = df_cleaned.dropna(subset=["text"])
        
        # Ensure date column is in datetime format
        df_cleaned["date"] = pd.to_datetime(df_cleaned["date"], errors='coerce')
        
        # Generate ULIDs based on timestamp
        df_cleaned.insert(0, "ID", [str(ulid.from_timestamp(d.timestamp() if pd.notna(d) else datetime.utcnow().timestamp())) for d in df_cleaned["date"]])
        
        # Rename columns
        df_cleaned = df_cleaned.rename(columns={
            "text": "Comment",
            "likesCount": "Likes Count",
            "date": "Comment Timestamp",
            "facebookUrl": "Post URL",
            "commentUrl": "Comment URL"
        })
        
        # Add empty annotation columns
        df_cleaned["Annot 1"] = ""
        df_cleaned["Annot 2"] = ""
        df_cleaned["Annot 3"] = ""
        df_cleaned["Final Annotation"] = ""
        
        # Save the cleaned dataset as CSV
        df_cleaned.to_csv(output_file, index=False, encoding="utf-8")
        
        print(f"Cleaned dataset saved as {output_file}")

if __name__ == "__main__":
    preprocess()
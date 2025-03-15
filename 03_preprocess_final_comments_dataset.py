import pandas as pd

# Input files
input_files = [
    "scraped_datasets/fb_comments/KQZ_COMMENTS_PREPROCESSED_DATASET.csv",
    "scraped_datasets/fb_comments/DEBAT_PLUS_COMMENTS_PREPROCESSED_DATASET.csv",
    "scraped_datasets/fb_comments/KLAN_KOSOVA_COMMENTS_PREPROCESSED_DATASET.csv",
    "scraped_datasets/fb_comments/NACIONALE_COMMENTS_PREPROCESSED_DATASET.csv",
    "scraped_datasets/fb_comments/INDEKS_ONLINE_COMMENTS_PREPROCESSED_DATASET.csv",
    "scraped_datasets/fb_comments/KANAL_10_COMMENTS_PREPROCESSED_DATASET.csv"
]

# Read and merge all datasets
all_comments = []
for file in input_files:
    df = pd.read_csv(file, parse_dates=["Comment Timestamp"])
    df["Source"] = file.split("/")[-1].replace("_COMMENTS_PREPROCESSED_DATASET.csv", "")  
    all_comments.append(df)

merged_df = pd.concat(all_comments)

# Sort by ID (ULID-based sorting)
merged_df = merged_df.sort_values(by="ID")

# Save merged dataset
merged_df.to_csv("ALL_COMMENTS_PREPROCESSED_DATASET.csv", index=False)
print("ALL_COMMENTS_PREPROCESSED_DATASET.csv created successfully.")

# Sampling dataset
sample_size = 4000
sampled_data = []
for file in input_files:
    df = pd.read_csv(file, parse_dates=["Comment Timestamp"])
    df["Source"] = file.split("/")[-1].replace("_COMMENTS_PREPROCESSED_DATASET.csv", "")
    if "KQZ" in file:
        sampled_data.append(df)  # Take all 400 rows from KQZ
    else:
        sampled_data.append(df.sample(n=sample_size, random_state=42))  # Sample 4000 rows from others

sampled_df = pd.concat(sampled_data)

# Sort sampled dataset by ID (ULID-based sorting)
sampled_df = sampled_df.sort_values(by="ID")

# Save sampled dataset
sampled_df.to_csv("SAMPLE_ALL_COMMENTS_PREPROCESSED_DATASET.csv", index=False)
print("SAMPLE_ALL_COMMENTS_PREPROCESSED_DATASET.csv created successfully.")

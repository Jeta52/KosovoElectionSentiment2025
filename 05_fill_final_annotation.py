import pandas as pd
import random

# Input and output file paths
input_file = "scraped_datasets/fb_comments/SAMPLE_ALL_COMMENTS_PREPROCESSED_DATASET.1.csv"
output_file = "scraped_datasets/fb_comments/SAMPLE_ALL_COMMENTS_PREPROCESSED_DATASET.99.csv"

# Load the dataset
df = pd.read_csv(input_file)

# Function to determine the most frequent annotation with tie-breaking
def fill_final_annotation(row):
    if pd.isna(row["Final Annotation"]):  # Check if Final Annotation is not filled
        annotations = [row["Annot 1"], row["Annot 2"], row["Annot 3"]]
        annotations = [ann for ann in annotations if not pd.isna(ann)]  # Filter out NaN values
        
        if annotations:  # If there are annotations available
            counts = {ann: annotations.count(ann) for ann in set(annotations)}  # Count occurrences
            max_occurrences = max(counts.values())
            most_common = [k for k, v in counts.items() if v == max_occurrences]
            
            if len(most_common) == 1:
                return most_common[0]  # If one clear most common value, return it
            else:
                return random.choice(most_common)  # In case of a tie, return a random choice
    return row["Final Annotation"]  # If Final Annotation is already filled, return it as is

# Apply the function to fill missing Final Annotation values
df["Final Annotation"] = df.apply(fill_final_annotation, axis=1)

# Save the updated dataset
df.to_csv(output_file, index=False)
print(f"{output_file} created successfully.")
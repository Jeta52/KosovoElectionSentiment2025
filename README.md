# Kosovo 2025 Elections Sentiment Analysis

<table>
  <tr>
   <td>
     <img src="https://github.com/user-attachments/assets/97d82b15-7058-4498-bdd4-efbae2425810" alt="University Logo" width="200" >
    </td>
    <td>
      <h2>UNIVERSITY OF PRISHTINA “HASAN PRISHTINA”</h2>
      <p><strong>FECE | Faculty of Electrical and Computer Engineering</p>
      <p><strong>Department: </strong>Computer Engineering</p>
      <p><strong>Program: </strong>Computer and Software Engineering</p>
    </td>
   
  </tr>
</table>

## Course Details
- **Machine Learning**
- **Prof. Dr. Ing. Lule AHMEDI**
- **Asst. Dr. Sc. Mërgim H. HOTI**
- **Level:** Master’s
- **Year:** 2024/2025

## Working Group
  - [Diare Daqi](https://github.com/Diaredaqi1)
  - [Jetë Lajçi](https://github.com/Jeta52)
  - [Melisa Alaj](https://github.com/melisaalaj)

## Content

- [Kosovo 2025 Elections Sentiment Analysis](#kosovo-2025-elections-sentiment-analysis)
  - [Course Details](#course-details)
  - [Working Group](#working-group)
  - [Content](#content)
  - [Project Description](#project-description)
    - [Data Scraping](#data-scraping)
- [01_Model Preparation](#01_model-preparation)
  - [Posts - Preprocessing Script](#posts---preprocessing-script)
    - [Overview](#posts-overview)
    - [Input Files](#posts-input-files)
    - [How It Works](#posts-how-it-works)
    - [Output Files](#posts-output-files)
  - [Comments - Preprocessing Script](#comments---preprocessing-script)
    - [Overview](#comments-overview)
    - [Input Files](#comments-input-files)
    - [How It Works](#comments-how-it-works)
    - [Output Files](#comments-output-files)
  - [Comments - Final Preprocessing Script](#comments---final-preprocessing-script)
    - [Overview](#final-comments-overview)
    - [Input Files](#final-comments-input-files)
    - [How It Works](#final-comments-how-it-works)
    - [Output Files](#final-comments-output-files)

## Project Description

This project focuses on **sentiment analysis** of public opinion regarding the **2025 Elections in Kosovo**. It aims to analyze social media comments to understand public sentiment on electoral processes.

We generated a dataset of **92888** comments (scraped_datasets/fb_comments/ALL_COMMENTS_PREPROCESSED_DATASET.csv), from which we took a sample of **20429 comments** (scraped_datasets/fb_comments/SAMPLE_ALL_COMMENTS_PREPROCESSED_DATASET.csv), by scraping comments from six different official Facebook pages of some of the biggest media platforms in Kosovo.

The final dataset we will be working with is **SAMPLE_ALL_COMMENTS_PREPROCESSED_DATASET.csv** which has: **20429 rows** and **12 columns**:

- **`ID`** – Unique identifier for each comment.  
- **`Comment`** – The content of the Facebook comment analyzed for sentiment.  
- **`Likes Count`** – The number of reactions (likes) the comment received.  
- **`Comment Timestamp`** – The date and time when the comment was posted.  
- **`postTitle`** – The title or content of the original Facebook post that the comment belongs to.  
- **`Comment URL`** – The direct URL to the specific comment on Facebook.  
- **`Post URL`** – The direct URL to the Facebook post where the comment was made.  
- **`Annot 1`** – First annotator's sentiment label for the comment.
- **`Annot 2`** – Second annotator's sentiment label for the comment.  
- **`Annot 3`** – Third annotator's sentiment label for the comment.  
- **`Final Annotation`** – The final sentiment label of the comment, determined using a **majority voting** strategy based on the three annotations.  
- **`Source`** – The Facebook page where the comment was collected from (e.g., `NACIONALE`, `KANAL_10`, etc.).  

Each annotator classifies the comment into one of the following categories:  
  - **`0` (Neutral)** – The comment does not express a clear positive or negative sentiment.  
  - **`1` (Positive)** – The comment expresses approval, support, or a favorable opinion.  
  - **`2` (Negative)** – The comment conveys criticism, disagreement, or a negative opinion.  

![image](https://github.com/user-attachments/assets/1632d805-2970-47be-aed8-d1c6141437b7)

Overview of project structure:

![ML-PROJECT-2025 (1)](https://github.com/user-attachments/assets/01a43b6b-83b9-422d-81b8-f312ee0c4366)

### Data Scraping  

For this project, we used the **Apify** platform ([https://apify.com/](https://apify.com/)) to scrape Facebook posts and comments related to the **2025 Kosovo Elections**. Apify provides automation tools, including specialized scrapers for social media platforms, allowing efficient data extraction.  

To gather election-related discussions, we used the following scrapers:  
- **Facebook Posts Scraper**: [https://apify.com/apify/facebook-posts-scraper](https://apify.com/apify/facebook-posts-scraper)  
- **Facebook Comments Scraper**: [https://apify.com/apify/facebook-comments-scraper](https://apify.com/apify/facebook-comments-scraper)  

We targeted posts and comments from six major Kosovo news and election-related Facebook pages:  

1. [KQZ - Komisioni Qendror i Zgjedhjeve](https://www.facebook.com/kqzkosova)  
2. [Debat Plus - Dukagjini](https://www.facebook.com/dukagjinidebatplus)  
3. [Klan Kosova](https://www.facebook.com/KlanKosovaOfficial1)  
4. [Nacionale](https://www.facebook.com/nacionalecom)  
5. [Indeks Online](https://www.facebook.com/IndeksonlineOfficial)  
6. [Kanal 10](https://www.facebook.com/Kanal10.live)  

#### Steps for Scraping  

The **scraping process** was carried out in a structured manner to ensure data quality and relevance:  

1. **Extracting Facebook Posts**  
   - We used the **Facebook Posts Scraper** ([link](https://apify.com/apify/facebook-posts-scraper)) to scrape posts from each Facebook page.  
   - Posts were collected from **January 9, 2025, to March 9, 2025**.  
   - The extracted data was saved into raw datasets for further processing.  

2. **Preprocessing Facebook Posts**  
   - We ran the **`01_preprocess_posts_datasets.py`** script to clean and filter the post datasets.  
   - The script removed irrelevant posts and extracted meaningful content, ensuring only **election-related posts** remained.  

3. **Extracting Comments from Election-Related Posts**  
   - We used the **Facebook Comments Scraper** ([link](https://apify.com/apify/facebook-comments-scraper)) to scrape comments from the **filtered posts** obtained in Step 2.  
   - All comments from these posts were collected and stored in raw datasets.  

4. **Preprocessing Facebook Comments**  
   - We ran the **`02_preprocess_comments_datasets.py`** script to clean and process the extracted comments.  
   - This step ensured that only structured, relevant, and properly formatted comment data was retained.  

5. **Merging Datasets**  
   - After processing all six Facebook pages, we merged the datasets into a single file:  
     - **`ALL_COMMENTS_PREPROCESSED_DATASET.csv`** → Contains all cleaned and processed comments.  
   - To ensure a representative dataset, we created a **sampled version** containing a balanced subset of comments:  
     - **`SAMPLE_ALL_COMMENTS_PREPROCESSED_DATASET.csv`** → A sampled dataset used for sentiment analysis.  

This structured approach ensured that the dataset contained only relevant election-related discussions, making it suitable for sentiment analysis and further machine learning applications.  


# 01_Model Preparation

## Posts - Preprocessing Script

### Overview <a id="posts-overview"></a>

This Python script (`01_preprocess_posts_datasets.py`) processes 6 Facebook posts datasets from JSON files and outputs cleaned versions as CSV files. The goal is to structure and prepare the data for so that we can have a list of posts that are related to 2025 Kosovo Elections. The **url** column contains links of all these facebook posts.

### Input Files <a id="posts-input-files"></a>

The script reads data from these XLSX files:

- `scraped_datasets/fb_posts/KQZ_POSTS_INITIAL_DATASET.json`
- `scraped_datasets/fb_posts/DEBAT_PLUS_POSTS_INITIAL_DATASET.json`
- `scraped_datasets/fb_posts/KLAN_KOSOVA_POSTS_INITIAL_DATASET.json`
- `scraped_datasets/fb_posts/NACIONALE_POSTS_INITIAL_DATASET.json`
- `scraped_datasets/fb_posts/INDEKS_ONLINE_POSTS_INITIAL_DATASET.json`
- `scraped_datasets/fb_posts/KANAL_10_POSTS_INITIAL_DATASET.json`

### How It Works <a id="posts-how-it-works"></a>

The script loads each JSON file and extracts relevant information, including post text, URL, timestamp, likes, shares, and comments. If a post lacks text but contains a link, the script attempts to extract meaningful text from the link’s URL. The script converts all text to lowercase. It removes accents (e.g., "ë" → "e") to ensure uniformity.

The script applies a keyword-based filtering system to retain only election-related posts. The filtering ensures that posts related to general topics or unrelated news are removed from the dataset. An exception is made for DEBAT PLUS, where the word "debat" was excluded to avoid misclassifications.

The following keywords were used to filter the dataset:

- `"zgjedhje"` (elections)  
- `"vota"` (vote)  
- `"kqz"` (Central Election Commission)  
- `"komision"` (commission)  
- `"kryeminister"` (prime minister)  
- `"pdk"` (Democratic Party of Kosovo - political party)  
- `"ldk"` (Democratic League of Kosovo - political party)  
- `"vv"` (Self-Determination Movement - political party)  
- `"vetevendosje"` (Self-Determination)  
- `"aak"` (Alliance for the Future of Kosovo - political party)  
- `"nisma"` (Social Democratic Initiative - political party)  
- `"9 shkurt"` (February 9)  
- `"numerim"` (vote counting)  
- `"preleminare"` (preliminary)  
- `"debat"` (debate)  
- `"kandidat"` (candidate)  
- `"opozit"` (opposition)  
- `"koalicion"` (coalition)  
- `"elektorat"` (electorate)  
- `"parti"` (party)  
- `"qeveri"` (government)  
- `"mandat"` (mandate)  
- `"kuvend"` (assembly)  
- `"deputet"` (MPs)  
- `"parlament"` (parliament)  
- `"fushat"` (campaign)  
- `"kurti"` (Albin Kurti)  
- `"abdixhiku"` (Lumir Abdixhiku)  
- `"haradinaj"` (Ramush Haradinaj)  
- `"hamza"` (Bedri Hamza)  

### Output Files <a id="posts-output-files"></a>

| File Name                                        | Description                             |
| ------------------------------------------------ | --------------------------------------- |
| `KQZ_POSTS_PREPROCESSED_DATASET.csv`            | Processed posts dataset from KQZ Page  |
| `DEBAT_PLUS_POSTS_PREPROCESSED_DATASET.csv`     | Processed posts dataset from DEBAT PLUS Page  |
| `KLAN_KOSOVA_POSTS_PREPROCESSED_DATASET.csv`    | Processed posts dataset from KLAN KOSOVA Page  |
| `NACIONALE_POSTS_PREPROCESSED_DATASET.csv`      | Processed posts dataset from NACIONALE Page  |
| `INDEKS_ONLINE_POSTS_PREPROCESSED_DATASET.csv`  | Processed posts dataset from INDEKS ONLINE Page  |
| `KANAL_10_POSTS_PREPROCESSED_DATASET.csv`       | Processed posts dataset from KANAL 10 Page  |

## Comments - Preprocessing Script

### Overview <a id="comments-overview"></a>

This Python script (`02_preprocess_comments_datasets.py.py`) processes 6 Facebook comments datasets from Excel files and outputs cleaned versions as CSV files. The goal is to structure and prepare the data for further analysis and annotation.

### Input Files <a id="comments-input-files"></a>

The script reads data from these XLSX files:

- `scraped_datasets/fb_comments/KQZ_COMMENTS_INITIAL_DATASET.xlsx`
- `scraped_datasets/fb_comments/DEBAT_PLUS_COMMENTS_INITIAL_DATASET.xlsx`
- `scraped_datasets/fb_comments/KLAN_KOSOVA_COMMENTS_INITIAL_DATASET.xlsx`
- `scraped_datasets/fb_comments/NACIONALE_COMMENTS_INITIAL_DATASET.xlsx`
- `scraped_datasets/fb_comments/INDEKS_ONLINE_COMMENTS_INITIAL_DATASET.xlsx`
- `scraped_datasets/fb_comments/KANAL_10_COMMENTS_INITIAL_DATASET.xlsx`

Each file contains Facebook comments from specific Facebook pages.

### How It Works <a id="comments-how-it-works"></a>

The script processes Facebook comments datasets by reading multiple Excel files, filtering relevant columns, and cleaning the data. It removes rows where the comment text is missing and ensures that the date column is properly formatted as a datetime object. Each comment is assigned a unique ULID based on its timestamp, ensuring consistent and sortable identifiers.  

To improve readability, the script renames key columns, such as changing “text” to “Comment” and “likesCount” to “Likes Count.” It also adds empty annotation fields for future manual classification or labeling of comments. Once preprocessing is complete, the cleaned dataset is saved as a CSV file with UTF-8 encoding.  

The script processes multiple datasets in a loop, providing status updates for each file being processed. This structured approach ensures the data is properly formatted and ready for further analysis, annotation, or machine learning applications.

### Output Files <a id="comments-output-files"></a>

| File Name                                        | Description                             |
| ------------------------------------------------ | --------------------------------------- |
| `KQZ_COMMENTS_PREPROCESSED_DATASET.csv`          |  The comments dataset from KQZ Page       |
| `DEBAT_PLUS_COMMENTS_PREPROCESSED_DATASET.csv`   | The comments dataset from DEBAT_PLUS Page  |
| `KLAN_KOSOVA_COMMENTS_PREPROCESSED_DATASET` | The comments dataset from KLAN_KOSOVA Page  |
| `NACIONALE_COMMENTS_PREPROCESSED_DATASET.csv` | The comments dataset from NACIONALE Page   |
| `INDEKS_ONLINE_COMMENTS_PREPROCESSED_DATASET.csv` | The comments dataset from INDEKS_ONLINE Page  |
| `KANAL_10_COMMENTS_PREPROCESSED_DATASET.csv` | The comments dataset from KANAL_10 Page   |

## Comments - Final Preprocessing Script

### Overview <a id="final-comments-overview"></a>

This phase is done by processing multiple Facebook comments datasets, merges them into one file, extracts a sample, and assigns a final annotation based on majority voting. The script (`03_preprocess_final_comments_dataset.py`) produces three output files:

1. **ALL\_COMMENTS\_PREPROCESSED\_DATASET.csv** – Contains all merged comments.
2. **SAMPLE\_ALL\_COMMENTS\_PREPROCESSED\_DATASET.csv** – A balanced sample with 4000 comments per source (except KQZ, which has 400).
3. **SAMPLE\_ALL\_COMMENTS\_PREPROCESSED\_DATASET.1.csv** – The sampled dataset with an assigned final annotation.

### Input Files <a id="final-comments-input-files"></a>

The script reads data from these CSV files:

- `scraped_datasets/fb_comments/KQZ_COMMENTS_PREPROCESSED_DATASET.csv`
- `scraped_datasets/fb_comments/DEBAT_PLUS_COMMENTS_PREPROCESSED_DATASET.csv`
- `scraped_datasets/fb_comments/KLAN_KOSOVA_COMMENTS_PREPROCESSED_DATASET.csv`
- `scraped_datasets/fb_comments/NACIONALE_COMMENTS_PREPROCESSED_DATASET.csv`
- `scraped_datasets/fb_comments/INDEKS_ONLINE_COMMENTS_PREPROCESSED_DATASET.csv`
- `scraped_datasets/fb_comments/KANAL_10_COMMENTS_PREPROCESSED_DATASET.csv`

Each file contains Facebook comments along with timestamps and annotations.

### How It Works <a id="final-comments-how-it-works"></a>

The script first merges all input files into a single dataset. It reads each file, extracts relevant comment data, and adds a `Source` column to indicate where each comment originated from. Once merged, the dataset is sorted based on the `ID` column using ULID-based sorting. The final merged dataset is saved as `ALL_COMMENTS_PREPROCESSED_DATASET.csv`.

After merging, the script creates a sampled dataset. It takes 4000 random comments from each dataset except for KQZ, where all 400 comments are included. The sampled dataset is also sorted by `ID` and saved as `SAMPLE_ALL_COMMENTS_PREPROCESSED_DATASET.csv`.

Finally, the script processes annotations. It reads the sampled dataset and examines the `Annot1`, `Annot2`, and `Annot3` columns. The most frequent annotation value among these three is selected as the `Final Annotation`. If there is a tie, such as `1, 2, 0`, the script defaults to assigning `0` (neutral). The updated dataset, now with final annotations, is saved as `SAMPLE_ALL_COMMENTS_PREPROCESSED_DATASET.1.csv`.

### Output Files <a id="final-comments-output-files"></a>

| File Name                                        | Description                             |
| ------------------------------------------------ | --------------------------------------- |
| `ALL_COMMENTS_PREPROCESSED_DATASET.csv`          | Full merged dataset sorted by ID        |
| `SAMPLE_ALL_COMMENTS_PREPROCESSED_DATASET.csv`   | Sampled dataset with equal distribution |
| `SAMPLE_ALL_COMMENTS_PREPROCESSED_DATASET.1.csv` | Sampled dataset with final annotation   |


## Data quality report

This Python script (`04_data_quality_report.py`) analyzes a preprocessed Facebook comments dataset to assess data quality and variable types. It categorizes data into **nominal, ordinal, continuous, and binary** types, checks for missing values, detects duplicate rows, identifies potential outliers, and ensures data consistency.  

### **1. Data Type Categorization**  
The script classifies each column based on its data type and unique value count:  
- **Nominal Variables**: Categorical text data (e.g., comment content).  
- **Ordinal Variables**: Categorical numerical data with an inherent order but limited unique values.  
- **Continuous Variables**: Numeric data with many unique values.  
- **Binary Variables**: Variables with only two unique values.
  
![image](https://github.com/user-attachments/assets/e536eaac-1ecb-49d9-8baa-e7594d00f2d2)

### **2. Data Quality Report**  
The script generates a **data quality report**, providing insights into:  
- **Missing Values**: Identifies columns with missing data.  
- **Duplicate Rows**: Counts redundant entries.  
- **Unique Values**: Shows the number of unique values per column.  
- **Outliers**: Uses the **IQR (Interquartile Range) method** to detect potential outliers in numerical columns.  
- **Data Type Consistency**: Ensures column types are correctly interpreted.
  
![image](https://github.com/user-attachments/assets/1fb83904-81bc-45ca-90a9-d75bfe84e4e0)

![image](https://github.com/user-attachments/assets/1d764751-3d19-405b-b25d-0dd6f74d12e9)

### **3. Non-Null & Null Value Counts**  
For each column, the script computes:  
- The **number of non-null values**.  
- The **number of null values**.  

Finally, the results are structured into a DataFrame for better visualization and analysis.  

This script provides a comprehensive overview of the dataset, ensuring it is **clean, structured, and ready for further processing**. 


  ___
🏷️ **License**: This project is open to use for anyone. You are free to use, modify, and distribute the code as needed.

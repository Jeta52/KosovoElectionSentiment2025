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
- [02_Model Training](#02_model-training)
  - [XLM-RoBERTa](#xml-roberta)
    - [Overview](#overview)
    - [Why XLM-RoBERTa?](#why-xlm-roberta)
    - [Input Files](#input-files)
    - [How It Works](#how-it-works)
    - [Output](#output)
    - [Example Result](#example-result)
  - [XGBoost](#xgboost)
    - [Overview](#overview)
    - [Why XGBoost?](#why-xgboost)
    - [Input Files](#input-files)
    - [How It Works](#how-it-works)
    - [Output](#output)
  - [Random Forest](#random-forest)
    - [Overview](#overview)
    - [Why Random Forest?](#why-random-forest)
    - [Input Files](#input-files)
    - [How It Works](#how-it-works)
    - [Output](#output)

## Project Description

This project focuses on **sentiment analysis** of public opinion regarding the **2025 Elections in Kosovo**. It aims to analyze social media comments to understand public sentiment on electoral processes.

We generated a dataset of **92888** comments (scraped_datasets/fb_comments/ALL_COMMENTS_PREPROCESSED_DATASET.csv), from which we took a sample of **20429 comments** (scraped_datasets/fb_comments/SAMPLE_ALL_COMMENTS_PREPROCESSED_DATASET.csv), by scraping comments from six different official Facebook pages of some of the biggest media platforms in Kosovo.

The final dataset we will be working with is **SAMPLE_ALL_COMMENTS_PREPROCESSED_DATASET.99.csv** which has: **20429 rows** and **12 columns**:

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
3. **SAMPLE\_ALL\_COMMENTS\_PREPROCESSED\_DATASET.99.csv** – The sampled dataset with an assigned final annotation.

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

# 02_Model_Training

## XML-RoBERTa

The Python script `xml_roberta_base_sentiment.py` fine-tunes the **XLM-RoBERTa** transformer model for multilingual sentiment classification on Kosovo Facebook comments related to the 2025 elections. It uses Hugging Face’s `transformers` and `datasets` libraries, as well as `scikit-learn` for evaluation. The script trains on labeled data and outputs accuracy and classification performance per sentiment category.

The goal is to classify comments into one of the three categories:

- **`0` (Neutral)**

- **`1` (Positive)**

- **`3` (Negative)**

### Why XLM-RoBERTa?
**XLM-RoBERTa** (Cross-lingual RoBERTa) is a transformer-based language model pretrained on data from **100+ languages**, including **Albanian** and other Balkan languages commonly found in Facebook comments from Kosovo.

It is well-suited for this project because:
- It supports **multilingual data** natively without translation.
- It handles **code-switching**, which is common in Kosovo social media (mixing Albanian and English).
- It captures **semantic meaning** better than traditional models like TF-IDF.
- It's pre-trained on a massive amount of web text, giving it strong general language understanding.

This makes XLM-RoBERTa a great choice for high-quality **sentiment classification** on noisy, multilingual Facebook comment data.

### Input Files
The script reads data from this CSV file:
- `sample_data/SAMPLE_ALL_COMMENTS_PREPROCESSED_DATASET.1.csv`

### How It Works
1. **Dependencies**  
   - Installs required packages (`transformers`, `datasets`, `pandas`, `scikit-learn`).
   - Disables Weights & Biases tracking for simplicity.

2. **Data Loading & Preprocessing**  
   - Loads and cleans the CSV file, ensuring it has `Comment` and `Final Annotation` columns.
   - Maps sentiment labels to integers using a custom label encoding dictionary.
   - Splits the data into training and test sets (80/20 split).
   - Converts pandas dataframes to Hugging Face `Dataset` format.
   - Tokenizes all comments using the pretrained `xlm-roberta-base` tokenizer.

3. **Model Setup**  
   - Loads `XLM-RoBERTa` for sequence classification.
   - Specifies training parameters (batch size, learning rate, number of epochs, logging paths).
   - Wraps everything into a Hugging Face `Trainer`.

4. **Evaluation**  
   - Predicts on the test set.
   - Calculates **accuracy**, **precision**, **recall**, and **F1-score** using `classification_report` from `sklearn`.
  
### Output
<img width="479" alt="image" src="https://github.com/user-attachments/assets/1024c931-2497-4d80-95e5-b3172a5950d8" />


## XGBoost for Sentiment Classification

The Python script `xgboost_sentiment.py` trains on labeled data and outputs accuracy and classification performance per sentiment category.
In this project, XGBoost is chosen for its ability to handle structured data efficiently, and it's well-suited for tasks where you need to classify text into multiple categories, as is the case here with three sentiment categories.
The goal is to train the model on labeled Facebook comments, where each comment is associated with a sentiment label (Neutral, Positive, Negative), and then use the trained model to classify new, unseen comments into these categories. The XGBoost model will help automate the classification of large volumes of comments, making it easier to analyze public opinion on the political candidates or issues surrounding the election.

### Why XGBoost?
**XGBoost** (Extreme Gradient Boosting) is a tree-based algorithm that uses gradient boosting techniques to make predictions. It is well-suited for this project because:

- **High performance**: XGBoost is known for its **speed and efficiency**, making it capable of handling large datasets effectively.
- **Handles imbalanced data**: XGBoost can manage **imbalanced datasets** by adjusting sample weights, which is helpful for datasets where certain categories (e.g., negative comments) are less frequent.
- **Feature importance**: It provides **feature importance scores**, which help in understanding which features contribute the most to the model’s predictions.
- **Flexibility**: It is **versatile**, suitable for both **classification** and **regression tasks**.
- **Widely used**: XGBoost has been very successful in **machine learning competitions**, proving its efficacy across various domains, including **text classification tasks** like sentiment analysis.

### Input Files
The script reads data from this CSV file:
- `sample_data/SAMPLE_ALL_COMMENTS_PREPROCESSED_DATASET.1.csv`

### How It Works

### Dependencies
Installs required packages (`xgboost`, `scikit-learn`, `pandas`, `seaborn`, `matplotlib`).

#### Data Loading & Preprocessing
1. Loads and cleans the CSV file, ensuring it has the **Comment** and **Final Annotation** columns.
2. Maps sentiment labels to integers using a custom label encoding dictionary:
   - `0` → Neutral
   - `1` → Positive
   - `2` → Negative
3. Splits the data into **training** and **test sets** (80/20 split).
4. Converts **pandas DataFrames** into arrays suitable for **XGBoost** training.
5. **Tokenizes the comments** using **TF-IDF** to transform text data into numerical format.

#### Model Setup
1. Loads **XGBoost** for **multi-class classification**.
2. Specifies training parameters (learning rate, number of trees, depth of trees, etc.).
3. Wraps everything into a **scikit-learn interface** for easy training and evaluation.

#### Evaluation
1. Trains the model on the training data and makes predictions on the test set.
2. Calculates **accuracy**, **precision**, **recall**, and **F1-score** using **classification_report** from `sklearn`.

### Key Parameters
- **learning_rate**: Controls how much the model learns in each iteration. Lower values may lead to better performance but require more trees.
- **n_estimators**: Number of trees to build. More trees can improve accuracy, but may lead to overfitting.
- **max_depth**: Maximum depth of each tree. Shallower trees prevent overfitting.
- **subsample**: Fraction of samples used to build each tree. Reduces overfitting by introducing randomness.
- **colsample_bytree**: Fraction of features used to build each tree. Helps in reducing overfitting
  
### Results & Performance
- **Confusion Matrix** visualizes the misclassifications between the three sentiment categories (Neutral, Positive, Negative).
- **Classification Report** provides a detailed evaluation of precision, recall, and F1-score for each class.
  
#### Output
![image](https://github.com/user-attachments/assets/2425edd6-57d8-4a49-9b00-5f76a377861d)
![image](https://github.com/user-attachments/assets/df8fc5bb-451f-451e-b0a4-9f250416cb80)

## Random Forest

### Overview  
The script `random_forest_sentiment.py` trains a **Random Forest classifier** using the `SAMPLE_ALL_COMMENTS_PREPROCESSED_DATASET.99.csv` file.

Each comment is labeled as:
- **0 = Neutral**
- **1 = Positive**
- **2 = Negative**

### Why Random Forest?  
Random Forest builds multiple decision trees and combines their results, making predictions more stable than a single tree. We chose it here because it’s fast and works well with TF-IDF features for text.

### Input Files  
- `scraped_datasets/fb_comments/SAMPLE_ALL_COMMENTS_PREPROCESSED_DATASET.99.csv`

### How It Works  
1. The script reads the CSV file and removes any rows with missing or invalid labels.  
2. It converts the comment texts into numerical vectors using **TF-IDF** with a limit of 5000 features.  
3. Then, it splits the data: **80% for training**, **20% for testing**.  
4. A `RandomForestClassifier` is trained with 100 trees (`n_estimators=100`).  
5. Finally, it prints the **accuracy** and a **classification report** with precision, recall, and F1-score.

### Output  
The model reached an **accuracy of 72.49%** on the test set. Here's how to read the classification report:

![image](https://github.com/user-attachments/assets/577c141b-ffab-4e05-acdf-56e4e2630017)

The model performs best at identifying **negative comments (class 2)**, because they are the most common in the dataset.

## Logistic Regression

### Overview  
The `logistic_regression_sentiment.py` that implements **Logistic Regression** uses two datasets:
- `SAMPLE_ALL_COMMENTS_PREPROCESSED_DATASET.99.csv` for the main training and testing
- `LEMMATIZED_SAMPLE_DATASET.csv` for testing how lemmatization might affect performance

Each comment is labeled as:
- **0 = Neutral**
- **1 = Positive**
- **2 = Negative**

### How It Works  
The script first vectorizes the comment text using **TF-IDF**, turning words into numbers that the model can understand. It then trains a logistic regression model using the main dataset and evaluates it both on that and on a small lemmatized sample.

Lemmatization is a text normalization step where words like *foli*, *flet*, and *fliste* are reduced to their root form (e.g., *fol*). This can help the model generalize better. We didn’t apply lemmatization to the full dataset, but we made a small sample (`LEMMATIZED_SAMPLE_DATASET.csv`) just to see what effect it might have.

Example difference:
```
Original comment: "Respekt për kryeministrin shembullor"
Lemmatized version: "respekt për kryeministër shembullor"
```

### Output  

![image](https://github.com/user-attachments/assets/c3c4efb7-0e5a-4c91-8159-1a0dab836b25)

**On the full dataset** (`SAMPLE_ALL_COMMENTS_PREPROCESSED_DATASET.99.csv`):
- Accuracy: **77.64%**
- Best performance was on **negative comments (2)** with high precision and F1-score.
- Neutral comments had the lowest precision, but good recall.

**On the lemmatized sample**:
- Accuracy: **72.44%**
- Performance was more balanced across all three classes, especially helpful for **positive and neutral comments**, though the small size (only 196 samples) makes it hard to judge fully.

  ___
🏷️ **License**: This project is open to use for anyone. You are free to use, modify, and distribute the code as needed.

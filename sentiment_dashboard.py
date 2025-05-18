import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------
# LOAD & CLEAN DATA
# -------------------------------
df = pd.read_csv("scraped_datasets/fb_comments/SAMPLE_ALL_COMMENTS_PREPROCESSED_DATASET.99.csv")

# Safe timestamp parsing
df["Comment Timestamp"] = pd.to_datetime(df["Comment Timestamp"], errors="coerce", utc=True)
df = df.dropna(subset=["Comment Timestamp"])
df = df.set_index("Comment Timestamp")

# Clean sentiment label column
df["label"] = pd.to_numeric(df["Final Annotation"], errors="coerce")
df = df[df["label"].isin([0, 1, 2])]

# -------------------------------
# SIDEBAR FILTERS
# -------------------------------
st.sidebar.title("🔍 Filters")

# Source filter
selected_source = st.sidebar.multiselect("Select News Source", df["Source"].dropna().unique())

# Date filter
min_date = df.index.min().date()
max_date = df.index.max().date()
start_date, end_date = st.sidebar.date_input(
    "Select Date Range",
    [min_date, max_date],
    min_value=min_date,
    max_value=max_date
)

# Apply filters
filtered_df = df.copy()
if selected_source:
    filtered_df = filtered_df[filtered_df["Source"].isin(selected_source)]

filtered_df = filtered_df.loc[
    (filtered_df.index.date >= start_date) & (filtered_df.index.date <= end_date)
]

# -------------------------------
# MAIN DASHBOARD
# -------------------------------
st.title("🗳️ Kosovo Election 2025 - Comment Sentiment Dashboard")

# 📊 Sentiment distribution
st.header("📊 Sentiment Distribution")
sentiment_counts = filtered_df["label"].value_counts().sort_index()
sentiment_counts.index = ["Positive (0)", "Neutral (1)", "Negative (2)"]
st.bar_chart(sentiment_counts)

# 📈 Weekly sentiment trend
st.header("📈 Weekly Sentiment Trend")
weekly_avg = filtered_df.resample("W")["label"].mean()
st.line_chart(weekly_avg)

# 💬 Sample comments
st.header("💬 Sample Comments by Sentiment")
sentiment_filter = st.radio(
    "Choose Sentiment to Display", [1, 2, 0],  # Default = 0 = Positive
    format_func=lambda x: {1: "Positive", 0: "Neutral", 2: "Negative"}[x]
)

sample_comments = filtered_df[filtered_df["label"] == sentiment_filter]["Comment"].dropna()
if len(sample_comments) > 0:
    st.write(sample_comments.sample(min(5, len(sample_comments))).reset_index(drop=True))
else:
    st.warning("No comments found for this sentiment and filter.")

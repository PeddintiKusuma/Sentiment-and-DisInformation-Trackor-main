import streamlit as st
import pandas as pd
import urllib.parse as urlparse
from urllib.parse import parse_qs
import matplotlib.pyplot as plt
from googleapiclient.discovery import build
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from wordcloud import WordCloud
import re
import plotly.graph_objects as go
from googleapiclient.errors import HttpError

# ============================
# CONFIG
# ============================

API_KEY = "Your API"
youtube = build("youtube", "v3", developerKey=API_KEY)

SENTIMENT_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
TOXIC_MODEL = "unitary/toxic-bert"
MAX_COMMENTS = 1000

# ============================
# LOAD MODELS
# ============================

@st.cache_resource
def load_models():
    sent_tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL)
    sent_model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL)

    tox_tokenizer = AutoTokenizer.from_pretrained(TOXIC_MODEL)
    tox_model = AutoModelForSequenceClassification.from_pretrained(TOXIC_MODEL)

    return sent_tokenizer, sent_model, tox_tokenizer, tox_model

sent_tokenizer, sent_model, tox_tokenizer, tox_model = load_models()
sent_model.eval()
tox_model.eval()

# ============================
# FUNCTIONS
# ============================

def extract_video_id(url):
    parsed_url = urlparse.urlparse(url)
    if parsed_url.hostname == "youtu.be":
        return parsed_url.path[1:]
    if parsed_url.hostname in ("www.youtube.com", "youtube.com", "m.youtube.com"):
        return parse_qs(parsed_url.query).get("v", [None])[0]
    return None


def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    return text.strip()


def fetch_all_comments(video_id):
    comments = []
    next_page_token = None

    try:
        while True:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100,
                pageToken=next_page_token
            )

            response = request.execute()

            for item in response.get("items", []):
                snippet = item["snippet"]["topLevelComment"]["snippet"]
                comments.append({
                    "Username": snippet["authorDisplayName"],
                    "Comment": snippet["textOriginal"],
                    "Likes": snippet["likeCount"],
                    "Published": snippet["publishedAt"]
                })

            next_page_token = response.get("nextPageToken")
            if not next_page_token or len(comments) >= MAX_COMMENTS:
                break

    except HttpError as e:
        st.error(f"YouTube API Error: {e}")
        return []

    return comments


# ============================
# AI MODELS
# ============================

def batch_sentiment(texts, batch_size=32):
    results = []
    labels = ["Negative", "Neutral", "Positive"]

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = sent_tokenizer(batch, return_tensors="pt",
                                truncation=True, padding=True, max_length=512)

        with torch.no_grad():
            outputs = sent_model(**inputs)

        scores = F.softmax(outputs.logits, dim=1)
        predictions = torch.argmax(scores, dim=1)

        for j in range(len(batch)):
            label = labels[predictions[j].item()]
            confidence = scores[j][predictions[j]].item() * 100
            results.append((label, round(confidence, 2)))

    return results


def batch_toxicity(texts, batch_size=32):
    results = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tox_tokenizer(batch, return_tensors="pt",
                               truncation=True, padding=True, max_length=512)

        with torch.no_grad():
            outputs = tox_model(**inputs)

        scores = torch.sigmoid(outputs.logits)

        for j in range(len(batch)):
            results.append(round(scores[j][0].item() * 100, 2))

    return results


def detect_bot(text, likes, toxicity, comment_counts):
    score = 0
    suspicious_words = ["crypto", "bitcoin", "giveaway",
                        "telegram", "whatsapp", "earn money"]

    if any(word in text.lower() for word in suspicious_words):
        score += 2
    if len(text.split()) <= 2:
        score += 1
    if comment_counts.get(text, 0) > 3:
        score += 2
    if toxicity > 80:
        score += 1
    if likes == 0:
        score += 1

    return min(score * 20, 100)


# ============================
# UI
# ============================

st.set_page_config(layout="wide")
st.title("🚀 TruthPulseAI - AI Comment Intelligence Dashboard")

video_url = st.text_input("Enter YouTube Video URL")

if video_url:
    video_id = extract_video_id(video_url)

    if not video_id:
        st.error("Invalid YouTube URL")
    elif st.button("Analyze Comments"):

        with st.spinner("Fetching & Processing Comments..."):

            raw_comments = fetch_all_comments(video_id)

            if not raw_comments:
                st.warning("No comments found.")
                st.stop()

            texts = [clean_text(item["Comment"]) for item in raw_comments]

            comment_counts = {}
            for text in texts:
                comment_counts[text] = comment_counts.get(text, 0) + 1

            sentiment_results = batch_sentiment(texts)
            toxicity_results = batch_toxicity(texts)

            data = []
            for i, item in enumerate(raw_comments):
                sentiment, conf = sentiment_results[i]
                toxicity = toxicity_results[i]
                bot_score = detect_bot(texts[i], item["Likes"],
                                       toxicity, comment_counts)

                data.append({
                    "Username": item["Username"],
                    "Comment": texts[i],
                    "Likes": item["Likes"],
                    "Sentiment": sentiment,
                    "Toxicity %": toxicity,
                    "Bot Score %": bot_score,
                    "Published": item["Published"]
                })

        df = pd.DataFrame(data)

        # ============================
        # EXECUTIVE SUMMARY
        # ============================

        st.subheader("📊 Executive Summary")

        total = len(df)
        pos = (df["Sentiment"] == "Positive").mean() * 100
        neg = (df["Sentiment"] == "Negative").mean() * 100
        neu = (df["Sentiment"] == "Neutral").mean() * 100

        high_bot = (df["Bot Score %"] >= 80).mean() * 100
        medium_bot = ((df["Bot Score %"] >= 40) &
                      (df["Bot Score %"] < 80)).mean() * 100
        avg_bot = df["Bot Score %"].mean()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Positive %", f"{pos:.2f}%")
        col2.metric("Negative %", f"{neg:.2f}%")
        col3.metric("Neutral %", f"{neu:.2f}%")
        col4.metric("Avg Bot Risk", f"{avg_bot:.2f}%")

        st.write("### 🤖 Bot Risk Breakdown")
        st.write(f"🔴 High Risk Bots: {high_bot:.2f}%")
        st.write(f"🟡 Medium Risk Bots: {medium_bot:.2f}%")

        # ============================
        # SENTIMENT TREND
        # ============================

        st.subheader("📈 Sentiment Trend")

        df["Published"] = pd.to_datetime(df["Published"])
        df.sort_values("Published", inplace=True)

        sentiment_over_time = (
            df.groupby(pd.Grouper(key="Published", freq="1H"))["Sentiment"]
            .value_counts()
            .unstack()
            .fillna(0)
        )

        fig = go.Figure()

        for sentiment in ["Positive", "Neutral", "Negative"]:
            if sentiment in sentiment_over_time.columns:
                fig.add_trace(go.Scatter(
                    x=sentiment_over_time.index,
                    y=sentiment_over_time[sentiment],
                    mode="lines+markers",
                    name=sentiment
                ))

        fig.update_layout(template="plotly_dark",
                          hovermode="x unified",
                          height=500)

        st.plotly_chart(fig, use_container_width=True)

        # ============================
        # WORD CLOUD
        # ============================

        st.subheader("☁ Word Cloud")
        text_blob = " ".join(df["Comment"])
        wordcloud = WordCloud(width=800, height=400).generate(text_blob)

        fig_wc, ax_wc = plt.subplots()
        ax_wc.imshow(wordcloud)
        ax_wc.axis("off")
        st.pyplot(fig_wc)

        st.subheader("💬 All Comments")
        st.dataframe(df, use_container_width=True)

        st.download_button(
            "Download Full Analysis CSV",
            df.to_csv(index=False),
            "truthpulse_full_analysis.csv",
            "text/csv"
        )
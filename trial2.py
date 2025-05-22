# %%
import praw
import pandas as pd
from textblob import TextBlob
import yfinance as yf
from datetime import datetime, timedelta
import snowflake.connector


# %%
from dotenv import load_dotenv
import os



load_dotenv()
reddit = praw.Reddit(
    client_id=os.getenv('REDDIT_CLIENT_ID'),
    client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
    user_agent=os.getenv('REDDIT_USER_AGENT')
)


# %% [markdown]
# Getting hot/top posts from r/CryptoCurrency and r/bitcoin

# %%
import pandas as pd
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

def fetch_posts(subreddit_name, limit=100):
    subreddit = reddit.subreddit(subreddit_name)
    posts = []
    for post in subreddit.hot(limit=limit):
        sentiment = analyzer.polarity_scores(post.title + " " + (post.selftext or ""))
        posts.append({
            'subreddit': subreddit_name,
            'title': post.title,
            'score': post.score,
            'comments': post.num_comments,
            'created_utc': datetime.utcfromtimestamp(post.created_utc),
            'sentiment': sentiment['compound']
        })
    return pd.DataFrame(posts)

df_crypto = fetch_posts('CryptoCurrency')
df_bitcoin = fetch_posts('bitcoin')

df_all = pd.concat([df_crypto, df_bitcoin])
df_all['sentiment_class'] = df_all['sentiment'].apply(lambda x: 'positive' if x > 0.2 else 'negative' if x < -0.2 else 'neutral')

df_all.head()

# %%
import praw
import pandas as pd
from datetime import datetime, timedelta
import re
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
import os



load_dotenv()
reddit = praw.Reddit(
    client_id=os.getenv('REDDIT_CLIENT_ID'),
    client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
    user_agent=os.getenv('REDDIT_USER_AGENT')
)



# Timeframe
one_week_ago = datetime.utcnow() - timedelta(days=7)

# Fetch posts
subreddit = reddit.subreddit("CryptoCurrency")
posts = []
for post in subreddit.top(time_filter="week", limit=1000):
    if datetime.utcfromtimestamp(post.created_utc) >= one_week_ago:
        posts.append({
            "title": post.title,
            "selftext": post.selftext,
            "created": datetime.utcfromtimestamp(post.created_utc).strftime('%Y-%m-%d'),
            "score": post.score,
            "num_comments": post.num_comments
        })

df = pd.DataFrame(posts)

# Crypto keyword match
keywords = ["BTC", "Bitcoin", "ETH", "Ethereum", "SOL", "Solana", "DOGE", "Dogecoin", 
            "ADA", "Cardano", "XRP", "AVAX", "DOT", "SHIB", "MATIC", "Polygon"]
pattern = r"\b(" + "|".join(re.escape(word.lower()) for word in keywords) + r")\b"
df["combined_text"] = (df["title"] + " " + df["selftext"]).str.lower()

# Count mentions
all_mentions = []
for text in df["combined_text"]:
    mentions = re.findall(pattern, text)
    all_mentions.extend(mentions)

mention_counts = Counter(all_mentions)
mention_df = pd.DataFrame(mention_counts.items(), columns=["Coin", "Mentions"]).sort_values(by="Mentions", ascending=False)
print("mention_count",mention_counts)
mention_df.to_csv("crypto_popularity.csv")
# Plot
plt.figure(figsize=(10, 6))
sns.barplot(data=mention_df.head(10), x="Mentions", y="Coin", palette="viridis")
plt.title("Top 10 Most Mentioned Cryptos on r/CryptoCurrency (Past Week)")
plt.xlabel("Mentions")
plt.ylabel("Coin")
plt.tight_layout()
plt.show()


# %% [markdown]
# Bitcoin sentiment vs price analysis

# %%
end_date = datetime.utcnow()
start_date = end_date - timedelta(days=30)
posts = []

for post in reddit.subreddit("bitcoin").top(time_filter="month", limit=500):
    created = datetime.utcfromtimestamp(post.created_utc)
    if created >= start_date:
        posts.append({
            "date": created.date(),
            "title": post.title,
            "selftext": post.selftext
        })

reddit_df = pd.DataFrame(posts)
reddit_df

# %%
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import pandas as pd
# # Combine title and selftext
reddit_df["combined_text"] = reddit_df["title"] + " " + reddit_df["selftext"]

# # Sentiment analysis
# def analyze_sentiment(text):
#     return TextBlob(text).sentiment.polarity

# reddit_df["sentiment"] = reddit_df["combined_text"].apply(analyze_sentiment)

# # Add sentiment class column
# def classify_sentiment(score):
#     if score > 0.2:
#         return "positive"
#     elif score < -0.2:
#         return "negative"
#     else:
#         return "neutral"

# reddit_df["sentiment_class"] = reddit_df["sentiment"].apply(classify_sentiment)
# reddit_df
# # Simulate BTC price data for 5 days
# price_data = [
#     {"date": datetime.utcnow().date() - timedelta(days=i), "price_usd": 65000 - i * 500}
#     for i in range(30)
# ]
# price_df = pd.DataFrame(price_data)
# price_df

# Load FinBERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
finbert_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Apply FinBERT to analyze sentiment
def analyze_sentiment_finbert(text):
    result = finbert_pipeline(text[:512])[0]  # truncate to 512 tokens for model input
    return result['label'], result['score']

# Assuming reddit_df is already defined with a 'combined_text' column
reddit_df[["sentiment_class", "sentiment_score"]] = reddit_df["combined_text"].apply(lambda x: pd.Series(analyze_sentiment_finbert(x)))
reddit_df.to_csv("btc_price_data.csv")
reddit_df





# %%
# import requests
# import pandas as pd
# import time
# import os
# import json
# from dotenv import load_dotenv

# # Load your Perplexity API token
# load_dotenv()
# PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")  # Replace with your token or hardcode it securely

# # Sample reddit_df with a combined_text column
# # Make sure you’ve already created reddit_df with the "combined_text" column
# reddit_df["combined_text"] = reddit_df["title"] + " " + reddit_df["selftext"]

# # Define function to call Perplexity API
# def get_perplexity_sentiment(text):
#     url = "https://api.perplexity.ai/chat/completions"
#     headers = {
#         "Authorization": f"Bearer {PERPLEXITY_API_KEY}",  # replace with your actual API key
#         "Content-Type": "application/json"
#     }
#     prompt = f"""
#     Analyze the following Reddit post and classify its sentiment as Positive, Neutral, or Negative.
#     Also provide a confidence score between 0 and 1.
    
#     Post: {text}
    
#     Respond ONLY with a JSON object like this:
#     {{ "sentiment": "Positive", "score": 0.92 }}
#     """

#     payload = {
#         "model": "sonar",
#         "messages": [
#             {"role": "system", "content": "Be precise and concise."},
#             {"role": "user", "content": prompt}
#         ]
#     }

#     try:
#         response = requests.post(url, json=payload, headers=headers)
#         if response.status_code == 200:
#             content = response.json()["choices"][0]["message"]["content"]

#             # Extract JSON using regex
#             match = re.search(r'\{.*\}', content, re.DOTALL)
#             if match:
#                 json_str = match.group(0)
#                 json_data = json.loads(json_str)
#                 return pd.Series([json_data.get("sentiment", "Neutral"), json_data.get("score", 0.0)])
#             else:
#                 print("No valid JSON found in response.")
#                 return pd.Series(["Neutral", 0.0])
#         else:
#             print("API error:", response.status_code)
#             return pd.Series(["Neutral", 0.0])

#     except Exception as e:
#         print("Error:", e)
#         return pd.Series(["Neutral", 0.0])

# # Apply sentiment extraction
# results = reddit_df["combined_text"].apply(lambda x: get_perplexity_sentiment(x))
# reddit_df[["sentiment_class", "sentiment_score"]] = results

# # Save to CSV
# reddit_df.to_csv("bitcoin_sentiment_perplexity.csv", index=False)


# %%
import requests
from datetime import datetime, timedelta

price_data = []
for i in range(30):
    date = (datetime.utcnow() - timedelta(days=i)).strftime("%d-%m-%Y")
    url = f"https://api.coingecko.com/api/v3/coins/bitcoin/history?date={date}"
    response = requests.get(url)
    if response.status_code == 200:
        json_data = response.json()
        try:
            price_usd = json_data["market_data"]["current_price"]["usd"]
            price_data.append({
                "date": datetime.strptime(date, "%d-%m-%Y").date(),
                "price_usd": price_usd
            })
        except KeyError:
            continue

price_df = pd.DataFrame(price_data)


# %%
merged_df = pd.merge(reddit_df, price_df, on="date", how="inner")


# %%
merged_df.to_csv("btc_sentiment_price.csv", index=False)
print("✅ Data saved to btc_sentiment_price.csv")
print(merged_df.dtypes)

# %%
import snowflake.connector
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

# Load your CSV
df = pd.read_csv("btc_sentiment_price.csv")

# user=os.getenv('SNOWFLAKE_USER')
# print(user)
# Snowflake connection info
# conn = snowflake.connector.connect(
#     user=os.getenv('SNOWFLAKE_USER'),
#     password=os.getenv('SNOWFLAKE_PASSWORD'),
#     account=os.getenv('SNOWFLAKE_ACCOUNT'), # e.g. abcde-xy12345
#     warehouse=os.getenv('SNOWFLAKE_WAREHOUSE'),
#     database=os.getenv('SNOWFLAKE_DATABASE'),
#     schema=os.getenv('SNOWFLAKE_SCHEMA')
# )


conn = snowflake.connector.connect(
    user="KUSHALI18",
    password="MaliniCse@1234",
    account="WATWEFG-QC14872",
    warehouse="SNOWFLAKE_LEARNING_WH",
    database="DATA_ANALYTICS_DB",
    schema="DATA_ANALYTICS_SCHEMA"
)


# %%
create_table_query = """
CREATE OR REPLACE TABLE BTC_SENTIMENT_PRICE (
    date DATE,
    title STRING,
    selftext STRING,
    combined_text STRING,
    sentiment_class STRING,
    sentiment_score FLOAT,
    price_usd FLOAT
);
"""

cursor = conn.cursor()
cursor.execute(create_table_query)
cursor.close()


# %%
# Fix column names to be lowercase without special characters
merged_df.columns = [col.lower() for col in merged_df.columns]
print(merged_df.columns)


# %%
from snowflake.connector.pandas_tools import write_pandas

# merged_df["record_date"] = pd.to_datetime(merged_df["record_date"]).dt.date

success, nchunks, nrows, _ = write_pandas(
    conn,
    merged_df,
    table_name="BTC_SENTIMENT_PRICE",
    auto_create_table=False,  # Set to True if you want Snowflake to auto-create
    overwrite=True             # Overwrites the table
)

print(f"✅ Upload success: {success}, Rows uploaded: {nrows}")




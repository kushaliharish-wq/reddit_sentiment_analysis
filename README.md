# 📈 Bitcoin Reddit Sentiment Analytics Dashboard

An end-to-end data engineering and analytics project that explores how Reddit sentiment correlates with Bitcoin’s price, trading volume, and community popularity using Snowflake, FinBERT, and Tableau.

---

## 📊 Sheet 1: Bitcoin Price vs Reddit Sentiment (30 Days)

### Objective
Explore how sentiment on Reddit’s r/Bitcoin subreddit aligns with Bitcoin’s daily price trends.

### 🗂️ Data Sources
- **Reddit API**: Top posts from r/Bitcoin
- **CoinGecko API**: BTC historical daily closing prices
- **FinBERT**: Financial sentiment classification model

### ETL Process
1. **Extract**:
   - Scrape Reddit posts with `praw`
   - Fetch BTC price data using CoinGecko API
2. **Transform**:
   - Clean and tokenize text
   - Run FinBERT to label posts as Positive, Negative, or Neutral
3. **Load**:
   - Store data in **Snowflake**
4. **Visualize**:
   - Dashboard built using **Tableau Public**

### 🔍 Key Insights
- BTC price showed an upward trend, even on Negative/Neutral sentiment days.
- Sentiment fluctuated frequently and did not predict short-term price moves.
- Reddit sentiment is better used in conjunction with volume or trend metrics.

### Takeaway
Reddit sentiment is a **contextual signal**, especially when paired with volume or volatility data. It reflects retail trader emotion more than direct price action.

---

## 🔍 Sheet 2: Reddit’s Most Popular Coins (Past 7 Days)

### Objective
Track which cryptocurrencies are mentioned most frequently on r/CryptoCurrency to identify market interest.


### 🔍 Key Insights
- **Bitcoin** leads Reddit conversations (~125 mentions).
- **Ethereum** and **BTC** follow closely (~45 mentions each).
- **Solana**, **Dogecoin**, and **XRP** show strong community traction.
- Less-discussed coins include **Polygon**, **Cardano**, and **ADA**.

### Takeaway
Rising mention counts can **signal early hype or upcoming volatility**. Monitoring social chatter is essential for understanding retail market dynamics.

---

## 📊 Sheet 3: 24hr Trading Volume vs Reddit Sentiment

### 🧭 Objective
Visualize how daily sentiment on Reddit aligns with Bitcoin’s 24-hour trading volume.

### 🔍 Key Insights
- **Positive sentiment** correlates with **high-volume** days (~$64B on May 8–9).
- **Negative sentiment** tends to appear on **low-volume** days (early May).
- **Neutral days** span the middle volume range, suggesting indecision.

### Takeaway
There’s a **loose correlation** between positive sentiment and higher trading activity, likely due to retail speculation. Sentiment can enhance volume analysis to predict spikes.

---

## ⚙️ Tech Stack

| Tool               | Purpose                                      |
|--------------------|----------------------------------------------|
| `Python`           | Core ETL logic                               |
| `praw`             | Reddit post extraction                       |
| `transformers`     | FinBERT sentiment classification             |
| `pandas`           | Data transformation                          |
| `Snowflake`        | Cloud data warehouse                         |
| `Tableau Public`   | Interactive dashboards                       |
| `CoinGecko API`    | BTC price and volume data                    |

---
## 📁 Project Structure

| File Name                 | Description                                  |
|--------------------------|----------------------------------------------|
| `reddit_sentiment_analysis.ipynb` | Jupyter notebook for ETL pipeline      |
| `btc_price_data.csv`    | Bitcoin sentiment and price data            |
| `btc_sentiment_price.csv`    | Bitcoin sentiment and price data            |
| `btc_sentiment_volume.csv`    | Bitcoin sentiment and 24hr volume data     |
| `crypto_popularity.csv`      | Coin mention counts from Reddit             |
| `volume_data.csv`            | BTC trading volume over the past 60 days    |
| `README.md`                  | Project documentation and insights          |


---

## How to Reproduce

1. Clone the repo.
2. Set up `.env` file with your Reddit and Snowflake credentials.
3. Run the ETL Jupyter notebook:
   ```bash
   jupyter notebook reddit_sentiment_etl.ipynb
4. Connect Tableau to Snowflake or the generated .csv files.

5. Explore visualizations linked below.

---

## 🔗 Links
[Live Dashboard on Tableau Public]([https://public.tableau.com/app/profile/your-dashboard-link](https://public.tableau.com/views/price_sentiment/Sheet1?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link))

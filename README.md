# media-engagement-pred-ML-model
## 1. Project Overview
This repository contains an end-to-end Classical Machine Learning pipeline designed to classify and predict social media content virality. Built entirely within Google Colab and engineered using a modular approach, this project bypasses raw engagement scoring to build a robust binary classification model. Rather than optimizing blindly for marginal accuracy gains, the architecture focuses heavily on thorough data auditing, schema validation, handling informative missingness, and structural feature engineering.

The pipeline processes multi-platform metadata, transforms linguistic and temporal features, benchmarks multiple classical machine learning algorithms (including Decision Trees and Random Forests), and extracts actionable feature importances to understand what truly drives online interaction velocity.

---

## 2. Problem Statement
In digital content strategy, predicting whether a post will achieve high audience traction is crucial for brand optimization. However, standard optimization pipelines face two core engineering challenges:
1. **Algorithmic Stability against Extreme Outliers:** Raw metrics like engagement rates frequently experience massive mathematical spikes due to low impression denominators, throwing off standard linear regression models.
2. **Feature Interdependence vs. Synthetic Bias:** Many publicly available data sandboxes utilize machine-generated metrics that violate real-world power laws. 

---

## 3. Dataset Description
The project utilizes a comprehensive, machine-generated **Social Media Engagement Dataset** simulating platform metrics across Instagram, TikTok, YouTube, and X (formerly Twitter). The dataset comprises 12,000 records tracking content properties, NLP sentiment metrics, and interaction counts.
Link to the dataset: https://www.kaggle.com/datasets/subashmaster0411/social-media-engagement-dataset/data 
(Credits goes to the respective owner for creating this dataset for public use. This project is merely for learning usecase.)

### Core Data Schema & Attributes:
* **Identification & Metadata:** `post_id`, `user_id`, `timestamp`, `platform`, `location`, `language`
* **Content Properties:** `text_content`, `hashtags`, `mentions`, `keywords`, `topic_category`
* **Marketing Context:** `brand_name`, `product_name`, `campaign_name`, `campaign_phase`
* **Linguistic/AI Appraisals:** * `sentiment_score` / `user_past_sentiment_avg`: Bounded polarity vectors ranging from `[-1.0, 1.0]` (Negative to Positive).
  * `toxicity_score`: Bounded probability scale `[0.0, 1.0]` tracking content aggressiveness.
* **Volume Metrics (Uniform Scale):** `likes_count`, `shares_count`, `comments_count`, `impressions`
* **Calculated Kinetics:** * `engagement_rate`: Derived platform interaction ratio.
  * `buzz_change_rate`: Velocity vector bounded between `[-100%, 100%]` tracking trend acceleration/decay.
## 4. Technologies Used
## 5. Project Structure
## 6. Results & Observations
## 7. Steps to Run the Project
### Clone the repo
```
git clone https://github.com/jaya-v07/media-engagement-pred-ML-model.git
```

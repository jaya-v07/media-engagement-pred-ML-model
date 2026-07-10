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
* numpy
* pandas
* kagglehub
* scikit-lean
* xgboost
## 5. Project Structure
media-engagement-pred-ML-model
|
|
| 

## 6. Results & Observations

To thoroughly evaluate the dataset, three modern, complex machine learning architectures were deployed across multiple controlled feature configurations:
* **Decision Tree Classifier** (Baseline non-linear boundary rules)
* **Random Forest Classifier** (Bagging Ensemble optimization)
* **XGBoost Classifier** (Sequential Gradient Boosting)

---

### 1. Controlled Experimentation Matrix

The pipeline was tracked across three distinct feature engineering strategies to locate true predictive signal vs. structural artifacts:

| Experiment Configuration | Features Included | Average Model Accuracy | Primary Data Finding |
| :--- | :--- | :--- | :--- |
| **A. Pre-Publishing Metadata** | `sentiment_score`, `toxicity_score`, `text_length`, `has_mention`, `user_past_sentiment_avg`, `user_engagement_growth`, `buzz_change_rate` | **~52.00%** | **Zero-Signal Boundary:** Equivalent to a random coin flip. Content characteristics are completely decoupled from performance metrics. |
| **B. Geo-Social Demographics** | Label Encoded `platform` and `language` indicators | **~52.00%** | **Zero Distribution Bias:** Current simple encodings show that specific platforms or languages hold no basic linear weight over virality. |
| **C. Script-Bias Isolation** | `impressions` (Isolating a single continuous column) | **~83.54%** | **Structural Leakage:** Bafflingly high accuracy. Exposes a hidden threshold embedded directly within the synthetic dataset generator's logic. |

---

### 2. Deep-Dive Analytical Insights

#### 🔍 The Metadata & Demographics Flatline (Configs A & B)
When the classification models are restricted strictly to features available *prior to hitting publish* (text analytics, categorical channels, historical account momentum), all three complex architectures—including XGBoost—flatline at an identical **52% accuracy**. 

The structured classification reports reveal completely uniform precision ($0.51$) and recall ($0.50$) thresholds across both target classes. This mathematically validates that the synthetic pipeline engine used to generate this dataset did not program any causal or correlative link between what a post contains (or where it is posted) and how it performs. 

#### 🚨 The Impressions Paradox (Config C)
The most jarring finding is that isolating **strictly the `impressions` column yields an anomalous ~83.54% accuracy**. In organic social platforms, impressions operate independently of raw virality boundaries (high views do not automatically guarantee high engagement rates). 

However, in this synthetic environment, this 83% score exposes the backend architecture of the generation script: the developer programmed interaction scales (`likes`, `comments`, `shares`) directly as a linear fraction of `impressions`. Because our target class threshold (`is_viral`) was split at the median engagement rate, it accidentally embedded a sharp mathematical boundary directly within the `impressions` feature space. The models aren't learning human behavior; they are exploiting a code artifact.

---

## 🔮 Future Improvements

While the baseline metadata and demographic models currently sit at a 52% random-guess threshold, the project can be advanced through the following steps:

1. **Advanced Categorical Feature Interaction:** Instead of using simple Label Encoding on `platform` and `language`, future iterations should utilize **Target Encoding** or **One-Hot Encoding Matrix Transformations**. This will force the models to look for complex, multi-variable correlations (e.g., *"Do high-sentiment English posts perform uniquely better on Instagram vs. Reddit?"*).
2. **Feature Crossing:** Engineering custom composite features, such as multiplying `buzz_change_rate` by specific encoded `platform` IDs, to see if catching a global trend wave matters more on specific distribution channels than others.
3. **Hyperparameter Tuning:** Implementing `GridSearchCV` or `RandomizedSearchCV` on the XGBoost classifier once high-interaction feature combinations are engineered, ensuring the gradient boosting paths are optimized for subtle demographic correlations.
## 7. Steps to Run the Project
### Clone the repo and go to repo folder
```
git clone https://github.com/jaya-v07/media-engagement-pred-ML-model.git
cd media-engagement-pred-ML-model
```
### Open the code in src/prediction-ml-model.ipynb and run it.

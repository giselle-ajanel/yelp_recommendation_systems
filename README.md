# Yelp Recommendation Systems & Similarity Search with PySpark

Scalable recommendation systems built on the Yelp dataset using Apache Spark and Python, covering locality sensitive hashing for similarity search, item-based collaborative filtering, model-based prediction with XGBoost, and a competition-entry hybrid system.

---

## Overview

This repo explores four approaches to the same problem: given a user and a business, predict the star rating the user would give. Each method makes different assumptions about what drives ratings and handles the core challenges of recommendation — data sparsity, cold start, and scalability — differently.

All systems are trained on `yelp_train.csv` and evaluated on test datasets. Datasets are not included due to size and licensing; they are available via the Yelp Open Dataset.

---

## Repo Structure

* `task1.py` — LSH-based business similarity search (Jaccard >= 0.5)
* `task2_1.py` — Item-based CF with Pearson similarity
* `task2_2.py` — Model-based CF with XGBoost
* `task2_3.py` — Hybrid: switching/blending CF + XGBoost
* `competition.py` — Competition entry: Adaptive hybrid system combining Pearson CF and fine-tuned XGBoost (RMSE 0.9949)

---

## Technical Highlights

* **Bayesian Rating Smoothing (task2_2.py, task2_3.py, competition.py)**  
  Raw average star ratings can be heavily skewed for users or businesses with only one or two reviews. Incorporating a global baseline prior smooths extreme ratings toward the dataset mean, stabilizing prediction baselines when handling sparse user and business profiles.

* **Adaptive Tiered Alpha Blending (task2_3.py, competition.py)**  
  A strict switch (use CF or use model) requires conservative thresholds, meaning CF almost never fires and the system defaults to the model in nearly every case. Rather than relying on a fixed alpha blend across all rows, `competition.py` dynamically adjusts reliance on collaborative filtering based on user and business interaction counts. Sparse interactions rely entirely on the XGBoost model, while active interactions progressively increase the contribution of item-based CF.

* **Pearson Similarity with Baseline Fallback (task2_1.py, task2_3.py, competition.py)**  
  Item-based CF computes Pearson similarity per business pair using shared user ratings. When items have 1 or fewer co-raters, traditional correlation is impossible to calculate reliably. The system gracefully falls back to a normalized similarity metric derived from overall business average ratings.

* **Extensive Feature Extraction from Metadata (competition.py)**  
  To maximize the predictive power of XGBoost, raw text attributes and categories in `business.json` are parsed into structured feature signals. This includes price ranges, primary domain category flags (such as Food, Restaurants, Coffee, Nightlife, Fast Food, and Bars), and operational booleans (credit card acceptance, delivery, suitability for kids, and reservations).

---

## Task Breakdown

### Task 1: LSH Business Similarity Search (`task1.py`)
Finds all business pairs with Jaccard similarity >= 0.5 without comparing all O(N^2) pairwise combinations.
* **Pipeline:**
  1. **Characteristic Mapping:** Constructs user-business interaction sets for each distinct business.
  2. **MinHash Signatures:** Computes 50 linear hash signatures per business to approximate similarity profiles.
  3. **LSH Banding:** Divides signatures into 25 bands with 2 rows per band, grouping candidate pairs into buckets.
  4. **Verification:** Computes exact Jaccard similarity on candidate pairs, saving valid pairs to the output file.

### Task 2.1: Item-Based Collaborative Filtering (`task2_1.py`)
Predicts ratings using Pearson similarity between businesses based on co-rated user feedback.
* **Pipeline:**
  1. Computes Pearson similarity across shared user ratings between target businesses.
  2. Uses business rating averages as similarity fallbacks when co-rating counts are insufficient.
  3. Predicts ratings by weighting user history against item similarities.
  4. Resolves cold-start cases by falling back through business, user, or global dataset averages.

### Task 2.2: Model-Based CF with XGBoost (`task2_2.py`)
Predicts ratings using user engagement metrics and business profile attributes.
* **Feature Set:**
  * **User:** Smoothed rating average, raw rating average, review count, useful votes, funny votes, cool votes, fans.
  * **Business:** Smoothed rating average, raw rating average, review count, average stars, latitude, longitude, open status.
  * **Interactions:** User bias, business bias, and rating difference metrics.
* **XGBoost Configuration:** `max_depth=4`, `n_estimators=1000`, `learning_rate=0.01`, `subsample=0.75`, `colsample_bytree=0.75`, `reg_lambda=3.0`.

### Task 2.3: Hybrid Recommender System (`task2_3.py`)
Combines predictions from item-based collaborative filtering and the XGBoost model.
* **Pipeline:**
  1. Calculates Pearson item-based predictions and XGBoost model predictions independently.
  2. Blends predictions using a weighted alpha factor (alpha = 0.1 for CF, 0.9 for XGBoost) when user and business history exist.
  3. Automatically defaults to pure model predictions when cold-start cases occur.

### Competition: Adaptive Hybrid Recommender System (`competition.py`)
Our best-performing system, expanding metadata features and dynamically scaling collaborative filtering weights based on interaction density.
* **Expanded Feature Set:**
  * **User:** Smoothed ratings, review counts, reaction votes (useful, funny, cool), fans, average star ratings.
  * **Business:** Smoothed ratings, review counts, price range, total category count, primary domain indicators (Food, Restaurants, Coffee, Nightlife, Fast Food, Bars), and attribute booleans (credit cards, delivery, good for kids, reservations).
* **Validation Results:**
  * **RMSE:** `0.9949`
  * **Execution Time:** ~653 seconds
  * **Error Distribution:**
    * [0, 1): 101,320
    * [1, 2): 33,286
    * [2, 3): 6,524
    * [3, 4): 913
    *  \>= 4: 1

---

## Tech Stack

* Python 3.x
* Apache Spark (RDD-based processing)
* XGBoost

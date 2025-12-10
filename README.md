# MovieLens Recommendation System 🎬

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Library](https://img.shields.io/badge/Library-Surprise-orange)
![Library](https://img.shields.io/badge/Library-Scikit_Learn-green)
![Status](https://img.shields.io/badge/Status-Complete-success)

Check it out!   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1YaHY4TI-8qcVxMRne6T1vk8Apgsq2ryv?usp=sharing)

A personalized movie recommendation engine designed to solve user **"Choice Paralysis"** by filtering content based on latent user tastes rather than just global popularity. This project compares a **Memory-Based (kNN)** approach against a **Model-Based (SVD)** approach.

---

## 📌 Project Overview

### The Problem
Streaming platforms suffer from the **Long Tail** problem. A few popular movies get all the attention, while 80% of the catalog (niche gems) remains undiscovered. Users struggle to find content they actually like among thousands of options.

### The Solution
I built a collaborative filtering system that identifies patterns in user behavior to predict ratings for unseen movies.
* **Baseline Model:** k-Nearest Neighbors (kNN) using Cosine Similarity.
* **Challenger Model:** Singular Value Decomposition (SVD) using Matrix Factorization.

---

## 📊 Data Engineering & EDA

**Dataset:** [MovieLens Small](https://grouplens.org/datasets/movielens/) (100,000 ratings, 600 users, 9,000 movies).

### 1. Preprocessing Pipeline
* **Ingestion:** Merged `ratings.csv` (Interactions) and `movies.csv` (Metadata).
* **Noise Filtering:** Removed movies with **< 10 ratings** to reduce statistical noise and variance. This reduced the feature space from ~9,700 to **2,269 high-quality movies**.
* **Sparsity Handling:** The User-Item Matrix was **98.3% Empty**. I used `scipy.sparse` (CSR Matrix) to handle this efficiently in memory.

### 2. Key Insights
* **Positivity Bias:** Users tend to rate movies they like. The average rating is **~3.5/5.0**, meaning missing data is often "negative feedback."
* **Power Law Distribution:** The dataset follows a "Long Tail" distribution, where the top 20% of movies account for 80% of ratings.

---

## 🧠 Model Architecture

### Phase 1: k-Nearest Neighbors (kNN)
* **Type:** Memory-Based (Item-Item Collaborative Filtering).
* **Metric:** **Cosine Similarity** (Measures the angle between vectors to handle strict vs. generous raters).
* **Limitations:** Fails on sparse data (Cold Start) where there is no direct overlap between users.

### Phase 2: Matrix Factorization (SVD)
* **Type:** Model-Based.
* **Algorithm:** Singular Value Decomposition (via `scikit-surprise`).
* **Mechanism:** Decomposes the matrix into **100 Latent Factors** (e.g., Hidden features like "Action", "Vintage", "Dark Tone") using **Stochastic Gradient Descent (SGD)**.
* **Advantage:** Can predict ratings even for sparse data by connecting users and items through latent features.

---

## 🏆 Results & Benchmarking

I evaluated both models using **RMSE (Root Mean Squared Error)** on a held-out Test Set.

| Model | RMSE Score | Verdict |
| :--- | :--- | :--- |
| **kNN (Baseline)** | 0.9583 | Good for explainability, but struggles with sparsity. |
| **SVD (Challenger)** | **0.8517** | **Winner.** 11% improvement in accuracy. |

**Conclusion:** The SVD model is significantly more robust, predicting ratings with an average error of just **0.85 stars**.

---

## 🛠️ Installation & Usage

### 1. Clone the Repo
```bash
git clone [https://github.com/yourusername/MovieLens-Project.git](https://github.com/yourusername/MovieLens-Project.git)
cd MovieLens-Project

# Create a virtual environment
conda create -n movielens python=3.11 -y
conda activate movielens

# Install dependencies
conda install -c conda-forge scikit-surprise pandas numpy scikit-learn matplotlib seaborn -y


## 📂 Repository Structure


├── data/                   # Raw CSV files (ratings.csv, movies.csv)
├── notebooks/              # The Core Project Files
│   ├── 01_eda_and_knn.ipynb   # Part 1: Data Analysis & Baseline Model
│   └── 02_svd_model.ipynb     # Part 2: Advanced Matrix Factorization
├── requirements.txt        # Dependency list
└── README.md               # Project Documentation

> *Authors:* Nakul Jain, Mannat Pal — Chitkara University  
> *Research Paper:* AI-Based Resume Screening and Job Recommendation using Machine Learning  
> *Dataset:* UpdatedResumeDataSet.csv — [Kaggle (jillanisofttech/updated-resume-dataset)](https://www.kaggle.com/datasets/jillanisofttech/updated-resume-dataset)

---

## 📌 Overview

This project implements and compares multiple *Machine Learning algorithms* for automating the resume screening process. Given the massive volume of job applications on modern hiring platforms, manual resume review is inefficient, biased, and unscalable. This research proves that an ML-based approach — specifically a *Support Vector Machine (SVM)* trained on *TF-IDF features* — can classify resumes into the correct job category with *~93% accuracy*, making AI-driven screening a practical alternative to manual recruiting.

---

##  Project Structure


resume-screening/
│
├── resume_screening_real_dataset.ipynb   ← Main research notebook
├── UpdatedResumeDataSet.csv              ← Dataset (download from Kaggle)
├── README.md                             ← This file
│
└── outputs/                              ← Auto-generated when notebook runs
    ├── fig_class_distribution.png
    ├── fig_token_distribution.png
    ├── fig_tfidf_terms.png
    ├── fig_table1_real.png
    ├── fig_metric_bars.png
    ├── fig_training_time.png
    ├── fig_confusion_matrices.png
    ├── fig_roc_curves.png
    ├── fig_cross_validation.png
    └── fig_radar_chart.png


---

## ⚙️ Setup and Installation

### Step 1 — Install Python Dependencies

bash
pip install scikit-learn nltk matplotlib seaborn pandas numpy jupyter


### Step 2 — Download the Dataset

1. Go to: https://www.kaggle.com/datasets/jillanisofttech/updated-resume-dataset
2. Download UpdatedResumeDataSet.csv
3. Place it in the *same folder* as the notebook

### Step 3 — Launch the Notebook

bash
jupyter notebook resume_screening_real_dataset.ipynb


Then click *Kernel → Restart & Run All*

> ⏱️ Total execution time: approximately 2–3 minutes

---

##  Dataset

| Property | Details |
|----------|---------|
| Source | Kaggle — UpdatedResumeDataSet |
| Total Samples | 962 resumes (oversampled to 1000) |
| Job Categories | 25 unique categories |
| Columns | Category, Resume |
| Train / Test Split | 80% / 20% (stratified) |

*Sample categories include:* Data Science, Software Engineering, HR, Marketing, Finance, Mechanical Engineering, Civil Engineering, Sales, Consulting, and more.

---

##  Methodology

### 1. Text Preprocessing
- Lowercase conversion
- URL and special character removal
- Tokenisation (splitting into words)
- Stop-word removal (e.g. "the", "and", "is")

### 2. Feature Extraction — TF-IDF

Converts cleaned resume text into numerical vectors. Words that are important for a specific job category but rare across all resumes receive higher scores.

$$TF\text{-}IDF(t, d) = TF(t,d) \times \log\frac{N}{DF(t)}$$

- *TF(t,d)* = frequency of term t in document d
- *DF(t)* = number of documents containing term t
- *N* = total number of documents

### 3. Models Trained

| Model | Description |
|-------|-------------|
| Logistic Regression | Simple, fast, interpretable baseline |
| Support Vector Machine (SVM) | Powerful for high-dimensional text data |
| Random Forest | Ensemble of decision trees, reduces overfitting |
| Decision Tree | Rule-based classifier, fast but prone to overfitting |

### 4. Evaluation Metrics

- *Accuracy* — Overall correct predictions out of all predictions
- *Precision* — When model predicts a category, how often it is correct
- *Recall* — Out of all actual resumes in a category, how many were found
- *F1-Score* — Harmonic mean of Precision and Recall (best overall metric)
- *Training Time* — Computational cost of fitting the model

---

##  Results

### Performance Comparison (Paper — Table 1)

| Model | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | Training Time (sec) |
|-------|-------------|--------------|-----------|-------------|-------------------|
| *Support Vector Machine* ✅ | *92.8* | *91.9* | *91.2* | *91.5* | 3.89 |
| Random Forest | 88.3 | 87.5 | 87.8 | 89.1 | 5.45 |
| Logistic Regression | 85.4 | 86.7 | 85.9 | 85.3 | 2.16 |
| Decision Tree | 81.7 | 80.2 | 79.6 | 78.9 | 1.65 |

###  Best Model — Support Vector Machine (SVM)

- *Accuracy:* 92.8%
- *F1-Score:* 91.5%
- *Cross-Validation Accuracy:* 93.2% (±0.63% std)
- *Training Time:* 3.89 seconds

SVM achieves the best balance of accuracy and speed, making it the most suitable model for a real-world automated resume screening system.

---

##  Visualisations Generated

The notebook automatically produces and saves the following figures:

| Figure | Description |
|--------|-------------|
| fig_class_distribution.png | Bar chart of resume count per job category |
| fig_token_distribution.png | Histogram of token counts after preprocessing |
| fig_tfidf_terms.png | Top 10 TF-IDF terms for each job category |
| fig_table1_real.png | Styled performance comparison table |
| fig_metric_bars.png | Grouped bar chart — all metrics across all models |
| fig_training_time.png | Line graph of training time (Figure 2 from paper) |
| fig_confusion_matrices.png | 2×2 grid of confusion matrices for all models |
| fig_roc_curves.png | ROC curves (One-vs-Rest) with AUC scores |
| fig_cross_validation.png | 5-fold CV accuracy boxplot |
| fig_radar_chart.png | Radar chart — 360° model comparison |

---

##  Key Findings

1. *SVM is the best model* for resume classification using TF-IDF features due to its ability to find optimal boundaries in high-dimensional text spaces.

2. *Decision Tree overfits* — good on training data but poor generalisation to unseen resumes (lowest F1 at 78.9%).

3. *Speed vs Accuracy tradeoff* — Decision Tree is fastest (1.65s) but least accurate. Random Forest is slowest (5.45s) but still loses to SVM. SVM is the sweet spot.

4. *TF-IDF captures domain vocabulary* well — "tensorflow", "keras" rise to the top for Data Science; "payroll", "onboarding" for HR; "spring boot", "microservices" for Software Engineering.

5. *ML-based screening is fair and consistent* — same evaluation rules applied to every resume with no fatigue, no unconscious bias.

---

##  Future Scope

- Semantic matching using BERT / sentence transformers instead of TF-IDF
- Explainable AI to justify why a resume was shortlisted or rejected
- Real-time job recommendation integration post-screening
- Bias detection and fairness auditing of model predictions

---

##  Libraries Used

| Library | Purpose |
|---------|---------|
| scikit-learn | ML models, TF-IDF, evaluation metrics, cross-validation |
| nltk | Tokenisation, stop-word removal, text preprocessing |
| pandas | Data loading and manipulation |
| numpy | Numerical operations |
| matplotlib | Chart and figure generation |
| seaborn | Statistical visualisation styling |

---

##  Citation


Nakul Jain, Mannat Pal. "AI-Based Resume Screening and Job Recommendation 
using Machine Learning." Chitkara University, 2026.


---

##  License

This project is for academic research purposes only.

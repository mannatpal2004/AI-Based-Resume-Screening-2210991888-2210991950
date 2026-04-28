# 📌 AI-Based Resume Screening and Job Recommendation

---

## 📖 Overview

This project presents an intelligent resume screening system using **Machine Learning and Natural Language Processing (NLP)**.
It automates the classification of resumes into job categories and identifies the most suitable candidates.

The system is supported by a **research paper** and uses a **real-world dataset of 1000 resumes** for experimentation and evaluation.

---

## 🎯 Problem Statement

Manual resume screening is:

* Time-consuming
* Prone to bias
* Not scalable

This project aims to develop an **automated, fair, and efficient system** to classify resumes and assist in recruitment decisions.

---

## 💡 Proposed Solution

We designed a pipeline that:

1. Cleans and preprocesses resume text
2. Converts text into numerical features using **TF-IDF**
3. Applies multiple ML models for classification
4. Compares performance using evaluation metrics
5. Identifies the best model for resume screening

---

## 🧠 Machine Learning Models Used

* Logistic Regression
* Support Vector Machine (SVM) ⭐ *(Best Performing Model)*
* Random Forest
* Decision Tree

---

## 🛠️ Technologies & Tools

* Python (3.10)
* Scikit-learn
* NLTK
* Pandas, NumPy
* Matplotlib, Seaborn
* Jupyter Notebook

---

## 📊 Dataset Information

* Source: Kaggle Resume Dataset
* Total Resumes: **1000**
* Multiple job categories (e.g., Java Developer, Data Science, HR, Testing, etc.)
* Train-Test Split: **80:20**

---

## ⚙️ Methodology

### 🔹 Data Preprocessing

* Lowercasing
* Removal of punctuation & special characters
* Tokenization
* Stopword removal

### 🔹 Feature Extraction

* TF-IDF Vectorization

### 🔹 Model Training

* Multiple supervised ML models trained and compared

### 🔹 Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-Score
* Cross-Validation

---

## 📈 Results & Analysis

### 🔥 Key Findings:

* **SVM achieved the best overall performance**
* Random Forest showed strong stability
* Logistic Regression was fast and efficient
* Decision Tree showed overfitting and lower performance

### 📊 Performance Summary:

| Model               | Accuracy | Precision | Recall | F1 Score |
| ------------------- | -------- | --------- | ------ | -------- |
| Logistic Regression | ~99%     | ~99%      | ~99%   | ~99%     |
| SVM                 | ~99%     | ~99%      | ~99%   | ~99%     |
| Random Forest       | ~99%     | ~99%      | ~99%   | ~99%     |
| Decision Tree       | ~73%     | ~74%      | ~73%   | ~71%     |

---

## 📊 Visualizations Included

This project includes advanced visual analysis:

* 📊 Confusion Matrices
* 📈 ROC Curves
* 📉 Cross-Validation Plot
* 📊 Performance Comparison Charts
* 🧠 TF-IDF Feature Analysis
* 📊 Token Distribution Graph
* ⏱️ Training Time Comparison

---

## 🚀 How to Run the Project

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/resume-screening.git
```

### 2️⃣ Navigate to folder

```bash
cd resume-screening
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the notebook / script

```bash
jupyter notebook
```

OR

```bash
python main.py
```

---

## 📂 Project Structure

```
├── data/
│   └── resume_dataset.csv
├── notebooks/
│   └── resume_screening.ipynb
├── models/
├── outputs/
│   └── graphs & results
├── research_paper.pdf
├── requirements.txt
├── README.md
```

---

## 🧪 Research Contribution

This project contributes by:

* Performing a **comparative study of ML models** on resume classification
* Using **real dataset-based experimentation**
* Applying **multiple validation techniques** (Cross-validation, ROC, Confusion Matrix)
* Providing **visual analytics for deeper insights**

---

## ⚠️ Limitations

* High accuracy due to structured dataset
* May not generalize perfectly to real-world resumes
* Potential bias if dataset is biased

---

## 🔮 Future Scope

* Deep Learning models (BERT, LSTM)
* Real-time job recommendation system
* Web-based deployment
* Explainable AI for transparency

---

## 📄 Research Paper

📎 The complete research paper is included in this repository.

---

## 👨‍💻 Authors

* Nakul Jain
* Mannat Pal

Chitkara University

---

## 📜 License

This project is developed for academic and educational purposes only.

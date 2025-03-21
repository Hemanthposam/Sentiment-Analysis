# Sentiment Analysis - Restaurant Reviews

Project Overview
This project performs **sentiment analysis** on restaurant reviews using **Natural Language Processing (NLP)** techniques.  
It classifies reviews as **Positive** or **Negative** based on customer feedback.

🔧 Technologies Used
- **Python**
- **NLTK (Natural Language Toolkit)**
- **Scikit-learn**
- **Pandas, NumPy**
- **Matplotlib, Seaborn**

 Dataset Information
- The dataset (`liked.tsv`) contains restaurant reviews and their sentiment labels (`0 = Negative, 1 = Positive`).
- The data is preprocessed using **text cleaning, tokenization, and stopword removal**.

Features
- Data Preprocessing:** Cleans and tokenizes text.
- Feature Extraction:** Converts text into numerical vectors using **CountVectorizer**.
- Model Training:** Uses **Naïve Bayes Classifier** for sentiment classification.
- Evaluation:** Displays accuracy, confusion matrix, and classification report.
- Real-time Prediction:** Allows users to input custom reviews for analysis.

Installation & Setup
1️⃣ **Clone the Repository**
```sh
git clone https://github.com/HemanthPosam/Sentiment-Analysis.git
cd Sentiment-Analysis

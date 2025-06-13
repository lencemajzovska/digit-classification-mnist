# Digit Classification with Machine Learning

Educational machine learning project created as part of my Data Science studies. The project focuses on classifying handwritten digits using the MNIST dataset and compares multiple models to identify the best-performing approach.

---

## Project Description

The goal was to build and evaluate machine learning models for digit classification. Three different models were compared: Logistic Regression, Random Forest, and Extra Trees. After hyperparameter tuning, the Extra Trees model showed the best performance and was implemented in a Streamlit application for real-time predictions.

---

## Models Used

- Logistic Regression
- Random Forest
- Extra Trees (selected as the best model)

---

## Technologies and Libraries

- Scikit-learn
- PCA (Principal Component Analysis) for dimensionality reduction
- GridSearchCV for hyperparameter tuning
- Streamlit for interactive web application

---

## Features of the Streamlit App

- Draw a digit directly in the browser
- Real-time model predictions
- Visualization of model confidence (probability)

---

## How to Run the Streamlit App

```bash
streamlit run streamlit/streamlit_app.py
```

## 🌐 Live demo
[Open the Streamlit app](https://ds24ml-8r59cjjwdshqsdstsrp7ig.streamlit.app)


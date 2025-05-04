# CIFAR-10 Classification – Logistic Regression (Assignment 1) 🤖

This project implements and evaluates two multiclass classification approaches on the CIFAR-10 dataset using logistic regression:
- **One-vs-All (OvA)** 🎯
- **Softmax (Multinomial)** 🔄

## Key Components 🔑

- Model training and evaluation using:
  - Accuracy ✅
  - Log Loss (Cost function) 📉
  - F1-Mean 📊
  - Training time ⏱️
- Confusion matrix analysis for OvA 📈
- Custom refinement for classifying between similar classes (3 vs 5) 🎯
- Confidence-based model integration to improve prediction quality 🚀
- Utility function: `testmymodel(model, X_file, y_file)` for model testing on external `.npy` data 🧪

## Dataset 📊

- Features and labels loaded from `.npy` files (CIFAR-10 preprocessed features)
- Train/test split: 70% training, 30% testing

## How to Run 🚀

Run all cells in `ex1.ipynb`.  
Ensure that the `.npy` files are placed in the same directory.

## Author 👥

This project was completed as the **first assignment** in a computational learning course by Amru, Nagham and Abedalrahman.
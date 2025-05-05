# CIFAR-10 Classification – Logistic Regression (Assignment 1) 🤖

This project implements and evaluates two multiclass classification approaches on the CIFAR-10 dataset using logistic regression:
- **One-vs-All (OvA)** 🎯 - Custom implementation from scratch
- **Softmax (Multinomial)** 🔄 - Using scikit-learn's optimized implementation

## Implementation Details 🛠️

### One-vs-All (OvA) Implementation
- Custom implementation using NumPy
- Learning rate: 0.5 (optimized after experimentation)
- Maximum iterations: 900 (prevents overfitting)
- Includes mathematical formulation of gradient descent and sigmoid function
- Enhanced with specialized binary classifier for classes 3 and 5

### Softmax Implementation
- Uses scikit-learn's LogisticRegression
- Solver: 'lbfgs'
- Maximum iterations: 900
- Approximately 8x faster than our OvA implementation due to C/Cython optimization

## Key Components 🔑

- Model training and evaluation metrics:
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
- Features dimension: (n_samples, 16)

## How to Run 🚀

1. Ensure the following files are in the same directory:
   - `ex1.ipynb`
   - `cifar10_features.npy`
   - `cifar10_labels.npy`

2. Run all cells in `ex1.ipynb`

3. To test models, use the `testmymodel` function with either:
   ```python
   # For One-vs-All implementation:
   testmymodel(model_ova, "cifar10_features.npy", "cifar10_labels.npy")
   
   # For Softmax implementation:
   testmymodel(model_softmax, "cifar10_features.npy", "cifar10_labels.npy")
   ```

## Author 👥

This project was completed as the **first assignment** in a computational learning course by Amru, Nagham and Abedalrahman.
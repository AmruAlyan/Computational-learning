# Final Project Report: Tree-Based Learning Algorithms from Scratch
## Computational Learning Course

**Students:** Nagham Mhesen, Amru Alyan, Abdulrahman Muhtaseb
**Course:** Computational Learning  

---

## 1. Introduction and Problem Statement

### 1.1 Project Overview

This project implements three fundamental tree-based machine learning algorithms from first principles: CART Decision Trees, Random Forest, and AdaBoost. The goal is to demonstrate deep understanding of algorithmic foundations while solving a real-world binary classification problem in financial fraud detection.

### 1.2 Dataset: Banknote Authentication

We selected the Banknote Authentication dataset, which represents a binary classification problem in financial security. The dataset contains 1,372 samples with 4 statistical features extracted from Wavelet Transform analysis of banknote images:

- **Variance**: Measures spread of wavelet coefficients - authentic notes show consistent printing patterns
- **Skewness**: Captures asymmetry in coefficient distribution - forgeries often exhibit irregular patterns  
- **Curtosis**: Quantifies tail heaviness - indicates printing quality consistency
- **Entropy**: Information content measure - authentic notes have predictable statistical signatures

**Dataset Characteristics:**
- **Size**: 1,372 samples (762 forged, 610 authentic)
- **Class Balance**: 55.5% forged vs 44.5% authentic (well-balanced)
- **Features**: 4 continuous numerical features
- **Target**: Binary classification (0 = forged, 1 = authentic)

**Why This Dataset is Algorithmically Interesting:**
1. **Non-linear separability** requires sophisticated decision boundaries
2. **Balanced classes** prevent bias-related overfitting
3. **Continuous features** showcase threshold optimization in tree splits
4. **Real-world relevance** demonstrates practical ML application in financial security

---

## 2. Algorithm Implementations

### 2.1 CART Decision Tree (5 points)

**Mathematical Foundation:**

Our decision tree implements greedy recursive binary partitioning using Gini impurity optimization:

**Gini Impurity:**
```
Gini(S) = 1 - Σ(p_i)²
```

**Information Gain:**
```
Information Gain = Gini(parent) - Σ(|S_child|/|S_parent| × Gini(S_child))
```

**Key Implementation Features:**
- **Weighted Gini Impurity**: Essential for AdaBoost integration with sample reweighting
- **Threshold Selection**: Using actual feature values rather than midpoints for natural decision boundaries
- **Multiple Stopping Criteria**: max_depth, min_samples_leaf, and pure node detection
- **Weighted Majority Voting**: Handles sample weights for boosting algorithms

**Algorithm Complexity:**
- **Training**: O(n log n) per feature due to sorting for threshold search
- **Prediction**: O(depth) per sample for tree traversal
- **Space**: O(2^depth) for tree storage

### 2.2 Random Forest (5 points)

**Ensemble Learning Theory:**

Random Forest addresses the bias-variance tradeoff through two key randomization strategies:

**1. Bootstrap Aggregating (Bagging):**
```
For each tree t: Sample D_t ~ Bootstrap(D_original)
```

**2. Feature Subsampling:**
```
At each split: Consider √d features instead of all d features
```

**Key Implementation Features:**
- **Bootstrap Sampling**: Sampling with replacement creates ~63.2% unique observations per tree
- **Feature Subsampling**: √d features per split for decorrelation and computational efficiency
- **Independent Randomization**: Each tree uses different random seeds for diversity
- **Majority Voting**: Final prediction through ensemble consensus

**Theoretical Benefits:**
- **Variance Reduction**: Averaging uncorrelated predictors reduces overall variance
- **Bias Preservation**: Individual trees maintain low bias of full decision trees
- **Robustness**: Resistant to overfitting even with deep trees

### 2.3 AdaBoost (10 points)

**Sequential Learning Theory:**

AdaBoost implements adaptive boosting through iterative sample reweighting and exponential loss minimization:

**Mathematical Foundation:**
```
H(x) = sign(Σ α_t h_t(x))
```

**Exponential Loss Function:**
```
L(y, f(x)) = exp(-y·f(x))
```

**Sample Reweighting Strategy:**
```
w_t+1(i) = w_t(i) × exp(-α_t y_i h_t(x_i))
```

**Weak Learner Weight Calculation:**
```
α_t = (1/2) × log((1-ε_t)/ε_t)
```

**Key Implementation Features:**
- **Decision Stumps**: Single-split trees as weak learners to prevent overfitting
- **Adaptive Focus**: Increases weights on misclassified samples exponentially
- **Numerical Stability**: Robust error handling in logarithmic calculations
- **Early Stopping**: Prevents degenerate cases with perfect weak learners

**Theoretical Guarantees:**
- **Exponential Error Reduction**: Training error decreases exponentially with rounds
- **Margin Maximization**: Encourages confident predictions through exponential loss
- **Weak Learning Assumption**: Each weak learner performs better than random guessing

---

## 3. Experimental Design and Methodology

### 3.1 Data Preprocessing

**Minimal Preprocessing Strategy:**
- **Scale Invariance**: Decision trees split on feature ordering, not absolute values
- **Robustness to Outliers**: Threshold-based splits are naturally robust
- **No Distributional Assumptions**: Trees make no parametric assumptions about data

**Preprocessing Pipeline:**
1. **Missing Value Handling**: Ensures data completeness for algorithmic stability
2. **Random Train/Test Splitting**: 80/20 split with reproducible randomization (seed=42)
3. **Feature-Target Separation**: Explicit type casting for numerical stability

### 3.2 Hyperparameter Configuration

**Decision Tree:**
- `max_depth=6`: Sufficient complexity for feature interactions
- `min_samples_leaf=2`: Prevents overfitting to individual samples
- `max_features=None`: Uses full feature set for optimal single-tree performance

**Random Forest:**
- `n_estimators=100`: Adequate ensemble size for variance reduction
- `max_depth=8`: Deeper trees than single tree (ensemble robustness)
- `max_features='sqrt'`: Optimal feature subsampling for decorrelation

**AdaBoost:**
- `n_estimators=50`: Sufficient boosting rounds for convergence
- Decision stumps: Maximum simplicity for weak learners
- Adaptive weighting: Focuses on difficult examples progressively

### 3.3 Evaluation Metrics

**Comprehensive Binary Classification Framework:**
- **Accuracy**: Overall correctness = (TP+TN)/(TP+TN+FP+FN)
- **Precision**: Positive predictive value = TP/(TP+FP)
- **Recall (Sensitivity)**: True positive rate = TP/(TP+FN)
- **F1-Score**: Harmonic mean of precision/recall = 2·P·R/(P+R)
- **Confusion Matrix**: Detailed error analysis by class

---

## 4. Results and Performance Analysis

### 4.1 Overall Performance Comparison

| Algorithm | Accuracy | Precision | Recall | F1-Score |
|-----------|----------|-----------|--------|----------|
| Decision Tree | 0.978 | 0.976 | 0.976 | 0.976 |
| Random Forest | 0.996 | 1.000 | 0.992 | 0.996 |
| AdaBoost | 0.993 | 0.992 | 0.992 | 0.992 |

### 4.2 Detailed Analysis

**Decision Tree Performance:**
- **Strengths**: Excellent baseline performance (97.8% accuracy), good interpretability
- **Limitations**: Slight overfitting tendency, 3 false positives and 3 false negatives
- **Confusion Matrix**: Balanced error distribution between classes

**Random Forest Performance:**
- **Strengths**: Best overall performance (99.6% accuracy), perfect precision, robust generalization
- **Key Insight**: Only 1 false negative, demonstrating excellent ensemble diversity
- **Theoretical Validation**: Bootstrap and feature subsampling effectively reduce variance

**AdaBoost Performance:**
- **Strengths**: Strong performance (99.3% accuracy), excellent adaptive learning
- **Key Insight**: Only 1 false positive and 1 false negative, showing balanced error reduction
- **Theoretical Validation**: Sequential weak learner training effectively focuses on difficult examples

### 4.3 Algorithmic Insights

**Ensemble Effectiveness:**
- **Variance Reduction**: Random Forest's 99.6% vs Decision Tree's 97.8% demonstrates ensemble benefits
- **Bias Reduction**: AdaBoost's balanced error distribution shows adaptive learning effectiveness
- **Feature Importance**: All algorithms achieve high performance, indicating good feature separability

**Error Pattern Analysis:**
- **Random Forest**: Minimal false negatives (1), critical for fraud detection
- **AdaBoost**: Balanced errors (1 FP, 1 FN), showing no systematic bias
- **Decision Tree**: Slightly higher error rate, demonstrating single-tree limitations

---

## 5. Conclusions and Future Work

### 5.1 Key Achievements

**Algorithmic Mastery:**
- Successfully implemented three fundamental tree-based algorithms from mathematical foundations
- Demonstrated deep understanding of impurity measures, ensemble methods, and boosting theory
- Achieved excellent performance without external ML libraries

**Theoretical Validation:**
- **Bias-Variance Tradeoff**: Random Forest's superior performance validates ensemble theory
- **Adaptive Learning**: AdaBoost's balanced error reduction confirms boosting effectiveness
- **Feature Engineering**: High performance across all algorithms indicates excellent feature design

**Real-World Application:**
- **Financial Security**: Demonstrated practical application in banknote authentication
- **Scalability**: Algorithms handle real-world dataset sizes efficiently
- **Interpretability**: Decision trees provide clear decision boundaries for domain experts

### 5.2 Technical Contributions

**Implementation Innovations:**
- **Weighted Gini Impurity**: Seamless integration with AdaBoost sample reweighting
- **Natural Threshold Selection**: Using actual feature values for interpretable boundaries
- **Numerical Stability**: Robust handling of edge cases in exponential calculations
- **Modular Design**: Clean, extensible code structure for educational purposes

**Performance Validation:**
- **Competitive Results**: All algorithms achieve >97% accuracy on challenging dataset
- **Statistical Rigor**: Comprehensive evaluation with multiple metrics and confusion matrices
- **Reproducibility**: Fixed random seeds ensure consistent experimental results

### 5.3 Future Work

**Algorithmic Extensions:**
- **Gradient Boosting**: Implement XGBoost or LightGBM from scratch
- **Multi-class Classification**: Extend algorithms to handle multiple classes
- **Regression Trees**: Implement CART for continuous target variables

**Practical Applications:**
- **Feature Selection**: Analyze feature importance across ensemble methods
- **Hyperparameter Optimization**: Implement grid search or Bayesian optimization
- **Cross-validation**: Add k-fold cross-validation for more robust evaluation

**Theoretical Investigations:**
- **Margin Analysis**: Study decision margins in AdaBoost for generalization insights
- **Ensemble Diversity**: Quantify tree correlation in Random Forest
- **Convergence Analysis**: Analyze AdaBoost convergence rates with different weak learners

### 5.4 Educational Impact

This project successfully demonstrates the transition from theoretical understanding to practical implementation. By building algorithms from scratch, we gained deep insights into:

- **Mathematical Foundations**: Gini impurity, exponential loss, bootstrap sampling
- **Algorithmic Design**: Recursive partitioning, ensemble methods, sequential learning
- **Implementation Challenges**: Numerical stability, computational efficiency, code organization
- **Evaluation Methodology**: Statistical metrics, error analysis, comparative assessment

The excellent performance achieved (up to 99.6% accuracy) validates both our theoretical understanding and implementation correctness, providing a solid foundation for advanced machine learning studies.


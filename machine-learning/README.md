# Machine Learning Repository

Modular machine learning library with parallel R and Python implementations for classification, regression, and unsupervised methods. Includes clean workflows, reproducible pipelines, and model‑specific subfolders for clear organization and cross‑language comparison.

---

## Folder Structure

```
machine-learning/
    classification/
        r/
        python/
    regression/
        r/
        python/
    unsupervised/
        r/
        python/
```

Each folder contains model‑specific subdirectories. Every model includes:

- R implementation (`.R`)
- Python implementation (`.py`)

---

# Classification

Supervised learning methods for predicting categorical outcomes.

## Classification Models

| Model | R Library | R Function | Python Library | Python Function | When to Use | Why to Use |
|-------|-----------|------------|----------------|------------------|-------------|------------|
| LDA | MASS | lda() | sklearn.discriminant_analysis | LinearDiscriminantAnalysis() | When classes are well-separated and covariance is similar across groups | Fast, interpretable, low-variance classifier with linear boundaries |
| QDA | MASS | qda() | sklearn.discriminant_analysis | QuadraticDiscriminantAnalysis() | When class covariances differ or boundaries are nonlinear | More flexible than LDA; captures curved decision boundaries |
| Logistic Regression | stats | glm(family="binomial") | sklearn.linear_model | LogisticRegression() | When interpretability matters or as a baseline classifier | Coefficients are easy to interpret; strong baseline performance |
| Random Forest | randomForest | randomForest() | sklearn.ensemble | RandomForestClassifier() | When data has nonlinearities, interactions, or noise | Robust, low-tuning, handles high-dimensional data well |
| SVM | e1071 | svm() | sklearn.svm | SVC() | When data is high-dimensional or requires nonlinear kernels | Excellent margin-based classifier; strong performance with kernels |
| XGBoost / GBM | xgboost | xgboost() | xgboost | XGBClassifier() | When maximizing predictive accuracy on structured/tabular data | State-of-the-art boosting; handles complex patterns and interactions |

---

# Regression

Supervised learning methods for predicting continuous outcomes.

## Regression Models

| Model | R Library | R Function | Python Library | Python Function | When to Use | Why to Use |
|-------|-----------|------------|----------------|------------------|-------------|------------|
| Linear Regression | stats | lm() | sklearn.linear_model | LinearRegression() | When relationships are linear and interpretability is needed | Simple, transparent, and statistically grounded |
| Ridge | glmnet | glmnet(alpha=0) | sklearn.linear_model | Ridge() | When predictors are correlated or model overfits | Shrinks coefficients to reduce variance and improve stability |
| Lasso | glmnet | glmnet(alpha=1) | sklearn.linear_model | Lasso() | When feature selection is needed | Drives coefficients to zero, producing sparse models |
| Elastic Net | glmnet | glmnet(alpha=0.5) | sklearn.linear_model | ElasticNet() | When predictors are correlated and selection + shrinkage are needed | Balances Ridge and Lasso for more stable selection |
| PCR | pls | pcr() | sklearn.decomposition + sklearn.linear_model | PCA() + LinearRegression() | When multicollinearity is severe | Regresses on principal components to stabilize estimates |
| PLS | pls | plsr() | sklearn.cross_decomposition | PLSRegression() | When predictors are correlated and linked to outcome | Extracts components that maximize covariance with Y |
| Tree-Based Regression | rpart | rpart() | sklearn.tree | DecisionTreeRegressor() | When relationships are nonlinear or involve interactions | Captures nonlinearities without feature engineering |

---

# Unsupervised Learning

Methods for discovering structure in unlabeled data.

## Unsupervised Models

| Model | R Library | R Function | Python Library | Python Function | When to Use | Why to Use |
|-------|-----------|------------|----------------|------------------|-------------|------------|
| K-Means | stats | kmeans() | sklearn.cluster | KMeans() | When clusters are spherical and well-separated | Fast, scalable, and easy to interpret |
| Hierarchical Clustering | stats | hclust() | scipy.cluster.hierarchy | linkage() | When exploring nested cluster structure | Produces dendrograms for visual cluster exploration |
| DBSCAN | dbscan | dbscan() | sklearn.cluster | DBSCAN() | When clusters are irregularly shaped or contain noise | Identifies arbitrary shapes and isolates outliers |
| PCA | stats | prcomp() | sklearn.decomposition | PCA() | When reducing dimensionality or visualizing structure | Preserves maximum variance in fewer components |
| t-SNE | Rtsne | Rtsne() | sklearn.manifold | TSNE() | When visualizing high-dimensional data in 2D/3D | Reveals local structure and cluster separation |
| UMAP | umap | umap() | umap | UMAP() | When needing fast, stable manifold learning | Faster and more stable than t-SNE; preserves global structure |


---

# 🎯 Design Philosophy

This repository emphasizes:

- **Modularity** — each model is self‑contained  
- **Reproducibility** — consistent workflows across R and Python  
- **Clarity** — minimal folder depth, intuitive organization  
- **Comparability** — parallel implementations for cross‑language benchmarking  


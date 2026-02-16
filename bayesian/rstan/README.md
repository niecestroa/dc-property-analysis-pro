# **README.md — Bayesian Modeling in R & Stan**  
### *Linear & Logistic Regression (Simple + Hierarchical)*  
### *DC Housing Dataset*

---

## **Overview**

This repository contains a complete, modern Bayesian modeling workflow implemented in **R** and **Stan**, using the DC Housing dataset. Four models are included:

1. **Bayesian Linear Regression** (PRICE_10K)  
2. **Hierarchical Bayesian Linear Regression** (random intercept by GRADE)  
3. **Bayesian Logistic Regression** (QUALIFIED_2)  
4. **Hierarchical Bayesian Logistic Regression** (random intercept by GRADE)

All models follow professional Bayesian best practices:

- Standardized predictors  
- Explicit factor encoding  
- Weakly‑informative priors  
- Non‑centered parameterization for hierarchical models  
- Posterior predictive checks  
- WAIC/LOO model comparison  
- Train/test predictive evaluation  
- Clean, modular R + Stan code  

---

# **Why Bayesian?**

Bayesian modeling is not just an alternative to classical statistics — it is a fundamentally different way of reasoning about uncertainty, prediction, and inference. This project uses Bayesian methods because they offer several advantages that are especially important for real‑world housing and classification problems.

### **1. Full uncertainty quantification**
Instead of point estimates, Bayesian models return **full posterior distributions** for:

- coefficients  
- predictions  
- group‑level effects  
- variance components  

This allows richer interpretation and more honest uncertainty reporting.

### **2. Natural regularization**
Weakly‑informative priors stabilize estimates, prevent extreme coefficients, and reduce overfitting — especially important when predictors vary in scale or are correlated.

### **3. Hierarchical modeling done right**
Bayesian methods handle multilevel structure (e.g., GRADE) seamlessly:

- partial pooling  
- shrinkage toward group means  
- non‑centered parameterization for stable sampling  

This improves predictive accuracy and prevents overfitting to small groups.

### **4. Posterior predictive checks**
Bayesian models allow direct simulation of new data from the model:

```text
y_rep ~ p(y | θ)
```

This makes model criticism intuitive and visual.

### **5. Predictive performance matters**
WAIC and LOO provide **fully Bayesian** model comparison metrics that evaluate out‑of‑sample predictive accuracy.

### **6. Interpretability**
Posterior distributions, credible intervals, and shrinkage effects provide a clearer story than p‑values or stepwise selection.

---

# **Repository Structure**

```
/R/
   data_prep_linear.R
   data_prep_logistic.R
   fit_linear.R
   fit_linear_hier.R
   fit_logistic.R
   fit_logistic_hier.R

/stan/
   linear_dc.stan
   linear_dc_hier.stan
   logistic_dc.stan
   logistic_dc_hier.stan

README.md   <-- this file
```

---

# **1. Data Preparation (R)**

All models begin with the same preprocessing pipeline:

- Train/test split (80/20)  
- Standardization of continuous predictors  
- Integer encoding of categorical variables  
- Construction of model matrices  
- Creation of Stan‑ready data lists  

Example (linear regression):

```r
X_train <- as.matrix(train_df %>%
  select(BATHRM_z, ROOMS_z, BEDRM_z, KITCHENS_z,
         FIREPL_z, AYB_z, EYB_z, REM_z))

stan_data_linear <- list(
  N      = nrow(X_train),
  C      = ncol(X_train),
  X      = X_train,
  y      = train_df$PRICE_10K,
  N_test = nrow(X_test),
  X_test = X_test
)
```

Example (logistic regression):

```r
X_train <- as.matrix(train_df %>%
  select(PRICE_100K_z, ROOMS_z, BEDRM_z, STORIES_z,
         KITCHENS_z, FIREPL_z, EYB_z, AC))

stan_data_logit <- list(
  N      = nrow(X_train),
  C      = ncol(X_train),
  X      = X_train,
  y      = ifelse(train_df$QUALIFIED_2 == 1, 1L, 0L),
  N_test = nrow(X_test),
  X_test = X_test
)
```

Hierarchical models add:

```r
GRADE_id = as.integer(train_df$GRADE)
N_GRADE  = length(unique(GRADE_id))
```

---

# **2. Stan Models**

## **2.1 Bayesian Linear Regression (`linear_dc.stan`)**

- Normal likelihood  
- Weakly‑informative priors  
- Posterior predictive draws  
- log_lik for WAIC/LOO  

---

## **2.2 Hierarchical Linear Regression (`linear_dc_hier.stan`)**

Adds:

- Random intercept by GRADE  
- Non‑centered parameterization  

---

## **2.3 Bayesian Logistic Regression (`logistic_dc.stan`)**

- Bernoulli likelihood  
- Weakly‑informative priors  
- Posterior predictive probabilities  

---

## **2.4 Hierarchical Logistic Regression (`logistic_dc_hier.stan`)**

Adds:

- Random intercept by GRADE  
- Non‑centered parameterization  

---

# **3. Model Comparison Table**

| Model | Outcome | Structure | Priors | Random Effects | Use Case | Strengths | Limitations |
|------|---------|-----------|--------|----------------|----------|-----------|-------------|
| **Simple Linear Regression** | PRICE_10K | \( y = X\beta + \epsilon \) | Normal(0, 2.5), Exponential(1) | None | Baseline continuous prediction | Fast, interpretable | Cannot capture group-level variation |
| **Hierarchical Linear Regression** | PRICE_10K | Random intercept by GRADE | Normal(0, 2.5), Exponential(1) | Yes | Price modeling with neighborhood/grade effects | Better fit, partial pooling | More complex, slower |
| **Simple Logistic Regression** | QUALIFIED_2 | \( \text{logit}(p) = X\beta \) | Normal(0, 2.5) | None | Binary classification | Clean, stable | Misses group-level heterogeneity |
| **Hierarchical Logistic Regression** | QUALIFIED_2 | Random intercept by GRADE | Normal(0, 2.5) | Yes | Binary classification with group effects | Best predictive accuracy | Most computationally expensive |

---

# **4. Visual Diagram of the Hierarchical Structure**

```
                         Population Level
                     ┌────────────────────────┐
                     │   Hyperparameters       │
                     │  μ_α (mean intercept)   │
                     │  σ_α (intercept SD)     │
                     └────────────┬───────────┘
                                  │
                                  ▼
                     ┌────────────────────────┐
                     │   GRADE-Level Effects   │
                     │  α_j = μ_α + σ_α * η_j  │
                     │  η_j ~ Normal(0,1)      │
                     └────────────┬───────────┘
                                  │
                                  ▼
                ┌──────────────────────────────────────┐
                │      Observation-Level Model          │
                │                                        │
                │  Linear:   y_i = α_{GRADE[i]} + X_iβ  │
                │  Logistic: logit(p_i)=α_{GRADE[i]}+X_iβ│
                └──────────────────────────────────────┘
```

---

# **5. How to Run the Models (Beginner-Friendly Guide)**

## **Step 1 — Install Packages**

```r
install.packages(c("tidyverse", "rstan", "loo", "bayesplot"))
```

Enable parallel Stan compilation:

```r
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())
```

---

## **Step 2 — Prepare the Data**

```r
source("R/data_prep_linear.R")
source("R/data_prep_logistic.R")
```

---

## **Step 3 — Fit a Model**

### Linear Regression

```r
source("R/fit_linear.R")
```

### Hierarchical Linear Regression

```r
source("R/fit_linear_hier.R")
```

### Logistic Regression

```r
source("R/fit_logistic.R")
```

### Hierarchical Logistic Regression

```r
source("R/fit_logistic_hier.R")
```

---

## **Step 4 — Evaluate Model Fit**

Posterior predictive checks:

```r
ppc_dens_overlay(y_train, posterior$y_rep[1:200, ])
```

WAIC / LOO:

```r
log_lik <- extract_log_lik(fit)
loo(log_lik)
waic(log_lik)
```

---

## **Step 5 — Predict on Test Data**

### Linear RMSE

```r
pred_mean <- apply(posterior$y_pred_test, 2, mean)
rmse <- sqrt(mean((pred_mean - test_df$PRICE_10K)^2))
```

### Logistic Accuracy

```r
pred_class <- ifelse(apply(posterior$p_test, 2, mean) > 0.5, 1, 0)
mean(pred_class == test_df$QUALIFIED_2)
```

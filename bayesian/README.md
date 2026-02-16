# **README.md — Comparing Bayesian Modeling in rjags vs. RStan**  
### *Linear & Logistic Regression (Simple + Hierarchical)*  
### *DC Housing Dataset*

---

## **Overview**

This repository presents a complete Bayesian modeling workflow implemented in **two frameworks**:

- **rjags** (Gibbs sampling via JAGS)  
- **RStan** (Hamiltonian Monte Carlo via Stan)  

Both frameworks are used to fit:

1. **Bayesian Linear Regression** (PRICE_10K)  
2. **Hierarchical Bayesian Linear Regression** (random intercept by GRADE)  
3. **Bayesian Logistic Regression** (QUALIFIED_2)  
4. **Hierarchical Bayesian Logistic Regression** (random intercept by GRADE)

The goal is to demonstrate:

- How the same Bayesian model is implemented in **rjags** vs. **RStan**  
- How modeling choices differ between Gibbs sampling and HMC  
- How hierarchical modeling is handled in each framework  
- The advantages and tradeoffs of each approach  

---

# **Why Compare rjags and RStan?**

Although both tools implement Bayesian models, they differ in philosophy, performance, and workflow.

### **rjags**
- Uses **Gibbs sampling / Metropolis‑within‑Gibbs**  
- Easy to write models  
- Great for teaching and prototyping  
- Slower mixing for correlated parameters  
- Struggles with hierarchical models unless carefully parameterized  

### **RStan**
- Uses **Hamiltonian Monte Carlo (HMC)** and **NUTS**  
- Requires more explicit model structure  
- Much faster mixing  
- Handles hierarchical models extremely well  
- Provides built‑in diagnostics and log‑lik extraction  

This repository shows how your modeling improves when moving from **basic rjags** → **professional rjags** → **RStan**.

---

# **1. Data Preparation (Shared R Workflow)**

Both rjags and RStan use the same preprocessing pipeline:

- Train/test split  
- Standardization of continuous predictors  
- Integer encoding of categorical variables  
- Construction of model matrices  
- Creation of framework‑specific data lists  

This ensures that differences in results come from the **modeling framework**, not the data.

---

# **2. rjags vs. RStan: Model Comparison Table**

| Feature | rjags | RStan |
|--------|--------|--------|
| **Sampling Algorithm** | Gibbs / Metropolis | HMC / NUTS |
| **Speed** | Slower | Much faster |
| **Mixing** | Can struggle with correlated parameters | Excellent mixing |
| **Hierarchical Models** | Requires careful tuning | Natural + stable |
| **Non‑Centered Param.** | Manual, often tricky | Built for it |
| **Diagnostics** | Basic (traceplots, R‑hat) | Full suite (ESS, BFMI, divergences) |
| **Posterior Predictive Checks** | Manual | Built‑in via generated quantities |
| **WAIC / LOO** | Requires custom extraction | Direct via log_lik |
| **Ease of Writing Models** | Very easy | More strict syntax |
| **Best Use Case** | Teaching, simple models | Production‑grade modeling |

---

# **3. Visual Diagram of the Hierarchical Structure**

Both rjags and RStan implement the same hierarchical model:

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

**Key difference:**  
Stan handles the non‑centered parameterization automatically and efficiently.  
JAGS requires more careful tuning to avoid slow mixing.

---

# **4. Model Implementations**

## **4.1 rjags Models**

### Simple Linear Regression
- Normal likelihood  
- Weakly‑informative priors  
- Manual PPC and WAIC extraction  

### Hierarchical Linear Regression
- Random intercept by GRADE  
- Non‑centered parameterization implemented manually  
- Slower mixing than Stan  

### Simple Logistic Regression
- Bernoulli likelihood  
- Weakly‑informative priors  

### Hierarchical Logistic Regression
- Random intercept by GRADE  
- Requires careful tuning to avoid autocorrelation  

---

## **4.2 RStan Models**

### Simple Linear Regression (`linear_dc.stan`)
- Vectorized likelihood  
- Weakly‑informative priors  
- Posterior predictive draws  
- log_lik for WAIC/LOO  

### Hierarchical Linear Regression (`linear_dc_hier.stan`)
- Non‑centered random intercepts  
- Excellent mixing  
- Stable estimation of group‑level variance  

### Simple Logistic Regression (`logistic_dc.stan`)
- Vectorized Bernoulli likelihood  
- Posterior predictive probabilities  

### Hierarchical Logistic Regression (`logistic_dc_hier.stan`)
- Random intercept by GRADE  
- Non‑centered parameterization  
- Best predictive performance  

---

# **5. Performance Comparison (Qualitative)**

| Task | rjags | RStan |
|------|--------|--------|
| **Linear Regression Fit** | Good | Excellent |
| **Logistic Regression Fit** | Good | Excellent |
| **Hierarchical Linear** | Slow, autocorrelation | Fast, stable |
| **Hierarchical Logistic** | Very slow | Fast, reliable |
| **Posterior Predictive Checks** | Manual | Built‑in |
| **WAIC / LOO** | Manual extraction | Direct via log_lik |
| **Interpretability** | High | High |
| **Computation Time** | Longer | Shorter |

---

# **6. How to Run the Models**

## **Step 1 — Install Packages**

```r
install.packages(c("tidyverse", "rjags", "rstan", "loo", "bayesplot"))
```

---

## **Step 2 — Prepare the Data**

```r
source("R/data_prep_linear.R")
source("R/data_prep_logistic.R")
```

---

## **Step 3 — Fit rjags Models**

```r
source("R/fit_rjags_linear.R")
source("R/fit_rjags_linear_hier.R")
source("R/fit_rjags_logistic.R")
source("R/fit_rjags_logistic_hier.R")
```

---

## **Step 4 — Fit RStan Models**

```r
source("R/fit_linear.R")
source("R/fit_linear_hier.R")
source("R/fit_logistic.R")
source("R/fit_logistic_hier.R")
```

---

## **Step 5 — Evaluate Models**

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

# **7. Summary**

This repository demonstrates:

- How the same Bayesian models behave in **rjags vs. RStan**  
- How hierarchical modeling improves predictive performance  
- Why HMC (Stan) is generally superior for complex models  
- How to build a clean, reproducible Bayesian workflow in R  

**rjags** is excellent for teaching and simple models.  
**RStan** is the tool of choice for serious Bayesian modeling.


# **README.md**  
## **Professional‑Grade Bayesian Modeling in R with rjags**  
### *Upgrading an Academic Project into a Modern Bayesian Workflow*

---

## **Abstract**

This repository presents a complete professional‑grade rewrite of a Bayesian regression model originally developed during my graduate studies at **American University**. The initial model used a standard *rjags* workflow typical of academic assignments: flat priors, raw predictors, centered random effects, and minimal diagnostics.  

The revised workflow implements **modern Bayesian best practices**, including hierarchical priors, non‑centered parameterization, standardized predictors, posterior predictive checks, WAIC/LOO model comparison, and a fully modular pipeline separating model specification, fitting, diagnostics, and predictive evaluation.  

This project demonstrates how a basic Bayesian model can be transformed into a **robust, interpretable, and computationally stable hierarchical model** suitable for applied research, clinical analytics, and production‑grade statistical modeling.

---

# **1. Core Improvements Implemented**

These improvements elevate the model from a basic academic example to a **research‑grade Bayesian analysis**.

### **1. Avoided `attach()` / `detach()`**
Replaced with explicit data lists (`jags_data`) to ensure reproducibility and avoid masked variables.

### **2. Explicit numeric JAGS data**
All categorical variables are converted to integer IDs with stored level mappings.

### **3. Centering and scaling predictors**
Standardized predictors improve MCMC mixing, reduce posterior correlations, and make priors interpretable.

### **4. Clean, vectorized model function**
Interactions are precomputed in R, redundant terms removed, and naming conventions clarified.

### **5. Consistent, weakly‑informative priors**
Replaced diffuse priors with hierarchical priors and weakly‑informative variance priors.

### **6. Hierarchical structure with non‑centered parameterization**
Random intercepts rewritten using non‑centered parameterization for better mixing.

### **7. Modular workflow**
Separated into:

- `model.bug`  
- `run_model.R`  
- `diagnostics.R`  
- `ppc.R`  
- `waic_loo.R`  
- `tables_main.R`  
- `tables_appendix.R`  

### **8. Posterior predictive checks (PPCs)**
Added replicated data, Bayesian p‑values, and distributional overlays.

### **9. WAIC and LOO**
Implemented modern Bayesian model comparison metrics.

### **10. Train/test Bayesian prediction**
Posterior predictive distributions used for RMSE and interval coverage.

---

# **2. Visual Workflow Diagram**

```
                ┌────────────────────────┐
                │   Raw Housing Data     │
                └────────────┬───────────┘
                             │
                             ▼
                ┌────────────────────────┐
                │  Data Cleaning & Prep  │
                │  - Factor encoding     │
                │  - Standardization     │
                │  - Interaction terms   │
                └────────────┬───────────┘
                             │
                             ▼
                ┌────────────────────────┐
                │   JAGS Model File      │
                │  - Hierarchical priors │
                │  - Non-centered RE     │
                │  - Weakly informative  │
                └────────────┬───────────┘
                             │
                             ▼
                ┌────────────────────────┐
                │     Model Fitting      │
                │   (run_model.R)        │
                └────────────┬───────────┘
                             │
                             ▼
        ┌──────────────────────────────────────────────┐
        │                Diagnostics                    │
        │  - Traceplots / ESS / R-hat                  │
        │  - Posterior predictive checks (PPC)         │
        │  - WAIC / LOO                                │
        └──────────────────────┬───────────────────────┘
                               │
                               ▼
                ┌────────────────────────┐
                │  Predictive Evaluation │
                │  - Test-set RMSE       │
                │  - Interval coverage   │
                └────────────────────────┘
```

---

# **3. Before vs After: Code Comparison**

## **Before (Basic Academic rjags)**

```r
model {
  for (i in 1:N) {
    mu[i] <- beta0 + beta1 * x[i]
    y[i] ~ dnorm(mu[i], tau)
  }
  beta0 ~ dnorm(0, 0.0001)
  beta1 ~ dnorm(0, 0.0001)
  tau ~ dgamma(0.001, 0.001)
}
```

**Limitations:**

- Flat priors  
- No hierarchical structure  
- No standardization  
- Centered random effects  
- No PPC, WAIC, or LOO  
- Minimal diagnostics  

---

## **After (Professional‑Grade rjags)**

```r
for (i in 1:N) {
  mu[i] <- alpha[GRADE_id[i]] +
           beta[1] * BATHRM_z[i] +
           beta[2] * ROOMS_z[i] +
           beta[3] * BEDRM_z[i] +
           beta[4] * KITCHENS_z[i] +
           beta[5] * FIREPL_z[i] +
           beta[6] * AYB_z[i] +
           beta[7] * EYB_z[i] +
           beta[8] * REM_z[i] +
           beta[9] * COND_id[i] +
           beta[10] * REM_COND_z[i] +
           beta[11] * ROOMS_AYB_z[i] +
           beta[12] * ROOMS_AC[i] +
           beta[13] * BATH_COND_z[i]

  y[i] ~ dnorm(mu[i], tau_y)
  y_rep[i] ~ dnorm(mu[i], tau_y)
  loglik[i] <- logdensity.norm(y[i], mu[i], tau_y)
}

# Hierarchical priors
for (k in 1:K) {
  beta[k] ~ dnorm(mu_beta, tau_beta)
}
mu_beta ~ dnorm(0, 0.001)
tau_beta ~ dgamma(1, 0.1)

# Non-centered random intercepts
for (j in 1:N_GRADE) {
  eta_alpha[j] ~ dnorm(0,1)
  alpha[j] <- mu_alpha + sigma_alpha * eta_alpha[j]
}
```

**Upgrades:**

- Hierarchical priors  
- Non‑centered random effects  
- Standardized predictors  
- Precomputed interactions  
- Posterior predictive checks  
- WAIC + LOO  
- Full diagnostics  
- Modular workflow  

---

# **4. Understanding Hierarchical Priors**

Hierarchical priors introduce **partial pooling**, allowing coefficients to share information:

\[
\beta_k \sim \text{Normal}(\mu_\beta, \tau_\beta)
\]

\[
\mu_\beta \sim \text{Normal}(0, 0.001)
\]

\[
\tau_\beta \sim \text{Gamma}(1, 0.1)
\]

### **Why this matters**

- Stabilizes estimates  
- Reduces overfitting  
- Shrinks extreme coefficients  
- Improves mixing  
- Reflects realistic uncertainty  

Hierarchical priors are a hallmark of **modern Bayesian modeling**.

---

# **5. Understanding Non‑Centered Parameterization**

Centered parameterization:

```r
alpha[j] ~ dnorm(0, tau_alpha)
```

Non‑centered parameterization:

```r
eta_alpha[j] ~ dnorm(0,1)
alpha[j] <- mu_alpha + sigma_alpha * eta_alpha[j]
```

### **Why non‑centered is better**

- Reduces funnel‑shaped posteriors  
- Improves MCMC mixing  
- Avoids divergences  
- Handles weakly identified group effects  
- Standard in hierarchical modeling (Stan, PyMC, Turing, etc.)

This is one of the most important upgrades in the entire rewrite.

---

# **6. Repository Structure**

```
/rjags_pro/
    model.bug
    run_model.R
    diagnostics.R
    ppc.R
    waic_loo.R
    tables_main.R
    tables_appendix.R

/rjags_basic/
    basic_model.bug
    basic_run.R

README.md   <-- this file
```

---

# **7. Purpose of This Repository**

This project demonstrates:

- how to upgrade a basic academic Bayesian model into a **professional‑grade hierarchical model**  
- how to structure a modern Bayesian workflow  
- how to implement diagnostics, PPCs, WAIC/LOO, and predictive evaluation  
- how to write clean, reproducible Bayesian code in R  

It serves as both:

- a **portfolio piece** showcasing applied Bayesian modeling skills  
- a **teaching example** for students learning how to move beyond basic rjags usage

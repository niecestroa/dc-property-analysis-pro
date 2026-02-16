## 1. Add WAIC and LOO from the posterior log‑likelihood

First, we need the **pointwise log‑likelihood** for each observation and posterior draw. Then we can compute WAIC and LOO manually.

### 1.1. Modify the JAGS model to save log‑likelihood

Add this block **inside the `for (i in 1:N)` loop** in your JAGS model:

```r
    # Log-likelihood contribution for WAIC/LOO
    loglik[i] <- logdensity.norm(PRICE_10K[i], mu[i], tau_y)
```

And add `loglik` to the **parameters to monitor**:

```r
params <- c(
  "beta", "mu_beta", "sigma_beta",
  "alpha", "mu_alpha", "sigma_alpha",
  "sigma_y", "s_y",
  "y_rep",
  "loglik"          # NEW: pointwise log-likelihood
)
```

Re‑run `jags(...)` so `final_fit` now contains `loglik`.

---

### 1.2. Extract log‑likelihood and compute WAIC

```r
## Extract log-likelihood matrix: S x N (S = posterior draws, N = obs)
loglik_mat <- final_fit$BUGSoutput$sims.list$loglik  # dimensions: S x N

S <- nrow(loglik_mat)
N <- ncol(loglik_mat)

## Pointwise lppd (log pointwise predictive density)
# lppd_i = log( mean_s exp(loglik_is) )
lppd_i <- apply(loglik_mat, 2, function(x) {
  max_x <- max(x)                         # for numerical stability
  max_x + log(mean(exp(x - max_x)))
})

lppd <- sum(lppd_i)

## Effective number of parameters p_waic
# p_waic = sum_i Var_s(loglik_is)
p_waic <- sum(apply(loglik_mat, 2, var))

## WAIC
waic <- -2 * (lppd - p_waic)

cat("WAIC:", round(waic, 1), "\n")
cat("Effective number of parameters (p_waic):", round(p_waic, 1), "\n")
```

---

### 1.3. Approximate LOO (PSIS‑LOO style, simple version)

A full PSIS‑LOO is more involved, but a simple LOO approximation using importance weights is:

```r
## Compute log-weights for each observation and draw ------------------------
# logw_is = loglik_is - max_s(loglik_is) for stability
logw <- apply(loglik_mat, 2, function(x) x - max(x))

## Convert to normalized importance weights per observation -----------------
w <- apply(logw, 2, function(x) {
  w_raw <- exp(x)
  w_raw / sum(w_raw)
})

## LOO log predictive density per observation ------------------------------
# loo_i = log( sum_s w_is * exp(loglik_is) )
loo_i <- numeric(N)
for (i in 1:N) {
  loo_i[i] <- log(sum(w[, i] * exp(loglik_mat[, i])))
}

loo_lpd <- sum(loo_i)
p_loo   <- lppd - loo_lpd
looic   <- -2 * loo_lpd

cat("LOOIC:", round(looic, 1), "\n")
cat("Effective number of parameters (p_loo):", round(p_loo, 1), "\n")
```

This is a **simplified LOO**; if you want full PSIS‑LOO, we’d typically move to `loo` + Stan, but this is already a strong Bayesian story for a class/report.

---

## 2. Compact summary table: priors, posteriors, interpretation

Let’s build a **report‑ready table** for your main parameters:  
hyperparameters, variance components, and a few key β’s.

### 2.1. Extract posterior summaries

```r
## Helper to summarize a vector of posterior draws --------------------------
summarize_posterior <- function(x) {
  c(
    mean  = mean(x),
    sd    = sd(x),
    q2.5  = quantile(x, 0.025),
    q50   = quantile(x, 0.5),
    q97.5 = quantile(x, 0.975)
  )
}

sims <- final_fit$BUGSoutput$sims.list

## Example: summarize key parameters ----------------------------------------
post_mu_beta    <- summarize_posterior(sims$mu_beta)
post_sigma_beta <- summarize_posterior(sims$sigma_beta)
post_sigma_y    <- summarize_posterior(sims$sigma_y)
post_sigma_alpha<- summarize_posterior(sims$sigma_alpha)

## Pick a few key betas to report (e.g., BATHRM, ROOMS, REM, interactions)
beta_names <- c(
  "BATHRM_z", "ROOMS_z", "BEDRM_z", "KITCHENS_z",
  "FIREPL_z", "AYB_z", "EYB_z", "REM_z",
  "COND_id", "REM_COND_z", "ROOMS_AYB_z", "ROOMS_AC", "BATH_COND_z"
)

beta_summaries <- t(apply(sims$beta, 2, summarize_posterior))
rownames(beta_summaries) <- beta_names
```

---

### 2.2. Build a compact table with priors, posteriors, interpretation

We’ll create a **data frame** you can drop straight into your report (via `knitr::kable`, `gt`, etc.).

```r
## Define prior descriptions manually ---------------------------------------
prior_desc <- tibble::tibble(
  Parameter = c(
    "mu_beta", "sigma_beta",
    "sigma_y", "sigma_alpha",
    beta_names
  ),
  Prior = c(
    "Normal(0, 0.001^-0.5)",     # mu_beta
    "Gamma(1, 0.1) on precision -> sigma_beta",  # sigma_beta
    "Gamma(1, 0.1) on precision -> sigma_y",     # sigma_y
    "Gamma(1, 0.1) on precision -> sigma_alpha", # sigma_alpha
    rep("Hierarchical Normal(mu_beta, tau_beta)", length(beta_names))
  )
)

## Posterior summary rows ---------------------------------------------------
post_core <- rbind(
  post_mu_beta,
  post_sigma_beta,
  post_sigma_y,
  post_sigma_alpha,
  beta_summaries
)

post_df <- as.data.frame(post_core)
post_df$Parameter <- rownames(post_df)

## Join priors and posteriors -----------------------------------------------
summary_table <- prior_desc %>%
  dplyr::left_join(post_df, by = "Parameter") %>%
  dplyr::select(
    Parameter, Prior,
    mean, sd, q2.5, q50, q97.5
  )
```

Now add a **short interpretation column**—you can tune the wording, but here’s a template:

```r
## Add interpretation text (edit as needed for your narrative) --------------
interpretation <- c(
  "Average effect size across predictors (hyper-mean).",
  "Between-predictor variability (hyper-SD of betas).",
  "Residual SD of house prices (in 10K units, after covariates).",
  "Between-GRADE SD of random intercepts.",
  "Effect of bathrooms (standardized) on price.",
  "Effect of rooms (standardized) on price.",
  "Effect of bedrooms (standardized) on price.",
  "Effect of kitchens (standardized) on price.",
  "Effect of fireplaces (standardized) on price.",
  "Effect of AYB age (standardized) on price.",
  "Effect of EYB age (standardized) on price.",
  "Effect of remodel age (standardized) on price.",
  "Main effect of condition category (coded integer).",
  "Interaction: remodel age × condition.",
  "Interaction: rooms × AYB age.",
  "Interaction: rooms × AC.",
  "Interaction: bathrooms × condition."
)

summary_table$Interpretation <- interpretation

## Optional: round for display ----------------------------------------------
summary_table_display <- summary_table %>%
  mutate(
    dplyr::across(
      c(mean, sd, q2.5, q50, q97.5),
      ~ round(.x, 3)
    )
  )

summary_table_display
```

In your R Markdown, you can render this as:

```r
knitr::kable(summary_table_display, caption = "Bayesian model priors, posteriors, and interpretations.")
```

# Appendix
**Organizing summary table**

### 1. Trimmed table for the main text

Pick the parameters that tell the clearest story:

- Overall variability terms  
- A few core main effects  
- Only the most interpretable interactions  

```r
## Choose a subset of parameters for the main text --------------------------
main_params <- c(
  "sigma_y",        # residual SD
  "sigma_alpha",    # between-GRADE SD
  "BATHRM_z",       # bathrooms
  "ROOMS_z",        # rooms
  "BEDRM_z",        # bedrooms
  "KITCHENS_z",     # kitchens
  "FIREPL_z",       # fireplaces
  "AYB_z",          # age of building
  "REM_z",          # remodel age
  "COND_id",        # condition main effect
  "REM_COND_z",     # remodel × condition
  "ROOMS_AYB_z"     # rooms × AYB
)

summary_main <- summary_table %>%
  dplyr::filter(Parameter %in% main_params) %>%
  dplyr::mutate(
    dplyr::across(
      c(mean, sd, q2.5, q50, q97.5),
      ~ round(.x, 3)
    )
  )

## In your R Markdown:
knitr::kable(
  summary_main,
  caption = "Key Bayesian parameter estimates: priors, posteriors, and interpretations."
)
```

This gives you a **tight, readable table** that supports your narrative without overwhelming the reader.

---

### 2. Detailed appendix table (all betas + random effects)

For the appendix, you can expose:

- All β coefficients  
- Hyperparameters  
- Variance components  
- Optionally, summaries of the random intercepts `alpha[j]`  

#### 2.1. All β’s + variance components + hyperparameters

```r
## Appendix table: all betas + core variance/hyperparameters ---------------
appendix_params <- c(
  "mu_beta", "sigma_beta",
  "sigma_y", "sigma_alpha",
  beta_names   # all regression coefficients
)

summary_appendix <- summary_table %>%
  dplyr::filter(Parameter %in% appendix_params) %>%
  dplyr::mutate(
    dplyr::across(
      c(mean, sd, q2.5, q50, q97.5),
      ~ round(.x, 3)
    )
  )

knitr::kable(
  summary_appendix,
  caption = "Full Bayesian parameter summary (hyperparameters, variance components, and all regression coefficients)."
)
```

#### 2.2. Optional: random intercepts by GRADE

If you want to show how **GRADE‑specific intercepts** vary, you can summarize `alpha[j]` as well:

```r
## Summarize random intercepts alpha[j] by GRADE ----------------------------
alpha_mat <- sims$alpha   # S x N_GRADE

alpha_summaries <- t(apply(alpha_mat, 2, summarize_posterior))
alpha_df <- as.data.frame(alpha_summaries)

alpha_df$GRADE_level <- levels(train_df$GRADE)[sort(unique(train_df$GRADE_id))]
alpha_df$Parameter   <- paste0("alpha[", alpha_df$GRADE_level, "]")

alpha_df_display <- alpha_df %>%
  dplyr::mutate(
    dplyr::across(
      c(mean, sd, q2.5, q50, q97.5),
      ~ round(.x, 3)
    )
  ) %>%
  dplyr::select(
    Parameter, GRADE_level,
    mean, sd, q2.5, q50, q97.5
  )

knitr::kable(
  alpha_df_display,
  caption = "Posterior summaries of GRADE-specific random intercepts."
)
```

# Trimmed Table for the Main Text  
This table highlights only the **most interpretable, scientifically meaningful effects** — the ones a reader can understand without digging into the full model structure.

### Code to generate the trimmed table

```r
## Choose a subset of parameters for the main text --------------------------
main_params <- c(
  "sigma_y",        # residual SD
  "sigma_alpha",    # between-GRADE SD
  "BATHRM_z",       # bathrooms
  "ROOMS_z",        # rooms
  "BEDRM_z",        # bedrooms
  "KITCHENS_z",     # kitchens
  "FIREPL_z",       # fireplaces
  "AYB_z",          # age of building
  "REM_z",          # remodel age
  "COND_id",        # condition main effect
  "REM_COND_z",     # remodel × condition
  "ROOMS_AYB_z"     # rooms × AYB
)

summary_main <- summary_table %>%
  dplyr::filter(Parameter %in% main_params) %>%
  dplyr::mutate(
    dplyr::across(
      c(mean, sd, q2.5, q50, q97.5),
      ~ round(.x, 3)
    )
  )

knitr::kable(
  summary_main,
  caption = "Key Bayesian parameter estimates: priors, posteriors, and interpretations."
)
```

### What this table accomplishes  
- Gives the reader a **digestible overview** of the model  
- Focuses on **main effects** and **major interactions**  
- Includes **variance components**, which are essential in hierarchical Bayesian models  
- Avoids overwhelming the main text with technical detail  

This is exactly what a polished Bayesian report should do.

---

# Full Appendix Table (All β’s + Hyperparameters + Variance Components)

This table is for readers who want the **complete statistical picture**. It includes:

- All regression coefficients  
- Hyperparameters (`mu_beta`, `sigma_beta`)  
- Variance components (`sigma_y`, `sigma_alpha`)  

### Code to generate the appendix table

```r
## Appendix table: all betas + core variance/hyperparameters ---------------
appendix_params <- c(
  "mu_beta", "sigma_beta",
  "sigma_y", "sigma_alpha",
  beta_names   # all regression coefficients
)

summary_appendix <- summary_table %>%
  dplyr::filter(Parameter %in% appendix_params) %>%
  dplyr::mutate(
    dplyr::across(
      c(mean, sd, q2.5, q50, q97.5),
      ~ round(.x, 3)
    )
  )

knitr::kable(
  summary_appendix,
  caption = "Full Bayesian parameter summary (hyperparameters, variance components, and all regression coefficients)."
)
```

### Why this belongs in the appendix  
- It documents the **entire posterior structure**  
- It supports reproducibility  
- It allows a technical reader to evaluate shrinkage, uncertainty, and effect sizes  
- It keeps the main text clean and readable  

---

# 3. Optional Appendix Table: Random Intercepts by GRADE  
If you want to show how the hierarchical structure behaves, include this too.

### Code

```r
## Summarize random intercepts alpha[j] by GRADE ----------------------------
alpha_mat <- sims$alpha   # S x N_GRADE

alpha_summaries <- t(apply(alpha_mat, 2, summarize_posterior))
alpha_df <- as.data.frame(alpha_summaries)

alpha_df$GRADE_level <- levels(train_df$GRADE)[sort(unique(train_df$GRADE_id))]
alpha_df$Parameter   <- paste0("alpha[", alpha_df$GRADE_level, "]")

alpha_df_display <- alpha_df %>%
  dplyr::mutate(
    dplyr::across(
      c(mean, sd, q2.5, q50, q97.5),
      ~ round(.x, 3)
    )
  ) %>%
  dplyr::select(
    Parameter, GRADE_level,
    mean, sd, q2.5, q50, q97.5
  )

knitr::kable(
  alpha_df_display,
  caption = "Posterior summaries of GRADE-specific random intercepts."
)
```

### Why include it  
- Shows how much **partial pooling** occurs  
- Demonstrates the hierarchical structure in action  
- Helps interpret differences between GRADE categories  


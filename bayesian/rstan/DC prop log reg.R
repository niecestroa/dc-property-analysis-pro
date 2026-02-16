# This replaces all the long, slow, messy code you had before.  
# It is modular, efficient, statistically correct, and ready for publication or portfolio use.

# **PROFESSIONAL‑GRADE BAYESIAN LOGISTIC REGRESSION (R + STAN)**  
### Outcome: **QUALIFIED_2** (0/1)  
### Predictors: Selected via Bayesian regularization (ridge, horseshoe, lasso)

# You already performed Bayesian variable selection using `bayesreg`, and the final chosen predictors were:

# - PRICE_100K  
# - ROOMS  
# - BEDRM  
# - STORIES  
# - KITCHENS  
# - FIREPLACES  
# - EYB.age  
# - AC  

---

# 1. **R Code — Preprocessing + Stan Data Construction**

This is the clean, modern version of your logistic preprocessing pipeline.

```r
library(tidyverse)
library(rstan)
library(loo)
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

#--------------------------------------------------
# 1. Load cleaned dataset
#--------------------------------------------------
df <- DC_data   # your cleaned dataset

#--------------------------------------------------
# 2. Train/test split
#--------------------------------------------------
set.seed(400)
n <- nrow(df)
idx <- sample(n, size = 0.8 * n)

train_df <- df[idx, ]
test_df  <- df[-idx, ]

#--------------------------------------------------
# 3. Standardize continuous predictors
#--------------------------------------------------
standardize <- function(x) {
  m <- mean(x)
  s <- sd(x)
  list(z = (x - m) / s, mean = m, sd = s)
}

std_list <- list(
  PRICE_100K = standardize(train_df$PRICE_100K),
  ROOMS      = standardize(train_df$ROOMS),
  BEDRM      = standardize(train_df$BEDRM),
  STORIES    = standardize(train_df$STORIES),
  KITCHENS   = standardize(train_df$KITCHENS),
  FIREPL     = standardize(train_df$FIREPLACES),
  EYB        = standardize(train_df$EYB.age)
)

train_df <- train_df %>%
  mutate(
    PRICE_100K_z = std_list$PRICE_100K$z,
    ROOMS_z      = std_list$ROOMS$z,
    BEDRM_z      = std_list$BEDRM$z,
    STORIES_z    = std_list$STORIES$z,
    KITCHENS_z   = std_list$KITCHENS$z,
    FIREPL_z     = std_list$FIREPL$z,
    EYB_z        = std_list$EYB$z
  )

# Apply same scaling to test set
test_df <- test_df %>%
  mutate(
    PRICE_100K_z = (PRICE_100K - std_list$PRICE_100K$mean) / std_list$PRICE_100K$sd,
    ROOMS_z      = (ROOMS      - std_list$ROOMS$mean)      / std_list$ROOMS$sd,
    BEDRM_z      = (BEDRM      - std_list$BEDRM$mean)      / std_list$BEDRM$sd,
    STORIES_z    = (STORIES    - std_list$STORIES$mean)    / std_list$STORIES$sd,
    KITCHENS_z   = (KITCHENS   - std_list$KITCHENS$mean)   / std_list$KITCHENS$sd,
    FIREPL_z     = (FIREPLACES - std_list$FIREPL$mean)     / std_list$FIREPL$sd,
    EYB_z        = (EYB.age    - std_list$EYB$mean)        / std_list$EYB$sd
  )

#--------------------------------------------------
# 4. Build model matrix for logistic regression
#--------------------------------------------------
X_train <- as.matrix(train_df %>%
  select(PRICE_100K_z, ROOMS_z, BEDRM_z, STORIES_z,
         KITCHENS_z, FIREPL_z, EYB_z, AC))

X_test <- as.matrix(test_df %>%
  select(PRICE_100K_z, ROOMS_z, BEDRM_z, STORIES_z,
         KITCHENS_z, FIREPL_z, EYB_z, AC))

y_train <- ifelse(train_df$QUALIFIED_2 == 1, 1L, 0L)

#--------------------------------------------------
# 5. Build Stan data list
#--------------------------------------------------
stan_data_logit <- list(
  N      = nrow(X_train),
  C      = ncol(X_train),
  X      = X_train,
  y      = y_train,
  N_test = nrow(X_test),
  X_test = X_test
)
```

---

# 2. **STAN MODEL — Bayesian Logistic Regression (Improved)**  
# Save this as: **`logistic_dc.stan`**

# This is the clean, modern version with:

#- weakly‑informative priors  
#- vectorized likelihood  
#- posterior predictive checks  
#- log_lik for WAIC/LOO  
#- test‑set predictions  

```stan
data {
  int<lower=1> N;              // number of training observations
  int<lower=1> C;              // number of predictors
  matrix[N, C] X;              // training predictors
  int<lower=0, upper=1> y[N];  // binary outcome

  int<lower=1> N_test;         // number of test observations
  matrix[N_test, C] X_test;    // test predictors
}

parameters {
  real intercept;
  vector[C] beta;
}

transformed parameters {
  vector[N] eta;
  vector[N] p;

  eta = X * beta + intercept;
  p   = inv_logit(eta);
}

model {
  // Weakly-informative priors
  intercept ~ normal(0, 2.5);
  beta      ~ normal(0, 2.5);

  // Likelihood
  y ~ bernoulli(p);
}

generated quantities {
  vector[N] log_lik;
  int y_rep[N];
  vector[N_test] p_test;
  int y_rep_test[N_test];

  for (n in 1:N) {
    log_lik[n] = bernoulli_lpmf(y[n] | p[n]);
    y_rep[n]   = bernoulli_rng(p[n]);
  }

  for (n in 1:N_test) {
    real eta_test = dot_product(X_test[n], beta) + intercept;
    p_test[n]     = inv_logit(eta_test);
    y_rep_test[n] = bernoulli_rng(p_test[n]);
  }
}
```

---

# 3. **R Code — Fit the Stan Model**

```r
fit_logit <- stan(
  file = "logistic_dc.stan",
  data = stan_data_logit,
  chains = 4,
  iter = 2000,
  warmup = 1000,
  seed = 400
)

print(fit_logit, pars = c("intercept", "beta"))
```

---

# 4. **Posterior Predictive Checks**

```r
posterior <- rstan::extract(fit_logit)

y_rep <- posterior$y_rep
ppc_dens_overlay(y_train, y_rep[1:200, ])
```

---

# 5. **WAIC / LOO**

```r
log_lik <- extract_log_lik(fit_logit)
loo_res  <- loo(log_lik)
waic_res <- waic(log_lik)

loo_res
waic_res
```

---

# 6. **Test‑Set Prediction Performance**

```r
p_test <- posterior$p_test
pred_mean <- apply(p_test, 2, mean)

pred_class <- ifelse(pred_mean > 0.5, 1, 0)
actual     <- test_df$QUALIFIED_2

accuracy <- mean(pred_class == actual)
accuracy
```

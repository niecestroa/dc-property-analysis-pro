# Below is the **fully improved, modern, clean, and modular Bayesian linear regression workflow**, using:

# - **PRICE_10K** as the continuous outcome  
# - **RStan** for fitting  
# - **Weakly‑informative priors**  
# - **Standardized predictors**  
# - **Posterior predictive checks**  
# - **WAIC/LOO**  
# - **Train/test prediction**  
# - **Clear comments explaining every step**  

---

# **PROFESSIONAL‑GRADE BAYESIAN LINEAR REGRESSION (R + STAN)**  
### Outcome: **PRICE_10K**  
### Predictors: same cleaned DC dataset you used earlier

---

# 1. **R Code — Preprocessing + Stan Data Construction**

This is the clean, modern version of your preprocessing pipeline.

```r
library(tidyverse)
library(rstan)
library(loo)
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

#--------------------------------------------------
# 1. Load cleaned dataset
#--------------------------------------------------
df <- DC_Final   # from your earlier cleaning pipeline

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
  BATHRM   = standardize(train_df$BATHRM),
  ROOMS    = standardize(train_df$ROOMS),
  BEDRM    = standardize(train_df$BEDRM),
  KITCHENS = standardize(train_df$KITCHENS),
  FIREPL   = standardize(train_df$FIREPLACES),
  AYB      = standardize(train_df$AYB.age),
  EYB      = standardize(train_df$EYB.age),
  REM      = standardize(train_df$REMODEL.age)
)

train_df <- train_df %>%
  mutate(
    BATHRM_z   = std_list$BATHRM$z,
    ROOMS_z    = std_list$ROOMS$z,
    BEDRM_z    = std_list$BEDRM$z,
    KITCHENS_z = std_list$KITCHENS$z,
    FIREPL_z   = std_list$FIREPL$z,
    AYB_z      = std_list$AYB$z,
    EYB_z      = std_list$EYB$z,
    REM_z      = std_list$REM$z
  )

# Apply same scaling to test set
test_df <- test_df %>%
  mutate(
    BATHRM_z   = (BATHRM   - std_list$BATHRM$mean)   / std_list$BATHRM$sd,
    ROOMS_z    = (ROOMS    - std_list$ROOMS$mean)    / std_list$ROOMS$sd,
    BEDRM_z    = (BEDRM    - std_list$BEDRM$mean)    / std_list$BEDRM$sd,
    KITCHENS_z = (KITCHENS - std_list$KITCHENS$mean) / std_list$KITCHENS$sd,
    FIREPL_z   = (FIREPLACES - std_list$FIREPL$mean) / std_list$FIREPL$sd,
    AYB_z      = (AYB.age  - std_list$AYB$mean)      / std_list$AYB$sd,
    EYB_z      = (EYB.age  - std_list$EYB$mean)      / std_list$EYB$sd,
    REM_z      = (REMODEL.age - std_list$REM$mean)   / std_list$REM$sd
  )

#--------------------------------------------------
# 4. Build model matrix for linear regression
#--------------------------------------------------
X_train <- as.matrix(train_df %>%
  select(BATHRM_z, ROOMS_z, BEDRM_z, KITCHENS_z,
         FIREPL_z, AYB_z, EYB_z, REM_z))

X_test <- as.matrix(test_df %>%
  select(BATHRM_z, ROOMS_z, BEDRM_z, KITCHENS_z,
         FIREPL_z, AYB_z, EYB_z, REM_z))

y_train <- train_df$PRICE_10K

#--------------------------------------------------
# 5. Build Stan data list
#--------------------------------------------------
stan_data_linear <- list(
  N      = nrow(X_train),
  C      = ncol(X_train),
  X      = X_train,
  y      = y_train,
  N_test = nrow(X_test),
  X_test = X_test
)
```

---

# 2. **STAN MODEL — Bayesian Linear Regression (Improved)**  
Save this as: **`linear_dc.stan`**

This is the clean, modern version with:

- weakly‑informative priors  
- vectorized likelihood  
- posterior predictive checks  
- log_lik for WAIC/LOO  
- test‑set predictions  

```stan
data {
  int<lower=1> N;              // number of training observations
  int<lower=1> C;              // number of predictors
  matrix[N, C] X;              // training predictors
  vector[N] y;                 // continuous outcome PRICE_10K

  int<lower=1> N_test;         // number of test observations
  matrix[N_test, C] X_test;    // test predictors
}

parameters {
  real intercept;
  vector[C] beta;
  real<lower=0> sigma;
}

transformed parameters {
  vector[N] mu;
  mu = X * beta + intercept;
}

model {
  // Weakly-informative priors
  intercept ~ normal(0, 2.5);
  beta      ~ normal(0, 2.5);
  sigma     ~ exponential(1);

  // Likelihood
  y ~ normal(mu, sigma);
}

generated quantities {
  vector[N] log_lik;
  vector[N] y_rep;
  vector[N_test] y_pred_test;

  for (n in 1:N) {
    log_lik[n] = normal_lpdf(y[n] | mu[n], sigma);
    y_rep[n]   = normal_rng(mu[n], sigma);
  }

  for (n in 1:N_test) {
    real mu_test = dot_product(X_test[n], beta) + intercept;
    y_pred_test[n] = normal_rng(mu_test, sigma);
  }
}
```

---

# 3. **R Code — Fit the Stan Model**

```r
fit_linear <- stan(
  file = "linear_dc.stan",
  data = stan_data_linear,
  chains = 4,
  iter = 2000,
  warmup = 1000,
  seed = 400
)

print(fit_linear, pars = c("intercept", "beta", "sigma"))
```

---

# 4. **Posterior Predictive Checks**

```r
posterior <- rstan::extract(fit_linear)

y_rep <- posterior$y_rep
ppc_dens_overlay(y_train, y_rep[1:200, ])
```

---

# 5. **WAIC / LOO**

```r
log_lik <- extract_log_lik(fit_linear)
loo_res  <- loo(log_lik)
waic_res <- waic(log_lik)

loo_res
waic_res
```

---

# 6. **Test‑Set Prediction Performance**

```r
y_pred <- posterior$y_pred_test
pred_mean <- apply(y_pred, 2, mean)

rmse <- sqrt(mean((pred_mean - test_df$PRICE_10K)^2))
rmse
```




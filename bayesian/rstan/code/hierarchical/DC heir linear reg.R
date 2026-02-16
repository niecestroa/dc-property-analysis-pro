### R: Stan data for hierarchical linear regression

# Reuse the `DC_Final` and standardization from your linear model, but add `GRADE_id`:

```r
df <- DC_Final %>% mutate(GRADE_id = as.integer(GRADE))
N_GRADE <- length(unique(df$GRADE_id))

set.seed(400)
n  <- nrow(df)
idx <- sample(n, size = 0.8 * n)
train_df <- df[idx, ]
test_df  <- df[-idx, ]

# standardization as before for BATHRM_z, ROOMS_z, etc.

X_train <- as.matrix(train_df %>%
  select(BATHRM_z, ROOMS_z, BEDRM_z, KITCHENS_z,
         FIREPL_z, AYB_z, EYB_z, REM_z))
X_test <- as.matrix(test_df %>%
  select(BATHRM_z, ROOMS_z, BEDRM_z, KITCHENS_z,
         FIREPL_z, AYB_z, EYB_z, REM_z))

y_train     <- train_df$PRICE_10K
GRADE_train <- train_df$GRADE_id
GRADE_test  <- test_df$GRADE_id

stan_data_linear_hier <- list(
  N         = nrow(X_train),
  C         = ncol(X_train),
  X         = X_train,
  y         = y_train,
  N_GRADE   = N_GRADE,
  grade_id  = GRADE_train,
  N_test    = nrow(X_test),
  X_test    = X_test,
  grade_test = GRADE_test
)
```

---

### Stan: hierarchical linear regression with non‑centered random intercepts

# Save as `linear_dc_hier.stan`:

```stan
data {
  int<lower=1> N;
  int<lower=1> C;
  matrix[N, C] X;
  vector[N] y;

  int<lower=1> N_GRADE;
  int<lower=1, upper=N_GRADE> grade_id[N];

  int<lower=1> N_test;
  matrix[N_test, C] X_test;
  int<lower=1, upper=N_GRADE> grade_test[N_test];
}

parameters {
  real intercept;
  vector[C] beta;
  real<lower=0> sigma;

  real mu_alpha;
  real<lower=0> sigma_alpha;
  vector[N_GRADE] eta_alpha;
}

transformed parameters {
  vector[N_GRADE] alpha;
  vector[N] mu;

  alpha = mu_alpha + sigma_alpha * eta_alpha;

  for (n in 1:N) {
    mu[n] = intercept + alpha[grade_id[n]] + X[n] * beta;
  }
}

model {
  // Priors
  intercept   ~ normal(0, 2.5);
  beta        ~ normal(0, 2.5);
  sigma       ~ exponential(1);

  mu_alpha    ~ normal(0, 2.5);
  sigma_alpha ~ exponential(1);
  eta_alpha   ~ normal(0, 1);

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
    real mu_t = intercept + alpha[grade_test[n]] + X_test[n] * beta;
    y_pred_test[n] = normal_rng(mu_t, sigma);
  }
}
```

```r
fit_linear_hier <- stan(
  file = "linear_dc_hier.stan",
  data = stan_data_linear_hier,
  chains = 4, iter = 2000, warmup = 1000, seed = 400
)
```

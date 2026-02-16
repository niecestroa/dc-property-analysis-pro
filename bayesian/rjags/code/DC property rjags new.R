### 1. Libraries and seed

```r
## Load required packages
library(tidyverse)   # data wrangling
library(R2jags)      # interface to JAGS
library(coda)        # MCMC diagnostics
library(superdiag)   # additional convergence diagnostics

set.seed(10000000)   # reproducibility
```

---

### 2. Train/test split and Bayesian‑ready preprocessing

```r
## Train/test split ---------------------------------------------------------
n <- nrow(DC_Final)                     # total number of observations
Z <- sample(n, n/2)                     # random half for training

train_df <- DC_Final[Z, ]
test_df  <- DC_Final[-Z, ]

## Encode factors as integers for JAGS --------------------------------------
# Store mappings so we can interpret later
train_df <- train_df %>%
  mutate(
    GRADE_id     = as.integer(GRADE),       # random intercept grouping
    COND_id      = as.integer(CONDITION),   # condition factor
    AC_id        = as.integer(AC)           # AC factor
  )

test_df <- test_df %>%
  mutate(
    GRADE_id     = as.integer(GRADE),
    COND_id      = as.integer(CONDITION),
    AC_id        = as.integer(AC)
  )

## Standardize continuous predictors for MCMC stability ---------------------
# Function to standardize and store mean/sd for later use on test set
standardize <- function(x) {
  m <- mean(x)
  s <- sd(x)
  list(
    z = (x - m) / s,
    mean = m,
    sd = s
  )
}

# Apply to key continuous predictors
std_BATHRM   <- standardize(train_df$BATHRM)
std_ROOMS    <- standardize(train_df$ROOMS)
std_BEDRM    <- standardize(train_df$BEDRM)
std_KITCHENS <- standardize(train_df$KITCHENS)
std_FIREPL   <- standardize(train_df$FIREPLACES)
std_AYB      <- standardize(train_df$AYB.age)
std_EYB      <- standardize(train_df$EYB.age)
std_REM      <- standardize(train_df$REMODEL.age)

train_df <- train_df %>%
  mutate(
    BATHRM_z   = std_BATHRM$z,
    ROOMS_z    = std_ROOMS$z,
    BEDRM_z    = std_BEDRM$z,
    KITCHENS_z = std_KITCHENS$z,
    FIREPL_z   = std_FIREPL$z,
    AYB_z      = std_AYB$z,
    EYB_z      = std_EYB$z,
    REM_z      = std_REM$z
  )

## Precompute interaction terms in R (more efficient for JAGS) --------------
train_df <- train_df %>%
  mutate(
    ROOMS_AYB_z   = ROOMS_z * AYB_z,
    ROOMS_AC      = ROOMS_z * AC_id,
    REM_COND_z    = REM_z * COND_id,
    BATH_COND_z   = BATHRM_z * COND_id
  )
```

---

### 3. JAGS model (hierarchical, weakly‑informative, non‑centered)

```r
## Write JAGS model to a file for clarity and reuse -------------------------
model_string <- "
model {

  # Likelihood --------------------------------------------------------------
  for (i in 1:N) {
    # Linear predictor with random intercept by GRADE
    mu[i] <- alpha[GRADE_id[i]] +
             beta[1]  * BATHRM_z[i]   +
             beta[2]  * ROOMS_z[i]    +
             beta[3]  * BEDRM_z[i]    +
             beta[4]  * KITCHENS_z[i] +
             beta[5]  * FIREPL_z[i]   +
             beta[6]  * AYB_z[i]      +
             beta[7]  * EYB_z[i]      +
             beta[8]  * REM_z[i]      +
             beta[9]  * COND_id[i]    +
             beta[10] * REM_COND_z[i] +
             beta[11] * ROOMS_AYB_z[i]+
             beta[12] * ROOMS_AC[i]   +
             beta[13] * BATH_COND_z[i]

    # Observation model
    PRICE_10K[i] ~ dnorm(mu[i], tau_y)

    # Residuals for posterior predictive SD
    e_y[i] <- PRICE_10K[i] - mu[i]

    # Posterior predictive replicated data
    y_rep[i] ~ dnorm(mu[i], tau_y)
  }

  # Hierarchical priors for regression coefficients -------------------------
  for (k in 1:K) {
    beta[k] ~ dnorm(mu_beta, tau_beta)
  }
  mu_beta  ~ dnorm(0, 0.001)          # hyper-mean for betas (weakly informative)
  tau_beta ~ dgamma(1, 0.1)           # hyper-precision for betas
  sigma_beta <- 1 / sqrt(tau_beta)    # hyper-SD for betas

  # Random intercepts for GRADE (non-centered parameterization) -------------
  for (j in 1:N_GRADE) {
    eta_alpha[j] ~ dnorm(0, 1)        # standard normal latent
    alpha[j] <- mu_alpha + sigma_alpha * eta_alpha[j]
  }
  mu_alpha    ~ dnorm(0, 0.001)       # mean of random intercepts
  tau_alpha   ~ dgamma(1, 0.1)        # precision of random intercepts
  sigma_alpha <- 1 / sqrt(tau_alpha)  # SD of random intercepts

  # Residual variance (use weakly-informative prior) ------------------------
  tau_y   ~ dgamma(1, 0.1)            # residual precision
  sigma_y <- 1 / sqrt(tau_y)          # residual SD

  # Posterior predictive summary -------------------------------------------
  s_y <- sd(e_y[])                    # SD of residuals
}
"

writeLines(model_string, con = "final_bayes_model.bug")
```

**What this model does (conceptually):**

- `mu[i]` is the expected log‑price (scaled) for house `i`.
- `alpha[GRADE_id[i]]` is a **random intercept** by GRADE (hierarchical).
- `beta[k]` have a **hierarchical prior** with hyperparameters `mu_beta`, `tau_beta`.
- `eta_alpha[j]` + `mu_alpha`, `sigma_alpha` implement a **non‑centered** random intercept.
- `y_rep[i]` are **posterior predictive draws** for model checking.
- `s_y`, `sigma_y`, `sigma_alpha`, `sigma_beta` summarize uncertainty.

---

### 4. JAGS data list

```r
## Prepare data list for JAGS -----------------------------------------------
jags_data <- list(
  PRICE_10K    = train_df$PRICE_10K,
  BATHRM_z     = train_df$BATHRM_z,
  ROOMS_z      = train_df$ROOMS_z,
  BEDRM_z      = train_df$BEDRM_z,
  KITCHENS_z   = train_df$KITCHENS_z,
  FIREPL_z     = train_df$FIREPL_z,
  AYB_z        = train_df$AYB_z,
  EYB_z        = train_df$EYB_z,
  REM_z        = train_df$REM_z,
  COND_id      = train_df$COND_id,
  GRADE_id     = train_df$GRADE_id,
  ROOMS_AYB_z  = train_df$ROOMS_AYB_z,
  ROOMS_AC     = train_df$ROOMS_AC,
  REM_COND_z   = train_df$REM_COND_z,
  BATH_COND_z  = train_df$BATH_COND_z,
  N            = nrow(train_df),
  N_GRADE      = length(unique(train_df$GRADE_id)),
  K            = 13                     # number of beta coefficients
)
```

---

### 5. Initial values and parameters to monitor

```r
## Initial values function --------------------------------------------------
inits_fun <- function() {
  list(
    beta       = rnorm(jags_data$K, 0, 0.1),   # start betas near 0
    mu_beta    = 0,
    tau_beta   = 1,
    mu_alpha   = 0,
    tau_alpha  = 1,
    tau_y      = 1,
    eta_alpha  = rnorm(jags_data$N_GRADE, 0, 0.1)
  )
}

## Parameters to monitor ----------------------------------------------------
params <- c(
  "beta", "mu_beta", "sigma_beta",
  "alpha", "mu_alpha", "sigma_alpha",
  "sigma_y", "s_y",
  "y_rep"
)
```

---

### 6. Run JAGS

```r
## Fit the model with JAGS --------------------------------------------------
final_fit <- jags(
  data              = jags_data,
  inits             = inits_fun,
  parameters.to.save= params,
  model.file        = "final_bayes_model.bug",
  n.chains          = 3,
  n.iter            = 20000,
  n.burnin          = 5000,
  n.thin            = 10,
  DIC               = TRUE
)

print(final_fit)   # summary of posterior + DIC
```

---

### 7. MCMC diagnostics

```r
## Convert to coda object for diagnostics -----------------------------------
mcmc_obj <- as.mcmc(final_fit)

## Basic traceplots and density plots
plot(mcmc_obj)

## Superdiag diagnostics (R-hat, ESS, etc.)
superdiag(as.mcmc.list(mcmc_obj), burnin = 0)
```

---

### 8. Posterior predictive checks (PPCs)

```r
## Extract posterior predictive draws ---------------------------------------
# y_rep is N x (iterations * chains) in the JAGS output structure
y_rep_mat <- final_fit$BUGSoutput$sims.list$y_rep   # matrix: iterations x N
y_obs     <- jags_data$PRICE_10K

## Compare observed vs replicated means and SDs -----------------------------
ppc_mean_obs <- mean(y_obs)
ppc_mean_rep <- apply(y_rep_mat, 1, mean)   # mean per posterior draw

ppc_sd_obs   <- sd(y_obs)
ppc_sd_rep   <- apply(y_rep_mat, 1, sd)

## Simple Bayesian p-values -------------------------------------------------
p_mean <- mean(ppc_mean_rep > ppc_mean_obs)
p_sd   <- mean(ppc_sd_rep   > ppc_sd_obs)

cat(\"Bayesian p-value (mean):\", round(p_mean, 3), \"\\n\")
cat(\"Bayesian p-value (sd):\",   round(p_sd,   3), \"\\n\")

## Visual PPC: histogram overlay -------------------------------------------
hist(y_obs, breaks = 40, col = rgb(0,0,1,0.4),
     main = \"Posterior Predictive Check: PRICE_10K\",
     xlab = \"PRICE_10K\")

# Add a few replicated histograms
for (b in 1:30) {
  lines(density(y_rep_mat[b, ]), col = rgb(1,0,0,0.1))
}
```

---

### 9. Posterior predictive performance on test set

```r
## Prepare test data in the same standardized space -------------------------
test_df <- test_df %>%
  mutate(
    BATHRM_z   = (BATHRM   - std_BATHRM$mean)   / std_BATHRM$sd,
    ROOMS_z    = (ROOMS    - std_ROOMS$mean)    / std_ROOMS$sd,
    BEDRM_z    = (BEDRM    - std_BEDRM$mean)    / std_BEDRM$sd,
    KITCHENS_z = (KITCHENS - std_KITCHENS$mean) / std_KITCHENS$sd,
    FIREPL_z   = (FIREPLACES - std_FIREPL$mean) / std_FIREPL$sd,
    AYB_z      = (AYB.age  - std_AYB$mean)      / std_AYB$sd,
    EYB_z      = (EYB.age  - std_EYB$mean)      / std_EYB$sd,
    REM_z      = (REMODEL.age - std_REM$mean)   / std_REM$sd,
    ROOMS_AYB_z= ROOMS_z * AYB_z,
    ROOMS_AC   = ROOMS_z * AC_id,
    REM_COND_z = REM_z * COND_id,
    BATH_COND_z= BATHRM_z * COND_id
  )

## Use posterior draws to generate predictive distribution for test set -----
sims <- final_fit$BUGSoutput$sims.list

# Number of posterior draws
S <- nrow(sims$beta)

# Storage for posterior predictive means for each test observation
mu_test_mat <- matrix(NA, nrow = S, ncol = nrow(test_df))

for (s in 1:S) {
  # Extract one draw of parameters
  beta_s       <- sims$beta[s, ]
  alpha_s      <- sims$alpha[s, ]
  tau_y_s      <- 1 / (sims$sigma_y[s]^2)

  # Compute mu for all test observations under draw s
  mu_test_mat[s, ] <-
    alpha_s[test_df$GRADE_id] +
    beta_s[1]  * test_df$BATHRM_z   +
    beta_s[2]  * test_df$ROOMS_z    +
    beta_s[3]  * test_df$BEDRM_z    +
    beta_s[4]  * test_df$KITCHENS_z +
    beta_s[5]  * test_df$FIREPL_z   +
    beta_s[6]  * test_df$AYB_z      +
    beta_s[7]  * test_df$EYB_z      +
    beta_s[8]  * test_df$REM_z      +
    beta_s[9]  * test_df$COND_id    +
    beta_s[10] * test_df$REM_COND_z +
    beta_s[11] * test_df$ROOMS_AYB_z+
    beta_s[12] * test_df$ROOMS_AC   +
    beta_s[13] * test_df$BATH_COND_z
}

## Posterior predictive mean and intervals for each test observation --------
mu_test_mean <- apply(mu_test_mat, 2, mean)
mu_test_lwr  <- apply(mu_test_mat, 2, quantile, 0.025)
mu_test_upr  <- apply(mu_test_mat, 2, quantile, 0.975)

## Compute RMSE and coverage on test set ------------------------------------
y_test <- test_df$PRICE_10K

rmse_test <- sqrt(mean((y_test - mu_test_mean)^2))
coverage  <- mean(y_test >= mu_test_lwr & y_test <= mu_test_upr)

cat(\"Test RMSE (posterior mean prediction):\", round(rmse_test, 3), \"\\n\")
cat(\"Test 95% interval coverage:\", round(coverage, 3), \"\\n\")
```

---

If you want, next step we can:

- Add WAIC/LOO from the posterior log‑likelihood.
- Build a compact summary table (priors, posteriors, interpretation) for your report.

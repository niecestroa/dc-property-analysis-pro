### 1. Core improvements to aim for

- **Avoid `attach`/`detach`:** They’re fragile and make bugs hard to track.
- **Make JAGS data explicit and numeric:** No factors inside the JAGS model; pre-code them in R.
- **Center/scale key predictors:** Helps mixing and interpretability.
- **Clean up the model function:** Remove redundancy, fix interactions, and name things clearly.
- **Use consistent, weakly-informative priors:** Still vague, but numerically stable.
- **Separate model building, fitting, and diagnostics into clear blocks.**
- **Improvemets based on the Academic project I completed at American University**

---

### 2. Preprocessing for JAGS (clean, explicit)

```r
library(tidyverse)
library(R2jags)
library(coda)
library(superdiag)
library(lme4)

set.seed(10000000)

# Assuming DC_Final already created as in your script
DC_Final <- DC_Final %>%
  mutate(
    GRADE2      = as.integer(GRADE),
    CONDITION2  = as.integer(CONDITION),
    AC2         = as.integer(AC),
    # Centering / scaling
    BATHRM_c    = scale(BATHRM, center = TRUE, scale = TRUE)[,1],
    ROOMS_c     = scale(ROOMS, center = TRUE, scale = TRUE)[,1],
    BEDRM_c     = scale(BEDRM, center = TRUE, scale = TRUE)[,1],
    KITCHENS_c  = scale(KITCHENS, center = TRUE, scale = TRUE)[,1],
    FIREPLACES_c= scale(FIREPLACES, center = TRUE, scale = TRUE)[,1],
    AYB_c       = scale(AYB.age, center = TRUE, scale = TRUE)[,1],
    EYB_c       = scale(EYB.age, center = TRUE, scale = TRUE)[,1],
    REM_c       = scale(REMODEL.age, center = TRUE, scale = TRUE)[,1]
  )

n <- nrow(DC_Final)
Z <- sample(n, n/2)

bayes.train <- DC_Final[Z, ]
bayes.test  <- DC_Final[-Z, ]

jags_data <- list(
  PRICE_10K   = bayes.train$PRICE_10K,
  BATHRM      = bayes.train$BATHRM_c,
  ROOMS       = bayes.train$ROOMS_c,
  BEDRM       = bayes.train$BEDRM_c,
  KITCHENS    = bayes.train$KITCHENS_c,
  FIREPLACES  = bayes.train$FIREPLACES_c,
  AYB         = bayes.train$AYB_c,
  EYB         = bayes.train$EYB_c,
  REM         = bayes.train$REM_c,
  AC          = bayes.train$AC2,
  CONDITION   = bayes.train$CONDITION2,
  GRADE       = bayes.train$GRADE2,
  N           = nrow(bayes.train),
  N_GRADE     = length(unique(bayes.train$GRADE2))
)
```

---

### 3. Cleaner JAGS model definition

Key changes: no duplicated terms, interactions are explicit, priors are consistent, and indexing is clear.

```r
final_bayes_model <- function() {
  for (i in 1:N) {
    mu[i] <- alpha[GRADE[i]] +
      beta[1]  * BATHRM[i] +
      beta[2]  * ROOMS[i] +
      beta[3]  * BEDRM[i] +
      beta[4]  * KITCHENS[i] +
      beta[5]  * FIREPLACES[i] +
      beta[6]  * AYB[i] +
      beta[7]  * EYB[i] +
      beta[8]  * REM[i] +
      beta[9]  * CONDITION[i] +
      beta[10] * REM[i] * CONDITION[i] +
      beta[11] * ROOMS[i] * AYB[i] +
      beta[12] * ROOMS[i] * AC[i] +
      beta[13] * CONDITION[i] * BATHRM[i]

    PRICE_10K[i] ~ dnorm(mu[i], tau_y)
    e_y[i] <- PRICE_10K[i] - mu[i]
  }

  # Priors for betas
  for (k in 1:13) {
    beta[k] ~ dnorm(0, 0.001)
  }

  # Residual precision
  tau_y ~ dgamma(1, 0.1)
  sigma_y <- 1 / sqrt(tau_y)

  # Random intercepts for GRADE
  for (j in 1:N_GRADE) {
    alpha[j] ~ dnorm(0, tau_alpha)
  }
  tau_alpha ~ dgamma(1, 0.1)
  sigma_alpha <- 1 / sqrt(tau_alpha)

  # Posterior predictive SD of residuals
  s_y <- sd(e_y[])
}
```

---

### 4. Initial values and model run

```r
grade_inits <- function() {
  list(
    tau_y     = 1,
    tau_alpha = 1,
    beta      = rnorm(13, 0, 0.1),
    alpha     = rnorm(jags_data$N_GRADE, 0, 0.1)
  )
}

grade_params <- c("beta", "alpha", "tau_y", "tau_alpha", "sigma_y", "sigma_alpha", "s_y")

final_fit <- jags(
  data     = jags_data,
  inits    = grade_inits,
  parameters.to.save = grade_params,
  model.file = final_bayes_model,
  n.chains = 3,
  n.iter   = 10000,
  n.burnin = 2000,
  n.thin   = 5,
  DIC      = TRUE
)

print(final_fit)
```

---

### 5. MCMC diagnostics (cleaned)

```r
update_fit <- update(final_fit, n.iter = 15000)
mcmc_obj   <- as.mcmc(update_fit)

# Basic diagnostics
superdiag(as.mcmc.list(mcmc_obj), burnin = 0)
plot(mcmc_obj)
traceplot(mcmc_obj)
```

---

### 6. Keep your lmer model as a frequentist benchmark

```r
final.model.train <- lmer(
  PRICE_10K ~ BATHRM + BEDRM + ROOMS + KITCHENS + FIREPLACES +
    AYB.age + EYB.age + REMODEL.age + AC + CONDITION +
    REMODEL.age:CONDITION + ROOMS:AYB.age + ROOMS:AC + CONDITION:BATHRM +
    (1 | GRADE),
  data = bayes.train
)
summary(final.model.train)
```


# 2. **STAN MODEL — Bayesian Logistic Regression (Improved)**  
# Save this as: **`logistic_dc.stan`**

# This is the clean, modern version with:

#- weakly‑informative priors  
#- vectorized likelihood  
#- posterior predictive checks  
#- log_lik for WAIC/LOO  
#- test‑set predictions  

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

# Save as `logistic_dc_hier.stan`:

```stan
data {
  int<lower=1> N;
  int<lower=1> C;
  matrix[N, C] X;
  int<lower=0, upper=1> y[N];

  int<lower=1> N_GRADE;
  int<lower=1, upper=N_GRADE> grade_id[N];

  int<lower=1> N_test;
  matrix[N_test, C] X_test;
  int<lower=1, upper=N_GRADE> grade_test[N_test];
}

parameters {
  real intercept;
  vector[C] beta;

  real mu_alpha;
  real<lower=0> sigma_alpha;
  vector[N_GRADE] eta_alpha;

}

transformed parameters {
  vector[N_GRADE] alpha;
  vector[N] eta;
  vector[N] p;

  alpha = mu_alpha + sigma_alpha * eta_alpha;

  for (n in 1:N) {
    eta[n] = intercept + alpha[grade_id[n]] + X[n] * beta;
    p[n]   = inv_logit(eta[n]);
  }
}

model {
  // Priors
  intercept   ~ normal(0, 2.5);
  beta        ~ normal(0, 2.5);
  mu_alpha    ~ normal(0, 2.5);
  sigma_alpha ~ exponential(1);
  eta_alpha   ~ normal(0, 1);

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
    real eta_t = intercept + alpha[grade_test[n]] + X_test[n] * beta;
    p_test[n]  = inv_logit(eta_t);
    y_rep_test[n] = bernoulli_rng(p_test[n]);
  }
}
```

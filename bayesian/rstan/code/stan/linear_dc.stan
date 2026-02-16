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

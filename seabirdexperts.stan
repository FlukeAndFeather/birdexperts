data {
  int S, X;       // Number of species and experts, respectively
  vector[S] u;    // Utilizations
  matrix[S, X] o; // Expert opinions
  array[S] int s; // Seabirds
  array[X] int x; // Experts
}
parameters {
  real<lower=0> phi; // Concentration of utilization variable
  real alpha;        // Intercept
  real sigma_x;      // Standard deviation of expert random intercept
  real sigma_s;      // Standard deviation of seabird random intercept
  row_vector[X] Z_x; // Expert random intercepts
  vector[S] Z_s;     // Seabird random intercepts
}
model {
  // Priors
  phi ~ gamma(1, 1e-5);
  alpha ~ normal(0, 5);
  sigma_x ~ exponential(1);
  sigma_s ~ exponential(1);

  // Likelihood
  vector[S] mu;
  vector[S] shape, rate;
  for (i in 1:S) {
    mu[i] = inv_logit(alpha + mean(o[i] + Z_x) + Z_s[i]);
  }
  Z_x ~ normal(0, sigma_x);
  Z_s ~ normal(0, sigma_s);
  // Transform mu and phi to the shape and rate parameters expected by Stan
  shape = mu * phi;
  rate = (1 - mu) * phi;
  u ~ beta(shape, rate);
}

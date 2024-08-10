use rand::prelude::Distribution;
use statrs::{distribution::Weibull, function::gamma::gamma};
use std::ops::Div;

use super::Runnable;

fn weibull_k(min: f64, max: f64, min_quantile: f64, max_quantile: f64) -> f64 {
    // ln (ln (q_min) / ln (q_max))
    // ---------------------------
    // ln (max / (min + 10))

    let top_exp = min_quantile.ln().div(max_quantile.ln());
    let bottom_exp = max.div(min + 10.0);
    top_exp.ln().div(bottom_exp.ln())
}

fn weibull_lambda(avg: f64, k: f64) -> f64 {
    avg / gamma(1.0 + 1.0 / k)
}

pub fn simulation_weibull(runnable: Runnable) -> Weibull {
    let min_quantile = 0.00001;
    let max_quantile = 0.99999;

    let k = weibull_k(
        runnable.bcet as f64,
        runnable.wcet as f64,
        min_quantile,
        max_quantile,
    );
    let lambda = weibull_lambda(runnable.acet as f64, k);

    Weibull::new(lambda, k).unwrap()
}

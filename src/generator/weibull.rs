use statrs::{distribution::Weibull, function::gamma::gamma};
use std::ops::Div;

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

pub fn simulation_weibull(bcet: f64, acet: f64, wcet: f64) -> Weibull {
    let min_quantile = 0.00001;
    let max_quantile = 0.99999;

    let k = weibull_k(bcet, wcet, min_quantile, max_quantile);
    let lambda = weibull_lambda(acet, k);

    Weibull::new(lambda, k).unwrap()
}

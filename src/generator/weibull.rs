use rand::prelude::Distribution;
use statrs::{distribution::Weibull, function::gamma::gamma};
use std::ops::Div;

#[derive(Debug, Clone)]
pub struct RunnableWeibull {
    weibull: Weibull,
    bcet: f64,
    wcet: f64,
}

impl RunnableWeibull {
    pub fn new(bcet: f64, acet: f64, wcet: f64) -> RunnableWeibull {
        assert!(bcet <= acet);
        assert!(acet <= wcet);
        assert!(bcet >= 0.0);

        let min_quantile = 0.00001;
        let max_quantile = 0.99999;

        let k = Self::weibull_k(bcet, wcet, min_quantile, max_quantile);
        let lambda = Self::weibull_lambda(bcet, acet, k);

        let w = Weibull::new(k, lambda).unwrap();
        RunnableWeibull {
            weibull: w,
            bcet: (bcet as u64) as f64,
            wcet: (wcet as u64) as f64,
        }
    }

    pub fn sample(&self, rng: &mut impl rand::Rng) -> f64 {
        // statrs Weibull distribution does not directly support a location parameter.
        // We need to shift the distribution to the right by the BCET.
        (self.weibull.sample(rng) + self.bcet)
            .max(self.bcet)
            .min(self.wcet)
    }

    fn weibull_k(min: f64, max: f64, min_quantile: f64, max_quantile: f64) -> f64 {
        // ln (ln (q_min) / ln (q_max))
        // ---------------------------
        // ln (max - min)

        let top_exp = (1.0 - max_quantile).ln().div((1.0 - min_quantile).ln());
        let bottom_exp = max - min;
        top_exp.ln().div(bottom_exp.ln())
    }

    fn weibull_lambda(bcet: f64, acet: f64, k: f64) -> f64 {
        (acet - bcet) / gamma(1.0 + 1.0 / k)
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use crate::generator::Runnable;

    #[test]
    fn sample_ok() {
        let bcet: f64 = Runnable::duration_to_time_unit(Duration::from_micros(50)) as f64;
        let acet = Runnable::duration_to_time_unit(Duration::from_micros(100)) as f64;
        let wcet = Runnable::duration_to_time_unit(Duration::from_micros(200)) as f64;

        let weibull = super::RunnableWeibull::new(bcet, acet, wcet);
        let rng = &mut rand::thread_rng();

        let mut sum: f64 = 0.0;
        for _ in 0..100000 {
            let s = weibull.sample(rng);
            sum += s;
            assert!(s <= wcet);
            assert!(s >= bcet);
            print!("{} ", (s / 1000.0) as u64);
        }

        let avg = sum / 100000.0;
        assert_eq!((avg / 100.0).round(), (acet / 100.0).round());
    }
}

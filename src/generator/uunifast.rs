use rand::Rng;

fn uunifast(utilization: f64, number_runnables: usize) -> Vec<f64> {
    let mut u = vec![0.0; number_runnables];
    let mut t = utilization;
    let mut rng = rand::thread_rng();

    for i in (1..number_runnables).rev() {
        let s = t * rng.gen::<f64>().powf(1.0 / i as f64);
        u[i] = t - s;
        t = s;
    }

    u[0] = t;
    u
}

fn valid_utilizations(utilizations: Vec<f64>, min_acet: f64, max_acet: f64, period: f64) -> bool {
    for u in utilizations.iter() {
        if min_acet + u * (period - min_acet) > max_acet {
            return false;
        }
    }
    true
}

pub fn runnables_acets_uunifast(
    number_runnables: usize,
    avg_acet: f64,
    min_acet: f64,
    max_acet: f64,
    period: f64,
) -> Vec<f64> {
    loop {
        let acets = uunifast(avg_acet, number_runnables);
        if valid_utilizations(acets.clone(), min_acet, max_acet, period) {
            return acets;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uunifast() {
        let u = uunifast(0.8, 5);
        println!("{:?}", u);
    }
}

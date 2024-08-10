use ctor::ctor;
use rand::prelude::{Distribution, SliceRandom};
use statrs::distribution::{Uniform, Weibull};
use std::time::Duration;

use crate::simulator::{
    task::{SimulatorTask, TimeUnit},
    SimulatorMode,
};

mod uunifast;
mod weibull;

// Data gathered from "Real World Automotive Benchmarks For Free", from
// Simon Kramer, Dirk Ziegenbein, Arne Hamann, a corporate research paper
// from Robert Bosch GmbH.
static RUNNABLE_PERIODS: [Duration; 9] = [
    Duration::from_millis(1),
    Duration::from_millis(2),
    Duration::from_millis(5),
    Duration::from_millis(10),
    Duration::from_millis(20),
    Duration::from_millis(50),
    Duration::from_millis(100),
    Duration::from_millis(200),
    Duration::from_millis(1000),
];
static RUNNABLE_SHARES: [TimeUnit; 9] = [3, 2, 2, 25, 25, 3, 20, 1, 4];
#[ctor]
static MIN_AVG_MAX_AVG_EXECUTION_TIMES: [[Duration; 3]; 9] = [
    [
        Duration::from_micros(34).mul_f32(0.01),
        Duration::from_micros(500).mul_f32(0.01),
        Duration::from_micros(3011).mul_f32(0.01),
    ],
    [
        Duration::from_micros(32).mul_f32(0.01),
        Duration::from_micros(420).mul_f32(0.01),
        Duration::from_micros(4069).mul_f32(0.01),
    ],
    [
        Duration::from_micros(36).mul_f32(0.01),
        Duration::from_micros(1104).mul_f32(0.01),
        Duration::from_micros(8338).mul_f32(0.01),
    ],
    [
        Duration::from_micros(21).mul_f32(0.01),
        Duration::from_micros(1009).mul_f32(0.01),
        Duration::from_micros(30987).mul_f32(0.01),
    ],
    [
        Duration::from_micros(25).mul_f32(0.01),
        Duration::from_micros(874).mul_f32(0.01),
        Duration::from_micros(29142).mul_f32(0.01),
    ],
    [
        Duration::from_micros(29).mul_f32(0.01),
        Duration::from_micros(1756).mul_f32(0.01),
        Duration::from_micros(9298).mul_f32(0.01),
    ],
    [
        Duration::from_micros(21).mul_f32(0.01),
        Duration::from_micros(1053).mul_f32(0.01),
        Duration::from_micros(42043).mul_f32(0.01),
    ],
    [
        Duration::from_micros(22).mul_f32(0.01),
        Duration::from_micros(256).mul_f32(0.01),
        Duration::from_micros(2195).mul_f32(0.01),
    ],
    [
        Duration::from_micros(37).mul_f32(0.01),
        Duration::from_micros(43).mul_f32(0.01),
        Duration::from_micros(46).mul_f32(0.01),
    ],
];
static BCET_WCET_FACTORS: [[f64; 4]; 9] = [
    [0.19, 0.92, 1.30, 29.11],
    [0.12, 0.89, 1.54, 19.04],
    [0.17, 0.94, 1.13, 18.44],
    [0.05, 0.99, 1.06, 30.03],
    [0.11, 0.98, 1.06, 15.61],
    [0.32, 0.95, 1.13, 7.76],
    [0.09, 0.99, 1.02, 8.88],
    [0.45, 0.98, 1.03, 4.90],
    [0.68, 0.80, 1.84, 4.75],
];
static RUNNABLES_PER_PERIOD_TAKE: [u8; 4] = [2, 3, 4, 5];
static RUNNABLES_PER_PERIOD_TAKE_WEIGHTS: [u8; 4] = [30, 40, 20, 10];

#[derive(Clone)]
pub struct Runnable {
    // Given a runnable with a given period,
    // the average execution time depends on
    // the nature of the task.
    acet: TimeUnit,

    // The best case exectution time (BCET) and the worst case
    // execution time (WCET) are calculated given a factor
    // f oscillating between f_min and f_max.
    bcet: TimeUnit,
    wcet: TimeUnit,

    // Used for sampling the execution time of the runnable.
    weibull: Weibull,
}

impl Runnable {
    fn new_batch(period: Duration) -> Vec<Runnable> {
        let period_index = RUNNABLE_PERIODS.iter().position(|&x| x == period).unwrap();
        let [min_acet, avg_acet, max_acet] = MIN_AVG_MAX_AVG_EXECUTION_TIMES[period_index];

        let runnables_per_period_take = *RUNNABLES_PER_PERIOD_TAKE
            .choose_weighted(&mut rand::thread_rng(), |p| {
                let index = RUNNABLES_PER_PERIOD_TAKE
                    .iter()
                    .position(|&x| x == *p)
                    .unwrap();
                RUNNABLES_PER_PERIOD_TAKE_WEIGHTS[index]
            })
            .unwrap();

        let acets = uunifast::runnables_acets_uunifast(
            runnables_per_period_take as usize,
            Self::duration_to_time_unit(avg_acet) as f64,
            Self::duration_to_time_unit(min_acet) as f64,
            Self::duration_to_time_unit(max_acet) as f64,
            Self::duration_to_time_unit(period) as f64,
        );

        let rng = &mut rand::thread_rng();

        acets
            .iter()
            .map(|&acet| {
                let [bcet_fmin, bcet_fmax, wcet_fmin, wcet_fmax] = BCET_WCET_FACTORS[period_index];
                let bcet_f = Uniform::new(bcet_fmin as f64, bcet_fmax as f64)
                    .unwrap()
                    .sample(rng);
                let wcet_f = Uniform::new(wcet_fmin as f64, wcet_fmax as f64)
                    .unwrap()
                    .sample(rng);
                let bcet = (acet * bcet_f) as TimeUnit;
                let wcet = (acet * wcet_f) as TimeUnit;
                Runnable {
                    acet: acet as TimeUnit,
                    bcet,
                    wcet,
                    weibull: weibull::simulation_weibull(bcet as f64, acet as f64, wcet as f64),
                }
            })
            .collect()
    }

    fn wcet_l_estimate(&self) -> TimeUnit {
        todo!()
    }

    pub fn duration_to_time_unit(duration: Duration) -> TimeUnit {
        // We'll represent a second as 100_000_000 units.
        // This allows us to represent us with a precision of 10^-2.
        (duration.as_secs_f64() * 100_000_000.0) as TimeUnit
    }

    pub fn sample_exec_time(&self) -> f64 {
        let rng = &mut rand::thread_rng();
        let s = self.weibull.sample(rng) + self.bcet as f64;
        assert!(s <= self.wcet as f64);
        assert!(s >= self.bcet as f64);
        s
    }
}

pub fn generate_tasks() -> Vec<SimulatorTask> {
    for _mode in &[SimulatorMode::LMode, SimulatorMode::HMode] {
        for _period in &RUNNABLE_PERIODS {}
    }
    todo!()
}

#[cfg(test)]
mod tests {}

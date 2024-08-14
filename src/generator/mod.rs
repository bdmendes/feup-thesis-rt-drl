use ctor::ctor;
use rand::prelude::{Distribution, SliceRandom};
use statrs::distribution::Uniform;
use std::time::Duration;
use weibull::RunnableWeibull;

use crate::simulator::{
    task::{SimulatorTask, Task, TaskProps, TimeUnit},
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
static WCET_L_PROBABILITIES_PER_PERIOD_L: [u64; 9] = [75, 75, 75, 67, 67, 67, 50, 50, 50];
static WCET_L_PROBABILITIES_PER_PERIOD_H: [u64; 9] = [80, 80, 80, 75, 75, 75, 67, 67, 67];

#[derive(Clone, Debug)]
pub struct Runnable {
    // Given a runnable with a given period,
    // the average execution time depends on
    // the nature of the task.
    _acet: TimeUnit,

    // The best case exectution time (BCET) and the worst case
    // execution time (WCET) are calculated given a factor
    // f oscillating between f_min and f_max.
    bcet: TimeUnit,
    wcet: TimeUnit,

    // Used for sampling the execution time of the runnable.
    weibull: RunnableWeibull,
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

        assert!((2..=5).contains(&runnables_per_period_take));

        let acets = uunifast::runnables_acets_uunifast(
            runnables_per_period_take as usize,
            Self::duration_to_time_unit(avg_acet) as f64,
            Self::duration_to_time_unit(min_acet) as f64,
            Self::duration_to_time_unit(max_acet) as f64,
            Self::duration_to_time_unit(period) as f64,
        );
        assert_eq!(acets.len(), runnables_per_period_take as usize);
        let rng = &mut rand::thread_rng();

        acets
            .iter()
            .map(|&acet| {
                let [bcet_fmin, bcet_fmax, wcet_fmin, wcet_fmax] = BCET_WCET_FACTORS[period_index];
                let bcet_f = Uniform::new(bcet_fmin, bcet_fmax).unwrap().sample(rng);
                let wcet_f = Uniform::new(wcet_fmin, wcet_fmax).unwrap().sample(rng);
                let bcet = acet * bcet_f;
                let wcet = acet * wcet_f;
                Runnable {
                    _acet: acet as TimeUnit,
                    bcet: bcet as TimeUnit,
                    wcet: wcet as TimeUnit,
                    weibull: RunnableWeibull::new(bcet, acet, wcet),
                }
            })
            .collect()
    }

    fn wcet_l_estimate(&self, period: Duration, mode: SimulatorMode) -> f64 {
        // Sample execution times 100 times and sort them.
        let rng = &mut rand::thread_rng();
        let mut samples = (0..100)
            .map(|_| self.weibull.sample(rng))
            .collect::<Vec<f64>>();
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Find the budget assurance for this period.
        let period_index = RUNNABLE_PERIODS.iter().position(|&x| x == period).unwrap();
        let wcet_l_probability = match mode {
            SimulatorMode::LMode => WCET_L_PROBABILITIES_PER_PERIOD_L[period_index],
            SimulatorMode::HMode => WCET_L_PROBABILITIES_PER_PERIOD_H[period_index],
        };

        // Return the execution time that satisfies the budget assurance.
        samples[wcet_l_probability as usize]
    }

    pub fn duration_to_time_unit(duration: Duration) -> TimeUnit {
        // We'll represent a second as 100_000_000 units.
        // This allows us to represent us with a precision of 10^-2.
        (duration.as_secs_f64() * 100_000_000.0) as TimeUnit
    }

    pub fn sample_exec_time(&self) -> f64 {
        let rng = &mut rand::thread_rng();
        let s = self.weibull.sample(rng);
        println!("s: {}, wcet: {}", s, self.wcet);
        assert!(s <= self.wcet as f64);
        assert!(s >= self.bcet as f64);
        s.max(1.0)
    }
}

pub fn generate_tasks() -> Vec<SimulatorTask> {
    let mut id = 0;
    let mut tasks = Vec::new();

    for mode in &[SimulatorMode::LMode, SimulatorMode::HMode] {
        for period in &RUNNABLE_PERIODS {
            let runnables = Runnable::new_batch(*period);
            let props = TaskProps {
                id,
                period: Runnable::duration_to_time_unit(*period),
                wcet_l: runnables
                    .iter()
                    .map(|r| r.wcet_l_estimate(*period, *mode))
                    .sum::<f64>() as TimeUnit,
                wcet_h: runnables.iter().map(|r| r.wcet).sum(),
                offset: 0,
            };
            tasks.push(SimulatorTask::new_with_runnables(
                match mode {
                    SimulatorMode::LMode => Task::LTask(props),
                    SimulatorMode::HMode => Task::HTask(props),
                },
                runnables,
            ));
            id += 1;
        }
    }

    tasks
}

#[cfg(test)]
mod tests {
    #[test]
    fn gen_tasks() {
        let tasks = super::generate_tasks();
        assert_eq!(tasks.len(), 18);

        for task in tasks {
            println!("Task: {:?}", task.task.props().id);
            println!("Period: {}", task.task.props().period);
            println!("WCET_L: {}", task.task.props().wcet_l);
            for runnable in task.clone().runnables.unwrap() {
                println!("BCET: {}, WCET: {}", runnable.bcet, runnable.wcet);
            }
            for sample_nr in 0..10 {
                println!("Sample {}: {}", sample_nr, task.sample_execution_time());
            }
            println!();
        }
    }
}

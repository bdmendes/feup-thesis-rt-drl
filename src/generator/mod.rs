use crate::simulator::{
    task::{SimulatorTask, Task, TaskProps, TimeUnit},
    SimulatorMode,
};
use ctor::ctor;
use rand::prelude::{Distribution, SliceRandom};
use rand::Rng;
use statrs::distribution::Uniform;
use std::{collections::HashMap, time::Duration};
use weibull::RunnableWeibull;

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
static RUNNABLE_DISTRIBUTION_PER_PERIOD: [u64; 9] = [3, 2, 2, 25, 25, 3, 20, 1, 4];
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

impl PartialEq for Runnable {
    fn eq(&self, other: &Self) -> bool {
        self._acet == other._acet && self.bcet == other.bcet && self.wcet == other.wcet
    }
}

impl Runnable {
    fn new_batch(period: Duration, number: usize) -> Vec<Runnable> {
        let period_index = RUNNABLE_PERIODS.iter().position(|&x| x == period).unwrap();
        let [min_acet, avg_acet, max_acet] = MIN_AVG_MAX_AVG_EXECUTION_TIMES[period_index];

        let acets = uunifast::runnables_acets_uunifast(
            number,
            Self::duration_to_time_unit(avg_acet) as f64,
            Self::duration_to_time_unit(min_acet) as f64,
            Self::duration_to_time_unit(max_acet) as f64,
            Self::duration_to_time_unit(period) as f64,
        );
        assert_eq!(acets.len(), number);
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
        assert!(s <= self.wcet as f64);
        assert!(s >= self.bcet as f64);
        s.max(1.0)
    }
}

pub fn generate_tasks(number_runnables: usize) -> Vec<SimulatorTask> {
    let rng = &mut rand::thread_rng();
    let mut period_runnables = HashMap::<Duration, usize>::new();
    let mut id = 0;
    let mut tasks = Vec::new();

    for _ in 0..number_runnables {
        let chosen_period = RUNNABLE_PERIODS
            .choose_weighted(rng, |&x| {
                RUNNABLE_DISTRIBUTION_PER_PERIOD
                    [RUNNABLE_PERIODS.iter().position(|&y| y == x).unwrap()]
            })
            .unwrap();
        period_runnables
            .entry(*chosen_period)
            .and_modify(|e| *e += 1)
            .or_insert(1);
    }

    for period in period_runnables.keys() {
        let runnables = Runnable::new_batch(*period, period_runnables[period]);
        let l_runnables = runnables
            .iter()
            .filter(|_| rng.gen_bool(0.5))
            .cloned()
            .collect::<Vec<Runnable>>();
        let h_runnables = runnables
            .iter()
            .filter(|r| !l_runnables.contains(r))
            .cloned()
            .collect::<Vec<Runnable>>();

        // L-task
        if !l_runnables.is_empty() {
            let l_task_props = TaskProps {
                id,
                offset: 0,
                period: Runnable::duration_to_time_unit(*period),
                wcet_l: l_runnables
                    .iter()
                    .map(|r| r.wcet_l_estimate(*period, SimulatorMode::LMode))
                    .sum::<f64>() as u64,
                wcet_h: l_runnables.iter().map(|r| r.wcet).sum(),
            };
            id += 1;
            tasks.push(SimulatorTask::new_with_runnables(
                Task::LTask(l_task_props),
                l_runnables,
            ));
        }

        // H-task
        if !h_runnables.is_empty() {
            let h_task_props = TaskProps {
                id,
                offset: 0,
                period: Runnable::duration_to_time_unit(*period),
                wcet_l: h_runnables
                    .iter()
                    .map(|r| r.wcet_l_estimate(*period, SimulatorMode::HMode))
                    .sum::<f64>() as u64,
                wcet_h: h_runnables.iter().map(|r| r.wcet).sum(),
            };
            id += 1;
            tasks.push(SimulatorTask::new_with_runnables(
                Task::HTask(h_task_props),
                h_runnables,
            ));
        }
    }

    tasks
}

#[cfg(test)]
mod tests {
    use crate::simulator::validation::feasible_schedule_design_time;

    #[test]
    fn gen_tasks() {
        let tasks = super::generate_tasks(80);

        for task in tasks {
            println!(
                "Task: {:?}, mode: {}",
                task.task.props().id,
                match task.task {
                    super::Task::LTask(_) => "L",
                    super::Task::HTask(_) => "H",
                }
            );
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

    #[test]
    fn schedulable_sets() {
        let mut data = vec![];

        for nr_runnables in (10..=700).step_by(10) {
            let mut schedulable_sets = 0;
            for _ in 0..500 {
                let tasks = super::generate_tasks(nr_runnables);
                if feasible_schedule_design_time(&tasks.clone()) {
                    schedulable_sets += 1;
                }
            }
            println!("{} {}", nr_runnables, schedulable_sets as f64 / 500.0);
            data.push(schedulable_sets as f64 / 500.0);
        }

        println!("{:?}", data);
    }
}

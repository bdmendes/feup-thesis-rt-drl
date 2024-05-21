use ctor::ctor;
use probability::source::Xorshift128Plus;
use probability::{
    distribution::{Sample, Triangular, Uniform},
    source::{self},
};
use rand::prelude::SliceRandom;
use std::time::Duration;

use crate::simulator::task::{SimulatorTask, Task, TaskProps, TimeUnit};

pub const AVG_TOTAL_RUNNABLES: usize = 1500;

// Data gathered from "Real World Automotive Benchmarks For Free", from
// Simon Kramer, Dirk Ziegenbein, Arne Hamann, a corporate research paper
// from Robert Bosch GmbH.

// Notes:
// 1. The last period is derived from T = 120 / (rpm * #cyl), considering 2000 rpm and 4 cylinders.

static RUNNABLE_PERIODS: [Duration; 10] = [
    Duration::from_millis(1),
    Duration::from_millis(2),
    Duration::from_millis(5),
    Duration::from_millis(10),
    Duration::from_millis(20),
    Duration::from_millis(50),
    Duration::from_millis(100),
    Duration::from_millis(200),
    Duration::from_millis(1000),
    Duration::from_millis(15),
];
static RUNNABLE_SHARES: [TimeUnit; 10] = [3, 2, 2, 25, 25, 3, 20, 1, 4, 15];
#[ctor]
static MIN_AVG_MAX_AVG_EXECUTION_TIMES: [[Duration; 3]; 10] = [
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
    [
        Duration::from_micros(45).mul_f32(0.01),
        Duration::from_micros(652).mul_f32(0.01),
        Duration::from_micros(8858).mul_f32(0.01),
    ],
];
static BCET_WCET_FACTORS: [[f32; 4]; 10] = [
    [0.19, 0.92, 1.30, 29.11],
    [0.12, 0.89, 1.54, 19.04],
    [0.17, 0.94, 1.13, 18.44],
    [0.05, 0.99, 1.06, 30.03],
    [0.11, 0.98, 1.06, 15.61],
    [0.32, 0.95, 1.13, 7.76],
    [0.09, 0.99, 1.02, 8.88],
    [0.45, 0.98, 1.03, 4.90],
    [0.68, 0.80, 1.84, 4.75],
    [0.13, 0.92, 1.20, 28.17],
];

#[derive(Clone)]
struct Runnable {
    // Given a runnable with a given period,
    // the average execution time depends on
    // the nature of the task.
    acet: TimeUnit,

    // The best case exectution time (BCET) and the worst case
    // execution time (WCET) are calculated given a factor
    // f oscillating between f_min and f_max.
    bcet: TimeUnit,
    wcet: TimeUnit,
}

impl Runnable {
    fn generate_new(period: Duration, source: &mut Xorshift128Plus) -> Runnable {
        let index = RUNNABLE_PERIODS.iter().position(|&x| x == period).unwrap();

        // We'll determine the average execution time of this runnable
        // using a triangular distribution, since we have the minimum,
        // average and maximum.
        let [min_acet, avg_acet, max_acet] = MIN_AVG_MAX_AVG_EXECUTION_TIMES[index];
        let acet = Triangular::new(
            Self::duration_to_time_unit(min_acet) as f64,
            Self::duration_to_time_unit(max_acet) as f64,
            Self::duration_to_time_unit(avg_acet) as f64,
        )
        .sample(source);

        // We'll determine the best case execution time (BCET) and the worst
        // case execution time (WCET) using a factor f oscillating between
        // f_min and f_max.
        let [bcet_fmin, bcet_fmax, wcet_fmin, wcet_fmax] = BCET_WCET_FACTORS[index];
        let bcet_f = Uniform::new(bcet_fmin as f64, bcet_fmax as f64).sample(source);
        let wcet_f = Uniform::new(wcet_fmin as f64, wcet_fmax as f64).sample(source);

        Runnable {
            acet: acet as TimeUnit,
            bcet: (acet * bcet_f) as TimeUnit,
            wcet: (acet * wcet_f) as TimeUnit,
        }
    }

    fn wcet_l_estimate(&self, source: &mut Xorshift128Plus) -> TimeUnit {
        if self.acet == self.wcet {
            return self.acet;
        }

        // The WCET_L is a random value between the ACET and the WCET.
        Uniform::new(self.acet as f64, self.wcet as f64).sample(source) as TimeUnit
    }

    fn duration_to_time_unit(duration: Duration) -> TimeUnit {
        // We'll represent a second as 100_000_000 units.
        // This allows us to represent us with a precision of 10^-2.
        (duration.as_secs_f64() * 100_000_000.0) as TimeUnit
    }
}

pub fn generate_tasks(
    lmode_prob: f64,
    number_tasks: usize,
    offsets_per_period_take: usize,
) -> Vec<SimulatorTask> {
    assert!((0.0..=1.0).contains(&lmode_prob));
    assert!(number_tasks > 0);
    assert!(offsets_per_period_take > 0);

    let mut tasks = Vec::new();

    let mut source = source::default(42);
    let mut rng = rand::thread_rng();

    let is_ltask_dist = Uniform::new(0.0, 1.0);
    let offset_dist = Uniform::new(
        0.0,
        Runnable::duration_to_time_unit(RUNNABLE_PERIODS[0] / 10) as f64,
    );

    let mut id = 0;
    while tasks.len() < number_tasks {
        // Choose a period.
        let period = *RUNNABLE_PERIODS
            .choose_weighted(&mut rng, |p| {
                let index = RUNNABLE_PERIODS.iter().position(|&x| x == *p).unwrap();
                RUNNABLE_SHARES[index]
            })
            .unwrap();
        let period_in_units = Runnable::duration_to_time_unit(period);

        // Generate a number of runnables.
        let runnables = (0..offsets_per_period_take)
            .map(|_| Runnable::generate_new(period, &mut source))
            .collect::<Vec<_>>();

        // Calculate offset for this take.
        let offset = offset_dist.sample(&mut source) as TimeUnit;

        // Create a task for each runnable.
        for runnable in runnables {
            let is_ltask = is_ltask_dist.sample(&mut source) < lmode_prob;
            let props = TaskProps {
                id,
                wcet_l: runnable.wcet_l_estimate(&mut source),
                wcet_h: runnable.wcet,
                offset,
                period: period_in_units,
            };
            id += 1;

            let task = if is_ltask {
                Task::LTask(props)
            } else {
                Task::HTask(props)
            };

            tasks.push(SimulatorTask {
                task,
                priority: period_in_units,
                acet: runnable.acet,
                bcet: runnable.bcet,
            });
        }
    }

    tasks.iter().take(number_tasks).cloned().collect()
}

#[cfg(test)]
mod tests {
    use crate::generator::AVG_TOTAL_RUNNABLES;

    #[test]
    fn smoke_generator() {
        let tasks = super::generate_tasks(0.3, AVG_TOTAL_RUNNABLES, 5);

        assert_eq!(tasks.len(), AVG_TOTAL_RUNNABLES);

        for task in tasks {
            assert!(task.task.props().wcet_h <= task.task.props().period);
            assert!(task.task.props().wcet_l <= task.task.props().wcet_h);
            assert!(task.task.props().wcet_l > 0);
            assert!(task.acet <= task.task.props().wcet_l);
            assert!(task.bcet <= task.acet);
        }
    }
}

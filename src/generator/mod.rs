use ctor::ctor;
use probability::distribution::Pert;
use probability::source::Xorshift128Plus;
use probability::{
    distribution::{Sample, Triangular, Uniform},
    source::{self},
};
use rand::prelude::SliceRandom;
use std::time::Duration;

use crate::simulator::task::{SimulatorTask, Task, TaskProps, TimeUnit};

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
static BCET_WCET_FACTORS: [[f32; 4]; 9] = [
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

#[derive(Debug, Clone, Copy)]
pub enum TimeSampleDistribution {
    Triangular,
    Pert,
}

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
}

impl Runnable {
    fn generate_new(
        period: Duration,
        source: &mut Xorshift128Plus,
        dist: TimeSampleDistribution,
    ) -> Runnable {
        let index = RUNNABLE_PERIODS.iter().position(|&x| x == period).unwrap();

        // We'll determine the average execution time of this runnable
        // using a pert distribution, since we have the minimum,
        // average and maximum.
        let [min_acet, avg_acet, max_acet] = MIN_AVG_MAX_AVG_EXECUTION_TIMES[index];
        let acet = match dist {
            TimeSampleDistribution::Triangular => Triangular::new(
                Self::duration_to_time_unit(min_acet) as f64,
                Self::duration_to_time_unit(max_acet) as f64,
                Self::duration_to_time_unit(avg_acet) as f64,
            )
            .sample(source),
            TimeSampleDistribution::Pert => Pert::new(
                Self::duration_to_time_unit(min_acet) as f64,
                Self::duration_to_time_unit(avg_acet) as f64,
                Self::duration_to_time_unit(max_acet) as f64,
            )
            .sample(source),
        };

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

    pub fn duration_to_time_unit(duration: Duration) -> TimeUnit {
        // We'll represent a second as 100_000_000 units.
        // This allows us to represent us with a precision of 10^-2.
        (duration.as_secs_f64() * 100_000_000.0) as TimeUnit
    }
}

pub fn generate_tasks(
    lmode_prob: f64,
    mut number_runnables: usize,
    dist: TimeSampleDistribution,
) -> Vec<SimulatorTask> {
    assert!((0.0..=1.0).contains(&lmode_prob));
    assert!(number_runnables > 0);

    let mut tasks = Vec::new();

    let mut source = source::default(42);
    let mut rng = rand::thread_rng();

    let is_ltask_dist = Uniform::new(0.0, 1.0);
    let offset_dist = Uniform::new(
        0.0,
        Runnable::duration_to_time_unit(RUNNABLE_PERIODS[0] / 10) as f64,
    );

    let mut id = 0;
    while number_runnables > 0 {
        // Choose a period.
        let period = *RUNNABLE_PERIODS
            .choose_weighted(&mut rng, |p| {
                let index = RUNNABLE_PERIODS.iter().position(|&x| x == *p).unwrap();
                RUNNABLE_SHARES[index]
            })
            .unwrap();
        let period_in_units = Runnable::duration_to_time_unit(period);

        // Calculate number of runnables in this period.
        let runnables_per_period_take = *RUNNABLES_PER_PERIOD_TAKE
            .choose_weighted(&mut rng, |p| {
                let index = RUNNABLES_PER_PERIOD_TAKE
                    .iter()
                    .position(|&x| x == *p)
                    .unwrap();
                RUNNABLES_PER_PERIOD_TAKE_WEIGHTS[index]
            })
            .unwrap();

        // Generate a number of runnables.
        let runnables = (0..runnables_per_period_take)
            .map(|_| Runnable::generate_new(period, &mut source, dist))
            .collect::<Vec<_>>();
        number_runnables = number_runnables.saturating_sub(runnables.len());

        // Calculate offset for this take.
        let offset = offset_dist.sample(&mut source) as TimeUnit;

        // Create a task with all these runnables.
        let is_ltask = is_ltask_dist.sample(&mut source) < lmode_prob;
        let props = TaskProps {
            id,
            wcet_l: runnables
                .iter()
                .map(|r| r.wcet_l_estimate(&mut source))
                .sum(),
            wcet_h: runnables.iter().map(|r| r.wcet).sum(),
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
            acet: runnables.iter().map(|r| r.acet).sum(),
            bcet: runnables.iter().map(|r| r.bcet).sum(),
        });
    }

    tasks
}

#[cfg(test)]
mod tests {
    use crate::{
        generator::TimeSampleDistribution::Pert,
        simulator::validation::feasible_schedule_design_time,
    };

    use super::RUNNABLE_PERIODS;

    #[test]
    fn schedulability_lmode03() {
        let l_mode_prob = 0.3;
        for number_taks in 1..150 {
            let is_schedulable_count = (0..1000)
                .map(|_| {
                    let tasks = super::generate_tasks(l_mode_prob, number_taks, Pert);
                    feasible_schedule_design_time(&tasks)
                })
                .filter(|&x| x)
                .count() as f64;
            print!("{}, ", is_schedulable_count / 1000.0);
        }
    }

    #[test]
    fn schedulability_lmode06() {
        let l_mode_prob = 0.6;
        for number_taks in 1..150 {
            let is_schedulable_count = (0..1000)
                .map(|_| {
                    let tasks = super::generate_tasks(l_mode_prob, number_taks, Pert);
                    feasible_schedule_design_time(&tasks)
                })
                .filter(|&x| x)
                .count() as f64;
            print!("{}, ", is_schedulable_count / 1000.0);
        }
    }

    #[test]
    fn utilization() {
        let tasks = super::generate_tasks(0.5, 100000, Pert);
        for period in RUNNABLE_PERIODS {
            let period_units = super::Runnable::duration_to_time_unit(period);
            let tasks_with_this_period = tasks
                .iter()
                .filter(|t| t.task.props().period == period_units)
                .take(1000)
                .collect::<Vec<_>>();
            let utilizations = tasks_with_this_period
                .iter()
                .map(|t| t.task.props().utilization())
                .filter(|&u| u <= 1.0)
                .collect::<Vec<_>>();
            println!("{:?},", utilizations);
        }
    }
}

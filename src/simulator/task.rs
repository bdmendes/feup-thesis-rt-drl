use probability::distribution::{Pert, Sample, Triangular};
use probability::source::Xorshift128Plus;

use crate::generator::TimeSampleDistribution;

use super::SimulatorMode;

pub type TaskId = u64;
pub type TimeUnit = u64;

#[derive(Clone, Debug)]
pub enum Task {
    LTask(TaskProps),
    HTask(TaskProps),
}

impl Task {
    pub fn set_id(&mut self, id: TaskId) {
        match self {
            Task::LTask(props) => props.id = id,
            Task::HTask(props) => props.id = id,
        }
    }

    pub fn props(&self) -> TaskProps {
        match self {
            Task::LTask(props) => *props,
            Task::HTask(props) => *props,
        }
    }

    pub fn props_mut(&mut self) -> &mut TaskProps {
        match self {
            Task::LTask(props) => props,
            Task::HTask(props) => props,
        }
    }

    pub fn sample_execution_time(
        acet: TimeUnit,
        bcet: TimeUnit,
        wcet: TimeUnit,
        source: &mut Xorshift128Plus,
        dist: TimeSampleDistribution,
    ) -> TimeUnit {
        match dist {
            TimeSampleDistribution::Triangular => {
                Triangular::new(bcet as f64, wcet as f64, acet as f64).sample(source) as TimeUnit
            }
            TimeSampleDistribution::Pert => {
                Pert::new(bcet as f64, acet as f64, wcet as f64).sample(source) as TimeUnit
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TaskProps {
    pub id: TaskId,
    pub wcet_l: TimeUnit,
    pub wcet_h: TimeUnit,
    pub offset: TimeUnit,
    pub period: TimeUnit,
}

impl TaskProps {
    pub fn new_empty(id: TaskId) -> Self {
        Self {
            id,
            wcet_l: 0,
            wcet_h: 0,
            offset: 0,
            period: 0,
        }
    }

    pub fn wcet_in_mode(&self, mode: SimulatorMode) -> TimeUnit {
        match mode {
            SimulatorMode::LMode => self.wcet_l,
            SimulatorMode::HMode => self.wcet_h,
        }
    }

    pub fn utilization(&self) -> f64 {
        self.wcet_h as f64 / self.period as f64
    }
}

#[derive(Clone, Debug)]
pub struct SimulatorTask {
    pub task: Task,
    pub custom_priority: Option<u64>,
    pub acet: TimeUnit, // Average Case Execution Time
    pub bcet: TimeUnit, // Best Case Execution Time
    pub next_arrival: TimeUnit,
}

impl SimulatorTask {
    pub fn new(task: Task, acet: TimeUnit, bcet: TimeUnit) -> Self {
        assert!(acet > 0, "Execution time must be greater than 0.");
        assert!(bcet > 0, "Execution time must be greater than 0.");
        Self {
            task: task.clone(),
            custom_priority: None,
            acet,
            bcet,
            next_arrival: task.props().offset,
        }
    }

    pub fn new_with_custom_priority(task: Task, priority: TimeUnit, acet: TimeUnit) -> Self {
        assert!(acet > 0, "Execution time must be greater than 0.");
        Self {
            task: task.clone(),
            custom_priority: Some(priority),
            acet,
            bcet: acet,
            next_arrival: task.props().offset,
        }
    }
}

#[cfg(test)]
mod tests {
    use probability::source;

    #[test]
    fn sample_time() {
        let (bcet, acet, wcet) = (3, 10, 30);
        let mut source = source::default(42);

        for _ in 0..10000 {
            let time = super::Task::sample_execution_time(
                acet,
                bcet,
                wcet,
                &mut source,
                super::TimeSampleDistribution::Pert,
            );
            print!("{}, ", time);
        }
        println!("\n");

        for _ in 0..10000 {
            let time = super::Task::sample_execution_time(
                acet,
                bcet,
                wcet,
                &mut source,
                super::TimeSampleDistribution::Triangular,
            );
            print!("{}, ", time);
        }
    }
}

use probability::distribution::{Sample, Triangular};
use probability::source::Xorshift128Plus;

use super::SimulatorMode;

pub type TaskId = u64;
pub type TimeUnit = u64;

#[derive(Clone, Debug)]
pub enum Task {
    LTask(TaskProps),
    HTask(TaskProps),
}

impl Task {
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
    ) -> TimeUnit {
        let dist = Triangular::new(bcet as f64, wcet as f64, acet as f64);
        dist.sample(source) as TimeUnit
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
    pub fn wcet_in_mode(&self, mode: SimulatorMode) -> TimeUnit {
        match mode {
            SimulatorMode::LMode => self.wcet_l,
            SimulatorMode::HMode => self.wcet_h,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SimulatorTask {
    pub task: Task,
    pub priority: TimeUnit,
    pub acet: TimeUnit, // Average Case Execution Time
    pub bcet: TimeUnit, // Best Case Execution Time
}

impl SimulatorTask {
    pub fn new(task: Task, acet: TimeUnit, bcet: TimeUnit) -> Self {
        assert!(acet > 0, "Execution time must be greater than 0.");
        assert!(bcet > 0, "Execution time must be greater than 0.");
        Self {
            task: task.clone(),
            priority: task.props().period, // RMS (Rate Monotonic Scheduling)
            acet,
            bcet,
        }
    }

    pub fn new_with_custom_priority(task: Task, priority: TimeUnit, acet: TimeUnit) -> Self {
        assert!(acet > 0, "Execution time must be greater than 0.");
        Self {
            task: task.clone(),
            priority,
            acet,
            bcet: acet,
        }
    }
}

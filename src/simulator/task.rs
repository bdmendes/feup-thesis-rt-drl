use crate::generator::Runnable;

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
    pub acet: Option<TimeUnit>, // Average Case Execution Time
    pub bcet: Option<TimeUnit>, // Best Case Execution Time
    pub next_arrival: TimeUnit,
    pub runnables: Option<Vec<Runnable>>,
}

impl SimulatorTask {
    pub fn new(task: Task, acet: TimeUnit, bcet: TimeUnit) -> Self {
        assert!(acet > 0, "Execution time must be greater than 0.");
        assert!(bcet > 0, "Execution time must be greater than 0.");
        Self {
            task: task.clone(),
            custom_priority: None,
            acet: Some(acet),
            bcet: Some(bcet),
            next_arrival: task.props().offset,
            runnables: None,
        }
    }

    pub fn new_with_runnables(task: Task, runnables: Vec<Runnable>) -> Self {
        Self {
            task: task.clone(),
            custom_priority: None,
            acet: None,
            bcet: None,
            next_arrival: task.props().offset,
            runnables: Some(runnables),
        }
    }

    pub fn new_with_custom_priority(task: Task, priority: TimeUnit, acet: TimeUnit) -> Self {
        assert!(acet > 0, "Execution time must be greater than 0.");
        Self {
            task: task.clone(),
            custom_priority: Some(priority),
            acet: Some(acet),
            bcet: None,
            next_arrival: task.props().offset,
            runnables: None,
        }
    }

    pub fn sample_execution_time(&self) -> TimeUnit {
        if let Some(runnables) = &self.runnables {
            runnables.iter().map(|r| r.sample_exec_time()).sum::<f64>() as TimeUnit
        } else {
            self.acet.unwrap()
        }
    }
}

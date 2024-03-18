pub type TaskId = u32;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Task {
    LTask(TaskProps),
    HTask(TaskProps),
    DRLAgent(TaskProps),
}

impl Task {
    pub fn props(&self) -> TaskProps {
        match self {
            Task::LTask(props) => props.clone(),
            Task::HTask(props) => props.clone(),
            Task::DRLAgent(props) => props.clone(),
        }
    }

    pub fn sample_execution_time(expected_execution_time: u32) -> u32 {
        // TODO: Implement random execution time
        expected_execution_time
    }

    pub fn activate(&self) {}
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TaskProps {
    pub id: TaskId,
    pub wcet_l: u32,
    pub wcet_h: u32,
    pub offset: u32,
    pub period: u32,
}

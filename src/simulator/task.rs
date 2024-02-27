pub type TaskId = u32;

#[derive(Debug, Clone)]
pub enum Task {
    LTask(TaskProps),
    HTask(TaskProps),
    DRLAgent {
        id: TaskId,
        period: u32,
        deadline: u32,
    },
}

impl Task {
    pub fn id(&self) -> Option<TaskId> {
        match self {
            Task::LTask(props) => Some(props.id),
            Task::HTask(props) => Some(props.id),
            Task::DRLAgent { id, .. } => Some(*id),
        }
    }

    pub fn sample_execution_time(expected_execution_time: u32) -> u32 {
        // TODO: Implement random execution time
        expected_execution_time
    }
}

#[derive(Debug, Clone)]
pub struct TaskProps {
    pub id: TaskId,
    pub wcet_l: u32,
    pub wcet_h: u32,
    pub offset: u32,
    pub period: u32,
    pub deadline: u32,
}

pub type TaskId = u32;

pub enum Task {
    LTask(TaskProps),
    HTask(TaskProps),
    DRLAgent {
        id: TaskId,
        period: u32,
        deadline: u32,
    },
}

pub struct TaskProps {
    pub id: TaskId,
    pub wcet_l: u32,
    pub wcet_h: u32,
    pub offset: u32,
    pub period: u32,
    pub deadline: u32,
}

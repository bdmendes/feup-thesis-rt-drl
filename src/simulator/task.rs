use super::SimulatorMode;

pub type TaskId = u32;

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

    pub fn sample_execution_time(expected_execution_time: u32) -> u32 {
        // TODO: Implement random execution time
        expected_execution_time
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TaskProps {
    pub id: TaskId,
    pub wcet_l: u32,
    pub wcet_h: u32,
    pub offset: u32,
    pub period: u32,
}

impl TaskProps {
    pub fn wcet_in_mode(&self, mode: SimulatorMode) -> u32 {
        match mode {
            SimulatorMode::LMode => self.wcet_l,
            SimulatorMode::HMode => self.wcet_h,
        }
    }
}

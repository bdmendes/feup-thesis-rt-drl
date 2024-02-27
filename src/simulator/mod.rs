use self::task::{Task, TaskId};

pub mod task;

pub struct SimulatorTask {
    // Task properties inside the simulator
    task: Task,
    priority: u32,
    expected_execution_time: u32,

    // Internally changed during simulation
    remaining_execution_time: u32,
    last_activation_instant: u32,
    running_for: u32,
    next_activation_instant: u32,
}

impl SimulatorTask {
    pub fn new(task: Task, priority: u32, expected_execution_time: u32) -> Self {
        Self {
            task: task.clone(),
            priority,
            expected_execution_time,
            remaining_execution_time: 0,
            last_activation_instant: 0,
            running_for: 0,
            next_activation_instant: match task {
                Task::LTask(props) => props.offset,
                Task::HTask(props) => props.offset,
                _ => unimplemented!("DRLAgent not implemented"),
            },
        }
    }
}

#[derive(Debug, PartialEq, Copy, Clone)]
pub enum SimulatorMode {
    LMode,
    HMode,
}

pub struct Simulator {
    pub mode: SimulatorMode,
    pub tasks: Vec<SimulatorTask>,
    pub random_execution_time: bool,
}

impl Simulator {
    pub fn run(&mut self, duration: u32) -> Option<Vec<(Option<TaskId>, SimulatorMode)>> {
        self.tasks.sort_by(|a, b| a.priority.cmp(&b.priority));

        let mut running_tasks_ids = Vec::with_capacity(duration as usize);
        let mut current_running_task = None;

        for current_time in 0..duration {
            if let Some(task_id) = current_running_task {
                if let Some(task) = self.tasks.iter_mut().find(|t| t.task.id() == Some(task_id)) {
                    // Discount the time from the task.
                    task.remaining_execution_time -= 1;
                    task.running_for += 1;

                    let wcet_l = match &task.task {
                        Task::LTask(props) => props.wcet_l,
                        Task::HTask(props) => props.wcet_l,
                        _ => panic!("Invalid task type"),
                    };
                    if self.mode == SimulatorMode::LMode && task.running_for > wcet_l {
                        // The task running time has surpassed its WCET_L.
                        // Switch to HMode immediately.
                        self.mode = SimulatorMode::HMode;
                    }
                }
            }

            // Select the next task to run.
            for task in &mut self.tasks {
                let eligible = match self.mode {
                    SimulatorMode::LMode => true,
                    SimulatorMode::HMode => matches!(task.task, Task::HTask(_)),
                };
                if !eligible {
                    continue;
                }

                if let Task::DRLAgent { .. } = task.task {
                    unimplemented!("DRLAgent not implemented");
                }

                let props = match &task.task {
                    Task::LTask(props) => props,
                    Task::HTask(props) => props,
                    _ => panic!("Invalid task type"),
                };

                let unfinished = task.remaining_execution_time != 0;
                let surpassed_deadline =
                    unfinished && current_time - task.last_activation_instant >= props.deadline;

                if surpassed_deadline {
                    // The system is not schedulable.
                    return None;
                }

                if unfinished {
                    // Run the task.
                    running_tasks_ids.push((Some(props.id), self.mode));
                    current_running_task = Some(props.id);
                    break;
                } else if current_time >= task.next_activation_instant {
                    // Activate the task.
                    task.remaining_execution_time = if self.random_execution_time {
                        Task::sample_execution_time(task.expected_execution_time)
                    } else {
                        task.expected_execution_time
                    };
                    task.last_activation_instant = current_time;
                    task.running_for = 0;
                    running_tasks_ids.push((Some(props.id), self.mode));
                    current_running_task = Some(props.id);
                    task.next_activation_instant += props.period;
                    break;
                }
            }

            // If we have not pushed any task, push None to indicate that no task is running.
            // We should also switch to LMode.
            if running_tasks_ids.len() <= current_time as usize {
                running_tasks_ids.push((None, self.mode));
                self.mode = SimulatorMode::LMode;
                current_running_task = None;
            }
        }

        Some(running_tasks_ids)
    }
}

#[cfg(test)]
mod tests {
    use super::{task::TaskProps, Simulator, SimulatorTask};

    #[test]

    fn same_mode_feasible() {
        let task1 = SimulatorTask::new(
            super::task::Task::LTask(TaskProps {
                id: 1,
                wcet_l: 1,
                wcet_h: 1,
                offset: 1,
                period: 4,
                deadline: 3,
            }),
            1,
            1,
        );
        let task2 = SimulatorTask::new(
            super::task::Task::LTask(TaskProps {
                id: 2,
                wcet_l: 2,
                wcet_h: 2,
                offset: 0,
                period: 4,
                deadline: 3,
            }),
            2,
            2,
        );

        let mut simulator = Simulator {
            mode: super::SimulatorMode::LMode,
            tasks: vec![task1, task2],
            random_execution_time: false,
        };
        let running_tasks = simulator.run(10).unwrap();

        assert_eq!(
            running_tasks,
            vec![
                (Some(2), super::SimulatorMode::LMode),
                (Some(1), super::SimulatorMode::LMode),
                (Some(2), super::SimulatorMode::LMode),
                (None, super::SimulatorMode::LMode),
                (Some(2), super::SimulatorMode::LMode),
                (Some(1), super::SimulatorMode::LMode),
                (Some(2), super::SimulatorMode::LMode),
                (None, super::SimulatorMode::LMode),
                (Some(2), super::SimulatorMode::LMode),
                (Some(1), super::SimulatorMode::LMode),
            ]
        );
    }

    #[test]
    fn different_modes_feasible_no_mode_change() {
        let task1 = SimulatorTask::new(
            super::task::Task::HTask(TaskProps {
                id: 1,
                wcet_l: 1,
                wcet_h: 1,
                offset: 1,
                period: 3,
                deadline: 3,
            }),
            1,
            1,
        );
        let task2 = SimulatorTask::new(
            super::task::Task::LTask(TaskProps {
                id: 2,
                wcet_l: 2,
                wcet_h: 2,
                offset: 0,
                period: 3,
                deadline: 3,
            }),
            2,
            2,
        );

        let mut simulator = Simulator {
            mode: super::SimulatorMode::LMode,
            tasks: vec![task1, task2],
            random_execution_time: false,
        };
        let running_tasks = simulator.run(8).unwrap();

        assert_eq!(
            running_tasks,
            vec![
                (Some(2), super::SimulatorMode::LMode),
                (Some(1), super::SimulatorMode::LMode),
                (Some(2), super::SimulatorMode::LMode),
                (Some(2), super::SimulatorMode::LMode),
                (Some(1), super::SimulatorMode::LMode),
                (Some(2), super::SimulatorMode::LMode),
                (Some(2), super::SimulatorMode::LMode),
                (Some(1), super::SimulatorMode::LMode),
            ]
        );
    }

    #[test]
    fn different_modes_feasible_mode_change() {
        let task1 = SimulatorTask::new(
            super::task::Task::LTask(TaskProps {
                id: 1,
                wcet_l: 2,
                wcet_h: 0,
                offset: 0,
                period: 5,
                deadline: 5,
            }),
            1,
            3,
        );
        let task2 = SimulatorTask::new(
            super::task::Task::HTask(TaskProps {
                id: 2,
                wcet_l: 2,
                wcet_h: 3,
                offset: 2,
                period: 5,
                deadline: 3,
            }),
            2,
            2,
        );

        let mut simulator = Simulator {
            mode: super::SimulatorMode::LMode,
            tasks: vec![task1, task2],
            random_execution_time: false,
        };
        let running_tasks = simulator.run(12).unwrap();

        assert_eq!(
            running_tasks,
            vec![
                (Some(1), super::SimulatorMode::LMode),
                (Some(1), super::SimulatorMode::LMode),
                (Some(1), super::SimulatorMode::LMode),
                (Some(2), super::SimulatorMode::HMode),
                (Some(2), super::SimulatorMode::HMode),
                (None, super::SimulatorMode::HMode),
                (Some(1), super::SimulatorMode::LMode),
                (Some(1), super::SimulatorMode::LMode),
                (Some(1), super::SimulatorMode::LMode),
                (Some(2), super::SimulatorMode::HMode),
                (Some(2), super::SimulatorMode::HMode),
            ]
        );
    }
}

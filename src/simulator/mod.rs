use self::task::{Task, TaskId};

pub mod task;

pub struct SimulatorTask {
    pub task: Task,
    pub priority: u32,
    pub expected_execution_time: u32,
    pub remaining_execution_time: u32,
    pub last_activation_instant: u32,
}

#[derive(Debug, PartialEq, Copy, Clone)]
pub enum SimulatorMode {
    LMode,
    HMode,
}

pub struct Simulator {
    pub mode: SimulatorMode,
    pub tasks: Vec<SimulatorTask>,
    pub time: u32,
}

impl Simulator {
    fn run(&mut self, duration: u32) -> Option<Vec<(Option<TaskId>, SimulatorMode)>> {
        self.tasks.sort_by(|a, b| a.priority.cmp(&b.priority));

        let mut running_tasks_ids = Vec::with_capacity(duration as usize);

        let mut current_time = 0;

        loop {
            current_time += 1;

            if current_time > duration {
                // Time has run out. Finish the simulation.
                break;
            }

            for task in &mut self.tasks {
                // Discard if task is not available in this mode
                let eligible = match self.mode {
                    SimulatorMode::LMode => true,
                    SimulatorMode::HMode => matches!(task.task, Task::HTask(_)),
                };
                if !eligible {
                    continue;
                }

                // If it's the DRL agent, check if it's time to run
                if let Task::DRLAgent { id, period, .. } = task.task {
                    if current_time % period != 0 {
                        continue;
                    }
                    running_tasks_ids.push((Some(id), self.mode));
                    // TODO: Execute DRL agent
                    break;
                }

                // If it's a regular task, check if it's time to run
                let props = match &task.task {
                    Task::LTask(props) => props,
                    Task::HTask(props) => props,
                    _ => panic!("Invalid task type"),
                };
                if task.remaining_execution_time != 0 {
                    if current_time % (props.period + props.offset) == 0 {
                        // This task is ready to run and previous instance is still running.
                        // The system is unschedulable.
                        return None;
                    }

                    if current_time - task.last_activation_instant >= props.deadline {
                        // This task has missed its deadline.
                        match self.mode {
                            SimulatorMode::LMode => {
                                // Switch to HMode.
                                self.mode = SimulatorMode::HMode;
                                current_time -= 1;
                                break;
                            }
                            SimulatorMode::HMode => {
                                // The system is unschedulable.
                                return None;
                            }
                        }
                    }

                    // Run the task.
                    task.remaining_execution_time -= 1;
                    running_tasks_ids.push((Some(props.id), self.mode));
                    break;
                } else if current_time % (props.period + props.offset) == 0 {
                    // Run the task.
                    // TODO: Estimate an execution time around this time.
                    task.remaining_execution_time = task.expected_execution_time;
                    task.last_activation_instant = current_time;
                    running_tasks_ids.push((Some(props.id), self.mode));
                    break;
                }
            }

            // If we have not pushed any task, push None to indicate that no task is running.
            if running_tasks_ids.len() < current_time as usize {
                running_tasks_ids.push((None, self.mode));
            }
        }

        Some(running_tasks_ids)
    }
}

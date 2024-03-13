use self::task::{Task, TaskId};

pub mod task;

#[derive(Debug, Clone, Copy)]
pub struct SimulatorTask {
    task: Task,
    priority: u32,
    expected_execution_time: u32,
}

impl SimulatorTask {
    pub fn new(task: Task, priority: u32, expected_execution_time: u32) -> Self {
        Self {
            task: task.clone(),
            priority,
            expected_execution_time,
        }
    }
}

#[derive(Debug, PartialEq, Copy, Clone)]
pub enum SimulatorMode {
    LMode,
    HMode,
}

#[derive(Debug, Copy, Clone)]
pub enum TaskEvent {
    Start(SimulatorTask, u32),
    End(SimulatorTask, u32),
}

pub struct Simulator {
    pub tasks: Vec<SimulatorTask>,
    pub random_execution_time: bool,
}

impl Simulator {
    pub fn run(&mut self, duration: u32) -> Option<Vec<(Option<TaskId>, SimulatorMode)>> {
        let mut run_history = vec![];
        let mut current_running_tasks = vec![];
        let mut current_mode = SimulatorMode::LMode;

        // Prepare task starts for the duration of the simulation.
        // The event queue is sorted by task priority.
        self.tasks.sort_by(|a, b| a.priority.cmp(&b.priority));
        let mut task_events = self
            .tasks
            .clone()
            .into_iter()
            .flat_map(|task| {
                let props = task.task.props();
                (props.offset..duration)
                    .step_by(props.period as usize)
                    .map(move |activation_time| TaskEvent::Start(task, activation_time))
            })
            .collect::<Vec<_>>();

        // Run the simulation.
        for time in 0..duration {
            // Remove tasks ending now
            current_running_tasks.retain(|(task, _): &(SimulatorTask, u32)| {
                task_events
                    .clone()
                    .into_iter()
                    .find(|event| match event {
                        TaskEvent::End(t, instant) => {
                            t.task.props().id == task.task.props().id && time == *instant
                        }
                        _ => false,
                    })
                    .is_none()
            });
            task_events.retain(|event| match event {
                TaskEvent::End(_, instant) => time != *instant,
                _ => true,
            });

            // Add tasks starting now, according to the current mode
            let new_tasks = task_events
                .clone()
                .into_iter()
                .filter(|event| match event {
                    TaskEvent::Start(task, instant) => {
                        time == *instant
                            && (matches!(current_mode, SimulatorMode::LMode)
                                || matches!(task.task, Task::HTask(_)))
                    }
                    _ => false,
                })
                .map(|event| match event {
                    TaskEvent::Start(task, _) => (task, 0),
                    _ => unreachable!(),
                })
                .collect::<Vec<_>>();

            current_running_tasks.extend(&new_tasks);
            current_running_tasks.sort_by(|a, b| a.0.priority.cmp(&b.0.priority));

            // Set end time for each task starting now
            for task in new_tasks.clone() {
                // TODO: Implement random execution time
                let end_time = time + task.0.expected_execution_time;
                task_events.push(TaskEvent::End(task.0, end_time));
                println!(
                    "Task {} activated at {} and will end at {}",
                    task.0.task.props().id,
                    time,
                    end_time
                );
            }

            // Adjust preempted tasks end times
            for new_task in new_tasks {
                for old_task in current_running_tasks.clone() {
                    if new_task.0.priority < old_task.0.priority {
                        // New task preempts old task, so shift old task end time
                        let old_task_end_time_event = task_events
                            .iter_mut()
                            .find(|event| match event {
                                TaskEvent::End(t, _) => {
                                    t.task.props().id == old_task.0.task.props().id
                                }
                                _ => false,
                            })
                            .unwrap();
                        let old_task_end_time = match old_task_end_time_event {
                            TaskEvent::End(_, instant) => instant,
                            _ => unreachable!(),
                        };
                        *old_task_end_time_event = TaskEvent::End(
                            old_task.0,
                            // TODO: Implement random execution time
                            *old_task_end_time + new_task.0.expected_execution_time,
                        );
                        println!(
                            "Task {} preempted by {} at {} and will now end at {}",
                            old_task.0.task.props().id,
                            new_task.0.task.props().id,
                            time,
                            time + old_task.0.expected_execution_time
                        );
                    } else if new_task.0.priority > old_task.0.priority {
                        // Old task preempts new task, so shift new task end time
                        let new_task_end_time_event = task_events
                            .iter_mut()
                            .find(|event| match event {
                                TaskEvent::End(t, _) => {
                                    t.task.props().id == new_task.0.task.props().id
                                }
                                _ => false,
                            })
                            .unwrap();
                        let new_task_end_time = match new_task_end_time_event {
                            TaskEvent::End(_, instant) => instant,
                            _ => unreachable!(),
                        };
                        *new_task_end_time_event = TaskEvent::End(
                            new_task.0,
                            *new_task_end_time + old_task.0.expected_execution_time - old_task.1,
                        );
                        println!(
                            "Task {} will not run yet, preempted by {} at {} and will now end at {}",
                            new_task.0.task.props().id,
                            old_task.0.task.props().id,
                            time,
                            time + old_task.0.expected_execution_time
                        );
                    }
                }
            }

            // Run the most prioritary task for this instant.
            let running_task = current_running_tasks.first_mut();
            if let Some((_, running_for)) = running_task {
                *running_for += 1;
                run_history.push((Some(running_task.unwrap().0.task.props().id), current_mode));
            }

            let running_task = current_running_tasks.first().cloned();
            if let Some((task, running_for)) = running_task {
                // If the task has surpassed its worst case execution time in L-mode, switch mode
                if matches!(current_mode, SimulatorMode::LMode)
                    && running_for > task.task.props().wcet_l
                {
                    current_mode = SimulatorMode::HMode;
                    current_running_tasks.retain(|(t, _)| matches!(t.task, Task::HTask(_)));
                    println!("Switching to HMode at instant {}", time);
                }
                // If there is another job of the same task already started, the system is not schedulable.
                if current_running_tasks
                    .iter()
                    .filter(|(t, _)| t.task.props().id == task.task.props().id)
                    .count()
                    > 1
                {
                    return None;
                }
            } else {
                // No task is running. Switch to LMode.
                current_mode = SimulatorMode::LMode;
                run_history.push((None, current_mode));
                println!("Switching to LMode at instant {}", time);
            }

            println!(
                "task start events pending: {:?}",
                task_events
                    .iter()
                    .filter(|e| matches!(e, TaskEvent::Start(_, _)))
                    .map(|e| match e {
                        TaskEvent::Start(t, instant) => (t.task.props().id, *instant),
                        _ => unreachable!(),
                    })
                    .collect::<Vec<_>>()
            );
        }

        Some(run_history)
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
                (None, super::SimulatorMode::LMode),
                (None, super::SimulatorMode::LMode),
                (Some(2), super::SimulatorMode::LMode),
                (Some(2), super::SimulatorMode::LMode),
                (None, super::SimulatorMode::LMode),
                (Some(1), super::SimulatorMode::LMode),
                (Some(1), super::SimulatorMode::LMode),
            ]
        );
    }
}

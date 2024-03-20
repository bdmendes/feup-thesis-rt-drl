use super::{task::Task, SimulatorMode, SimulatorTask};

pub fn feasible_schedule(tasks: &Vec<SimulatorTask>) -> bool {
    feasible_in_mode(tasks, SimulatorMode::LMode)
        && feasible_in_mode(tasks, SimulatorMode::HMode)
        && feasible_mode_changes(tasks)
}

fn wcet_in_mode(task: &Task, mode: SimulatorMode) -> u32 {
    match mode {
        SimulatorMode::LMode => task.props().wcet_l,
        SimulatorMode::HMode => task.props().wcet_h,
    }
}

fn response_time(
    task: &SimulatorTask,
    tasks: &Vec<SimulatorTask>,
    mode: SimulatorMode,
) -> Option<u32> {
    let wcet = wcet_in_mode(task.task, mode);
    let mut response_time = wcet as f32;

    for _ in 0..100 {
        let higher_priority_tasks = tasks.into_iter().filter(|t| t.priority < task.priority);
        let interference = higher_priority_tasks
            .map(|t| {
                (response_time as f32 / t.task.props().period as f32).ceil()
                    * wcet_in_mode(t.task, mode) as f32
            })
            .sum::<f32>();

        let new_response_time = wcet as f32 + interference;
        if new_response_time == response_time {
            return Some(new_response_time.ceil() as u32);
        } else {
            response_time = new_response_time;
        }
    }

    return None;
}

fn feasible_in_mode(tasks: &Vec<SimulatorTask>, mode: SimulatorMode) -> bool {
    let eligible_tasks = match mode {
        SimulatorMode::LMode => tasks.clone(),
        SimulatorMode::HMode => tasks
            .clone()
            .into_iter()
            .filter(|t| matches!(t.task, Task::HTask(_)))
            .collect::<Vec<_>>(),
    };

    for task in &eligible_tasks {
        if let Some(response_time) = response_time(&task, &eligible_tasks, mode) {
            if response_time > task.task.props().period {
                return false;
            }
        } else {
            return false;
        }
    }

    true
}

/// As per "Response-Time Analysis for Mixed Criticality Systems" (2011).
/// This calculates the response time during mode changes in AMC,
/// and ensures Ri > Ti for each HTask.
fn response_time_in_mode_changes(task: &SimulatorTask, tasks: &Vec<SimulatorTask>) -> Option<u32> {
    if !matches!(task.task, Task::HTask(_)) {
        return None;
    }

    let mut total_response_time = wcet_in_mode(task.task, SimulatorMode::HMode);

    let interference_by_ltasks = tasks
        .iter()
        .filter(|t| !matches!(t.task, Task::HTask(_)) && t.priority < task.priority)
        .map(|t| {
            ((response_time(task, tasks, SimulatorMode::LMode).unwrap() as f32)
                / t.task.props().period as f32)
                .ceil() as u32
                * wcet_in_mode(t.task, SimulatorMode::LMode)
        })
        .sum::<u32>();

    for _ in 0..100 {
        let interference_by_htasks = tasks
            .iter()
            .filter(|t| matches!(t.task, Task::HTask(_)) && t.priority < task.priority)
            .map(|t| {
                (total_response_time as f32 / t.task.props().period as f32).ceil() as u32
                    * wcet_in_mode(t.task, SimulatorMode::HMode)
            })
            .sum::<u32>();

        let new_total_response_time = wcet_in_mode(task.task, SimulatorMode::HMode)
            + interference_by_htasks
            + interference_by_ltasks;

        if new_total_response_time == total_response_time {
            return Some(new_total_response_time);
        } else {
            total_response_time = new_total_response_time;
        }
    }

    None
}

fn feasible_mode_changes(tasks: &Vec<SimulatorTask>) -> bool {
    let eligible_tasks = tasks
        .iter()
        .cloned()
        .filter(|t| matches!(t.task, Task::HTask(_)))
        .collect::<Vec<_>>();

    for task in &eligible_tasks {
        if let Some(response_time) = response_time_in_mode_changes(&task, &eligible_tasks) {
            if response_time > task.task.props().period {
                return false;
            }
        } else {
            return false;
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use crate::simulator::{
        schedulability::{
            feasible_in_mode, feasible_mode_changes, response_time, response_time_in_mode_changes,
        },
        task::TaskProps,
        SimulatorTask,
    };

    #[test]
    fn feasible_in_mode_1() {
        let task1 = SimulatorTask {
            task: &crate::simulator::task::Task::LTask(TaskProps {
                id: 1,
                wcet_l: 4,
                wcet_h: 4,
                offset: 0,
                period: 8,
            }),
            priority: 1,
            expected_execution_time: 0,
        };
        let task2 = SimulatorTask {
            task: &crate::simulator::task::Task::LTask(TaskProps {
                id: 2,
                wcet_l: 2,
                wcet_h: 2,
                offset: 0,
                period: 8,
            }),
            priority: 2,
            expected_execution_time: 0,
        };
        let task3 = SimulatorTask {
            task: &crate::simulator::task::Task::LTask(TaskProps {
                id: 3,
                wcet_l: 2,
                wcet_h: 2,
                offset: 0,
                period: 8,
            }),
            priority: 3,
            expected_execution_time: 0,
        };

        let tasks = vec![task1, task2, task3];

        assert_eq!(
            response_time(&task1, &tasks, crate::simulator::SimulatorMode::LMode),
            Some(4)
        );
        assert_eq!(
            response_time(&task2, &tasks, crate::simulator::SimulatorMode::LMode),
            Some(6)
        );
        assert_eq!(
            response_time(&task3, &tasks, crate::simulator::SimulatorMode::LMode),
            Some(8)
        );

        assert!(feasible_in_mode(
            &tasks,
            crate::simulator::SimulatorMode::LMode
        ));
    }

    #[test]
    fn non_feasible_in_mode_1() {
        let task1 = SimulatorTask {
            task: &crate::simulator::task::Task::LTask(TaskProps {
                id: 1,
                wcet_l: 4,
                wcet_h: 4,
                offset: 0,
                period: 8,
            }),
            priority: 1,
            expected_execution_time: 0,
        };
        let task2 = SimulatorTask {
            task: &crate::simulator::task::Task::LTask(TaskProps {
                id: 2,
                wcet_l: 2,
                wcet_h: 2,
                offset: 0,
                period: 8,
            }),
            priority: 2,
            expected_execution_time: 0,
        };
        let task3 = SimulatorTask {
            task: &crate::simulator::task::Task::LTask(TaskProps {
                id: 3,
                wcet_l: 3,
                wcet_h: 3,
                offset: 0,
                period: 8,
            }),
            priority: 3,
            expected_execution_time: 0,
        };

        let tasks = vec![task1, task2, task3];

        assert_eq!(
            response_time(&task1, &tasks, crate::simulator::SimulatorMode::LMode),
            Some(4)
        );
        assert_eq!(
            response_time(&task2, &tasks, crate::simulator::SimulatorMode::LMode),
            Some(6)
        );
        assert_eq!(
            response_time(&task3, &tasks, crate::simulator::SimulatorMode::LMode),
            Some(15)
        );

        assert!(!feasible_in_mode(
            &tasks,
            crate::simulator::SimulatorMode::LMode
        ));
    }

    #[test]
    fn non_feasible_in_mode_2() {
        let task1 = SimulatorTask {
            task: &crate::simulator::task::Task::HTask(TaskProps {
                id: 1,
                wcet_l: 1,
                wcet_h: 1,
                offset: 0,
                period: 8,
            }),
            priority: 1,
            expected_execution_time: 0,
        };
        let task2 = SimulatorTask {
            task: &crate::simulator::task::Task::HTask(TaskProps {
                id: 2,
                wcet_l: 1,
                wcet_h: 1,
                offset: 0,
                period: 8,
            }),
            priority: 2,
            expected_execution_time: 0,
        };
        let task3 = SimulatorTask {
            task: &crate::simulator::task::Task::LTask(TaskProps {
                id: 3,
                wcet_l: 4,
                wcet_h: 4,
                offset: 0,
                period: 10,
            }),
            priority: 3,
            expected_execution_time: 0,
        };
        let task4 = SimulatorTask {
            task: &crate::simulator::task::Task::LTask(TaskProps {
                id: 4,
                wcet_l: 2,
                wcet_h: 2,
                offset: 0,
                period: 10,
            }),
            priority: 4,
            expected_execution_time: 0,
        };
        let task5 = SimulatorTask {
            task: &crate::simulator::task::Task::LTask(TaskProps {
                id: 5,
                wcet_l: 3,
                wcet_h: 3,
                offset: 0,
                period: 10,
            }),
            priority: 5,
            expected_execution_time: 0,
        };

        let tasks = vec![task1, task2, task3, task4, task5];

        assert_eq!(
            response_time(&task1, &tasks, crate::simulator::SimulatorMode::HMode),
            Some(1)
        );
        assert_eq!(
            response_time(&task2, &tasks, crate::simulator::SimulatorMode::HMode),
            Some(2)
        );

        assert_eq!(
            response_time(&task1, &tasks, crate::simulator::SimulatorMode::LMode),
            Some(1)
        );
        assert_eq!(
            response_time(&task2, &tasks, crate::simulator::SimulatorMode::LMode),
            Some(2)
        );
        assert_eq!(
            response_time(&task3, &tasks, crate::simulator::SimulatorMode::LMode),
            Some(6)
        );
        assert_eq!(
            response_time(&task4, &tasks, crate::simulator::SimulatorMode::LMode),
            Some(8)
        );
        assert_eq!(
            response_time(&task5, &tasks, crate::simulator::SimulatorMode::LMode),
            Some(29)
        );

        assert!(feasible_in_mode(
            &tasks,
            crate::simulator::SimulatorMode::HMode
        ));

        assert!(!feasible_in_mode(
            &tasks,
            crate::simulator::SimulatorMode::LMode
        ));
    }

    #[test]
    fn feasible_mode_change_1() {
        let task1 = SimulatorTask {
            task: &crate::simulator::task::Task::HTask(TaskProps {
                id: 1,
                wcet_l: 3,
                wcet_h: 4,
                offset: 0,
                period: 8,
            }),
            priority: 3,
            expected_execution_time: 0,
        };
        let task2 = SimulatorTask {
            task: &crate::simulator::task::Task::LTask(TaskProps {
                id: 2,
                wcet_l: 2,
                wcet_h: 2,
                offset: 0,
                period: 8,
            }),
            priority: 1,
            expected_execution_time: 0,
        };
        let task3 = SimulatorTask {
            task: &crate::simulator::task::Task::LTask(TaskProps {
                id: 3,
                wcet_l: 2,
                wcet_h: 2,
                offset: 0,
                period: 8,
            }),
            priority: 2,
            expected_execution_time: 0,
        };

        let tasks = vec![task1, task2, task3];

        assert_eq!(response_time_in_mode_changes(&task1, &tasks,), Some(8));

        assert!(feasible_mode_changes(&tasks));
    }

    #[test]
    fn feasible_mode_change_2() {
        let task1 = SimulatorTask {
            task: &crate::simulator::task::Task::HTask(TaskProps {
                id: 1,
                wcet_l: 3,
                wcet_h: 4,
                offset: 0,
                period: 8,
            }),
            priority: 3,
            expected_execution_time: 0,
        };
        let task2 = SimulatorTask {
            task: &crate::simulator::task::Task::HTask(TaskProps {
                id: 2,
                wcet_l: 2,
                wcet_h: 2,
                offset: 0,
                period: 8,
            }),
            priority: 1,
            expected_execution_time: 0,
        };
        let task3 = SimulatorTask {
            task: &crate::simulator::task::Task::LTask(TaskProps {
                id: 3,
                wcet_l: 2,
                wcet_h: 2,
                offset: 0,
                period: 8,
            }),
            priority: 2,
            expected_execution_time: 0,
        };

        let tasks = vec![task1, task2, task3];

        assert_eq!(response_time_in_mode_changes(&task1, &tasks,), Some(8));
        assert_eq!(response_time_in_mode_changes(&task2, &tasks,), Some(2));

        assert!(feasible_mode_changes(&tasks));
    }
}
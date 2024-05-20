use super::{
    task::{Task, TimeUnit},
    SimulatorMode, SimulatorTask,
};

pub fn feasible_schedule_design_time(tasks: &[SimulatorTask]) -> bool {
    // At design time, we assess the full recurrence for testing the AMC feasibility.
    feasible_in_mode(tasks, SimulatorMode::LMode)
        && feasible_in_mode(tasks, SimulatorMode::HMode)
        && feasible_mode_changes::<false>(tasks)
}

pub fn feasible_schedule_online(tasks: &[SimulatorTask]) -> bool {
    // At runtime, we have no "time" to calculate the full recurrence.
    // Therefore, we assume Ri=Ti which is the worst case scenario.
    feasible_in_mode(tasks, SimulatorMode::LMode) && feasible_mode_changes::<true>(tasks)
}

fn response_time(
    task: &SimulatorTask,
    tasks: &[SimulatorTask],
    mode: SimulatorMode,
) -> Option<TimeUnit> {
    let wcet = task.task.props().wcet_in_mode(mode);
    let mut response_time = wcet as f32;

    for _ in 0..100 {
        let higher_priority_tasks = tasks.iter().filter(|t| t.priority < task.priority);
        let interference = higher_priority_tasks
            .map(|t| {
                (response_time / t.task.props().period as f32).ceil()
                    * t.task.props().wcet_in_mode(mode) as f32
            })
            .sum::<f32>();

        let new_response_time = wcet as f32 + interference;
        if new_response_time == response_time {
            return Some(new_response_time.ceil() as TimeUnit);
        } else {
            response_time = new_response_time;
        }
    }

    None
}

fn feasible_in_mode(tasks: &[SimulatorTask], mode: SimulatorMode) -> bool {
    let eligible_tasks = match mode {
        SimulatorMode::LMode => tasks.to_vec(),
        SimulatorMode::HMode => tasks
            .iter()
            .filter(|t| matches!(t.task, Task::HTask(_)))
            .map(|t| t.to_owned())
            .collect::<Vec<_>>(),
    };

    for task in &eligible_tasks {
        if task.task.props().wcet_in_mode(mode) == 0 {
            return false;
        }

        if let Some(response_time) = response_time(task, &eligible_tasks, mode) {
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
fn response_time_in_mode_changes<const APPROXIMATE: bool>(
    task: &SimulatorTask,
    tasks: &[SimulatorTask],
) -> Option<TimeUnit> {
    if !matches!(task.task, Task::HTask(_)) {
        return None;
    }

    let interference_by_ltasks = tasks
        .iter()
        .filter(|t| !matches!(t.task, Task::HTask(_)) && t.priority < task.priority)
        .map(|t| {
            ((response_time(task, tasks, SimulatorMode::LMode).unwrap() as f32)
                / t.task.props().period as f32)
                .ceil() as TimeUnit
                * t.task.props().wcet_in_mode(SimulatorMode::LMode)
        })
        .sum::<TimeUnit>();

    if APPROXIMATE {
        let interference_by_htasks = tasks
            .iter()
            .filter(|t| matches!(t.task, Task::HTask(_)) && t.priority < task.priority)
            .map(|t| {
                (task.task.props().period as f32 / t.task.props().period as f32).ceil() as TimeUnit
                    * t.task.props().wcet_in_mode(SimulatorMode::HMode)
            })
            .sum::<TimeUnit>();

        return Some(
            task.task.props().wcet_in_mode(SimulatorMode::HMode)
                + interference_by_ltasks
                + interference_by_htasks,
        );
    }

    let mut total_response_time = task.task.props().wcet_in_mode(SimulatorMode::HMode);

    for _ in 0..100 {
        let interference_by_htasks = tasks
            .iter()
            .filter(|t| matches!(t.task, Task::HTask(_)) && t.priority < task.priority)
            .map(|t| {
                (total_response_time as f32 / t.task.props().period as f32).ceil() as TimeUnit
                    * t.task.props().wcet_in_mode(SimulatorMode::HMode)
            })
            .sum::<TimeUnit>();

        let new_total_response_time = task.task.props().wcet_in_mode(SimulatorMode::HMode)
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

fn feasible_mode_changes<const APPROXIMATE: bool>(tasks: &[SimulatorTask]) -> bool {
    let eligible_tasks = tasks
        .iter()
        .filter(|t| matches!(t.task, Task::HTask(_)))
        .map(|t| t.to_owned())
        .collect::<Vec<_>>();

    for task in &eligible_tasks {
        if let Some(response_time) =
            response_time_in_mode_changes::<APPROXIMATE>(task, eligible_tasks.as_slice())
        {
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
        task::{TaskProps, TimeUnit},
        validation::{
            feasible_in_mode, feasible_mode_changes, response_time, response_time_in_mode_changes,
        },
        SimulatorTask,
    };

    const UNUSED_TIME: TimeUnit = TimeUnit::MAX;

    #[test]
    fn feasible_in_mode_1() {
        let task1 = SimulatorTask::new_with_custom_priority(
            crate::simulator::task::Task::LTask(TaskProps {
                id: 1,
                wcet_l: 4,
                wcet_h: 4,
                offset: 0,
                period: 8,
            }),
            1,
            UNUSED_TIME,
        );
        let task2 = SimulatorTask::new_with_custom_priority(
            crate::simulator::task::Task::LTask(TaskProps {
                id: 2,
                wcet_l: 2,
                wcet_h: 2,
                offset: 0,
                period: 8,
            }),
            2,
            UNUSED_TIME,
        );
        let task3 = SimulatorTask::new_with_custom_priority(
            crate::simulator::task::Task::LTask(TaskProps {
                id: 3,
                wcet_l: 2,
                wcet_h: 2,
                offset: 0,
                period: 8,
            }),
            3,
            UNUSED_TIME,
        );

        let tasks = vec![task1.clone(), task2.clone(), task3.clone()];

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
        let task1 = SimulatorTask::new_with_custom_priority(
            crate::simulator::task::Task::LTask(TaskProps {
                id: 1,
                wcet_l: 4,
                wcet_h: 4,
                offset: 0,
                period: 8,
            }),
            1,
            UNUSED_TIME,
        );
        let task2 = SimulatorTask::new_with_custom_priority(
            crate::simulator::task::Task::LTask(TaskProps {
                id: 2,
                wcet_l: 2,
                wcet_h: 2,
                offset: 0,
                period: 8,
            }),
            2,
            UNUSED_TIME,
        );
        let task3 = SimulatorTask::new_with_custom_priority(
            crate::simulator::task::Task::LTask(TaskProps {
                id: 3,
                wcet_l: 3,
                wcet_h: 3,
                offset: 0,
                period: 8,
            }),
            3,
            UNUSED_TIME,
        );

        let tasks = vec![task1.clone(), task2.clone(), task3.clone()];

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
        let task1 = SimulatorTask::new_with_custom_priority(
            crate::simulator::task::Task::HTask(TaskProps {
                id: 1,
                wcet_l: 1,
                wcet_h: 1,
                offset: 0,
                period: 8,
            }),
            1,
            UNUSED_TIME,
        );
        let task2 = SimulatorTask::new_with_custom_priority(
            crate::simulator::task::Task::HTask(TaskProps {
                id: 2,
                wcet_l: 1,
                wcet_h: 1,
                offset: 0,
                period: 8,
            }),
            2,
            UNUSED_TIME,
        );
        let task3 = SimulatorTask::new_with_custom_priority(
            crate::simulator::task::Task::LTask(TaskProps {
                id: 3,
                wcet_l: 4,
                wcet_h: 4,
                offset: 0,
                period: 10,
            }),
            3,
            UNUSED_TIME,
        );
        let task4 = SimulatorTask::new_with_custom_priority(
            crate::simulator::task::Task::LTask(TaskProps {
                id: 4,
                wcet_l: 2,
                wcet_h: 2,
                offset: 0,
                period: 10,
            }),
            4,
            UNUSED_TIME,
        );
        let task5 = SimulatorTask::new_with_custom_priority(
            crate::simulator::task::Task::LTask(TaskProps {
                id: 5,
                wcet_l: 3,
                wcet_h: 3,
                offset: 0,
                period: 10,
            }),
            5,
            UNUSED_TIME,
        );

        let tasks = vec![
            task1.clone(),
            task2.clone(),
            task3.clone(),
            task4.clone(),
            task5.clone(),
        ];

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
        let task1 = SimulatorTask::new_with_custom_priority(
            crate::simulator::task::Task::HTask(TaskProps {
                id: 1,
                wcet_l: 3,
                wcet_h: 4,
                offset: 0,
                period: 8,
            }),
            3,
            UNUSED_TIME,
        );
        let task2 = SimulatorTask::new_with_custom_priority(
            crate::simulator::task::Task::LTask(TaskProps {
                id: 2,
                wcet_l: 2,
                wcet_h: 2,
                offset: 0,
                period: 8,
            }),
            1,
            UNUSED_TIME,
        );
        let task3 = SimulatorTask::new_with_custom_priority(
            crate::simulator::task::Task::LTask(TaskProps {
                id: 3,
                wcet_l: 2,
                wcet_h: 2,
                offset: 0,
                period: 8,
            }),
            2,
            UNUSED_TIME,
        );

        let tasks = vec![task1.clone(), task2.clone(), task3.clone()];

        assert_eq!(
            response_time_in_mode_changes::<false>(&task1, &tasks,),
            Some(8)
        );

        assert!(feasible_mode_changes::<false>(&tasks));
    }

    #[test]
    fn feasible_mode_change_2() {
        let task1 = SimulatorTask::new_with_custom_priority(
            crate::simulator::task::Task::HTask(TaskProps {
                id: 1,
                wcet_l: 3,
                wcet_h: 4,
                offset: 0,
                period: 8,
            }),
            3,
            UNUSED_TIME,
        );
        let task2 = SimulatorTask::new_with_custom_priority(
            crate::simulator::task::Task::HTask(TaskProps {
                id: 2,
                wcet_l: 2,
                wcet_h: 2,
                offset: 0,
                period: 8,
            }),
            1,
            UNUSED_TIME,
        );
        let task3 = SimulatorTask::new_with_custom_priority(
            crate::simulator::task::Task::LTask(TaskProps {
                id: 3,
                wcet_l: 2,
                wcet_h: 2,
                offset: 0,
                period: 8,
            }),
            2,
            UNUSED_TIME,
        );

        let tasks = vec![task1.clone(), task2.clone(), task3.clone()];

        assert_eq!(
            response_time_in_mode_changes::<false>(&task1, &tasks,),
            Some(8)
        );
        assert_eq!(
            response_time_in_mode_changes::<false>(&task2, &tasks,),
            Some(2)
        );

        assert!(feasible_mode_changes::<false>(&tasks));
    }
}

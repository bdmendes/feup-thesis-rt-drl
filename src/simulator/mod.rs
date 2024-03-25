use self::task::{Task, TaskId};

pub mod task;
pub mod validation;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SimulatorTask<'a> {
    task: &'a Task,
    priority: u32,
    expected_execution_time: u32,
}

impl<'a> SimulatorTask<'a> {
    pub fn new(task: &'a Task, priority: u32, expected_execution_time: u32) -> Self {
        Self {
            task,
            priority,
            expected_execution_time,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct SimulatorJob<'a> {
    task: SimulatorTask<'a>,
    running_for: u32,
    remaining: u32,
}

#[derive(Debug, PartialEq, Copy, Clone)]
pub enum SimulatorMode {
    LMode,
    HMode,
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum SimulatorEvent<'a> {
    Start(SimulatorTask<'a>, u32),
    TaskKill(SimulatorTask<'a>, u32),
    ModeChange(SimulatorMode, u32),
}

pub struct Simulator<'a> {
    pub tasks: Vec<SimulatorTask<'a>>,
    pub random_execution_time: bool,
}

impl<'a> Simulator<'a> {
    pub fn run(&mut self, duration: u32) -> (Option<Vec<Option<TaskId>>>, Vec<SimulatorEvent<'a>>) {
        let mut run_history = vec![];
        let mut simulator_events_history = vec![];
        let mut current_mode = SimulatorMode::LMode;
        let mut current_running_jobs = Vec::<SimulatorJob>::new();

        // Prepare the first start events.
        let mut task_start_events = self
            .tasks
            .iter()
            .flat_map(|task| {
                let props = task.task.props();
                (props.offset..duration)
                    .step_by(props.period as usize)
                    .next()
                    .map(move |activation_time| SimulatorEvent::Start(*task, activation_time))
            })
            .collect::<Vec<_>>();

        // Run the simulation.
        for time in 0..duration {
            // Determine tasks starting now, according to the current mode
            let new_jobs = task_start_events
                .iter()
                .filter(|event| match event {
                    SimulatorEvent::Start(task, instant) => {
                        time == *instant
                            && (matches!(current_mode, SimulatorMode::LMode)
                                || matches!(task.task, Task::HTask(_)))
                    }
                    _ => unreachable!(),
                })
                .map(|event| match event {
                    SimulatorEvent::Start(task, _) => SimulatorJob {
                        task: *task,
                        running_for: 0,
                        remaining: if self.random_execution_time {
                            Task::sample_execution_time(task.expected_execution_time)
                        } else {
                            task.expected_execution_time
                        },
                    },
                    _ => unreachable!(),
                })
                .collect::<Vec<_>>();

            // Add new jobs to current jobs
            current_running_jobs.extend(&new_jobs);
            current_running_jobs.sort_by(|a, b| a.task.priority.cmp(&b.task.priority));

            // Check for multiple jobs for the same tasks.
            // This can't happen since we're assuming D=T.
            for job in &current_running_jobs {
                if current_running_jobs
                    .iter()
                    .filter(|j| j.task.task.props().id == job.task.task.props().id)
                    .count()
                    > 1
                {
                    return (None, simulator_events_history);
                }
            }

            // Update the most prioritary job for this instant.
            if let Some(job) = current_running_jobs.first_mut() {
                if job.running_for == 0 {
                    // Activate the task.
                    job.task.task.activate();

                    // Schedule the next start event.
                    let this_task_start_event = task_start_events
                        .iter_mut()
                        .find(|e| match e {
                            SimulatorEvent::Start(t, _) => {
                                t.task.props().id == job.task.task.props().id
                            }
                            _ => unreachable!(),
                        })
                        .unwrap();
                    let previous_instant = match this_task_start_event {
                        SimulatorEvent::Start(_, instant) => instant,
                        _ => unreachable!(),
                    };
                    let new_instant = *previous_instant + job.task.task.props().period;
                    *this_task_start_event = SimulatorEvent::Start(job.task, new_instant);
                }

                job.running_for += 1;
                job.remaining -= 1;
                run_history.push(Some(job.task.task.props().id));
            }

            if let Some(job) = current_running_jobs.first().cloned() {
                if job.remaining == 0 {
                    // Job has ended. Remove it.
                    current_running_jobs
                        .retain(|j| j.task.task.props().id != job.task.task.props().id);
                } else if matches!(current_mode, SimulatorMode::HMode)
                    && job.running_for >= job.task.task.props().wcet_h
                {
                    // Task has surpassed its worst case execution time in H-mode.
                    // The system is not schedulable.
                    return (None, simulator_events_history);
                } else if matches!(current_mode, SimulatorMode::LMode)
                    && job.running_for >= job.task.task.props().wcet_l
                {
                    // Task has surpassed its worst case execution time in L-mode
                    if matches!(job.task.task, Task::HTask(_)) {
                        // This is a HTask. We must switch mode immediately.
                        current_mode = SimulatorMode::HMode;
                        current_running_jobs.retain(|job| matches!(job.task.task, Task::HTask(_)));
                        simulator_events_history
                            .push(SimulatorEvent::ModeChange(SimulatorMode::HMode, time));
                    } else {
                        // We could go about this in some ways, but in an attempt to try to preserve
                        // more LTasks, we only kill the current job.
                        current_running_jobs
                            .retain(|j| j.task.task.props().id != job.task.task.props().id);
                        simulator_events_history.push(SimulatorEvent::TaskKill(job.task, time));
                    }
                }
            } else {
                if current_mode != SimulatorMode::LMode {
                    // No task is running. Switch to LMode.
                    current_mode = SimulatorMode::LMode;
                    simulator_events_history
                        .push(SimulatorEvent::ModeChange(SimulatorMode::LMode, time));
                }
                run_history.push(None);
            }
        }

        (Some(run_history), simulator_events_history)
    }
}

#[cfg(test)]
mod tests {
    use crate::simulator::SimulatorEvent;

    use super::{task::TaskProps, Simulator, SimulatorTask};

    #[test]

    fn same_criticality() {
        let task1 = SimulatorTask::new(
            &super::task::Task::LTask(TaskProps {
                id: 1,
                wcet_l: 1,
                wcet_h: 1,
                offset: 1,
                period: 4,
            }),
            1,
            1,
        );
        let task2 = SimulatorTask::new(
            &super::task::Task::LTask(TaskProps {
                id: 2,
                wcet_l: 2,
                wcet_h: 2,
                offset: 0,
                period: 4,
            }),
            2,
            2,
        );

        let mut simulator = Simulator {
            tasks: vec![task1, task2],
            random_execution_time: false,
        };
        let (tasks, events) = simulator.run(10);

        assert_eq!(
            tasks.unwrap(),
            vec![
                Some(2),
                Some(1),
                Some(2),
                None,
                Some(2),
                Some(1),
                Some(2),
                None,
                Some(2),
                Some(1),
            ]
        );

        assert_eq!(events, vec![]);
    }

    #[test]

    fn same_criticality_2() {
        let task1 = SimulatorTask::new(
            &super::task::Task::LTask(TaskProps {
                id: 1,
                wcet_l: 2,
                wcet_h: 2,
                offset: 1,
                period: 5,
            }),
            2,
            2,
        );
        let task2 = SimulatorTask::new(
            &super::task::Task::LTask(TaskProps {
                id: 2,
                wcet_l: 2,
                wcet_h: 2,
                offset: 0,
                period: 5,
            }),
            3,
            2,
        );
        let task3 = SimulatorTask::new(
            &super::task::Task::LTask(TaskProps {
                id: 3,
                wcet_l: 1,
                wcet_h: 1,
                offset: 1,
                period: 5,
            }),
            1,
            1,
        );

        let mut simulator = Simulator {
            tasks: vec![task1, task2, task3],
            random_execution_time: false,
        };
        let (tasks, events) = simulator.run(10);

        assert_eq!(
            tasks.unwrap(),
            vec![
                Some(2),
                Some(3),
                Some(1),
                Some(1),
                Some(2),
                Some(2),
                Some(3),
                Some(1),
                Some(1),
                Some(2),
            ]
        );

        assert_eq!(events, vec![]);
    }

    #[test]
    fn different_criticality() {
        let task1 = SimulatorTask::new(
            &super::task::Task::HTask(TaskProps {
                id: 1,
                wcet_l: 1,
                wcet_h: 1,
                offset: 1,
                period: 3,
            }),
            1,
            1,
        );
        let task2 = SimulatorTask::new(
            &super::task::Task::LTask(TaskProps {
                id: 2,
                wcet_l: 2,
                wcet_h: 2,
                offset: 0,
                period: 3,
            }),
            2,
            2,
        );

        let mut simulator = Simulator {
            tasks: vec![task1, task2],
            random_execution_time: false,
        };
        let (tasks, events) = simulator.run(8);

        assert_eq!(
            tasks.unwrap(),
            vec![
                Some(2),
                Some(1),
                Some(2),
                Some(2),
                Some(1),
                Some(2),
                Some(2),
                Some(1),
            ]
        );

        assert_eq!(events, vec![]);
    }

    #[test]
    fn different_criticality_task_kill() {
        let task1 = SimulatorTask::new(
            &super::task::Task::LTask(TaskProps {
                id: 1,
                wcet_l: 2,
                wcet_h: 0,
                offset: 0,
                period: 5,
            }),
            1,
            3,
        );
        let task2 = SimulatorTask::new(
            &super::task::Task::HTask(TaskProps {
                id: 2,
                wcet_l: 2,
                wcet_h: 3,
                offset: 2,
                period: 5,
            }),
            2,
            2,
        );

        let mut simulator = Simulator {
            tasks: vec![task1, task2],
            random_execution_time: false,
        };
        let (tasks, events) = simulator.run(12);

        assert_eq!(
            tasks.unwrap(),
            vec![
                Some(1),
                Some(1),
                Some(2),
                Some(2),
                None,
                Some(1),
                Some(1),
                Some(2),
                Some(2),
                None,
                Some(1),
                Some(1),
            ]
        );

        assert_eq!(
            events,
            vec![
                SimulatorEvent::TaskKill(task1, 1),
                SimulatorEvent::TaskKill(task1, 6),
                SimulatorEvent::TaskKill(task1, 11),
            ]
        );
    }

    #[test]
    fn different_criticality_mode_change() {
        let task1 = SimulatorTask::new(
            &super::task::Task::HTask(TaskProps {
                id: 1,
                wcet_l: 2,
                wcet_h: 3,
                offset: 0,
                period: 5,
            }),
            1,
            3,
        );
        let task2 = SimulatorTask::new(
            &super::task::Task::LTask(TaskProps {
                id: 2,
                wcet_l: 2,
                wcet_h: 3,
                offset: 2,
                period: 5,
            }),
            2,
            2,
        );

        let mut simulator = Simulator {
            tasks: vec![task1, task2],
            random_execution_time: false,
        };
        let (tasks, events) = simulator.run(12);

        assert_eq!(
            tasks.unwrap(),
            vec![
                Some(1),
                Some(1),
                Some(1),
                None,
                None,
                Some(1),
                Some(1),
                Some(1),
                None,
                None,
                Some(1),
                Some(1),
            ]
        );

        assert_eq!(
            events,
            vec![
                SimulatorEvent::ModeChange(crate::simulator::SimulatorMode::HMode, 1),
                SimulatorEvent::ModeChange(crate::simulator::SimulatorMode::LMode, 3),
                SimulatorEvent::ModeChange(crate::simulator::SimulatorMode::HMode, 6),
                SimulatorEvent::ModeChange(crate::simulator::SimulatorMode::LMode, 8),
                SimulatorEvent::ModeChange(crate::simulator::SimulatorMode::HMode, 11)
            ]
        );
    }

    #[test]
    fn non_feasible_simple() {
        let task1 = SimulatorTask::new(
            &super::task::Task::LTask(TaskProps {
                id: 1,
                wcet_l: 1,
                wcet_h: 1,
                offset: 1,
                period: 3,
            }),
            1,
            1,
        );
        let task2 = SimulatorTask::new(
            &super::task::Task::LTask(TaskProps {
                id: 2,
                wcet_l: 3,
                wcet_h: 3,
                offset: 0,
                period: 3,
            }),
            2,
            3,
        );

        let mut simulator = Simulator {
            tasks: vec![task1, task2],
            random_execution_time: false,
        };
        let (tasks, events) = simulator.run(10);

        assert_eq!(tasks, None);
        assert_eq!(events, vec![]);
    }

    #[test]
    fn non_feasible_mode_change() {
        let task1 = SimulatorTask::new(
            &super::task::Task::HTask(TaskProps {
                id: 1,
                wcet_l: 2,
                wcet_h: 3,
                offset: 0,
                period: 4,
            }),
            1,
            3,
        );
        let task2 = SimulatorTask::new(
            &super::task::Task::HTask(TaskProps {
                id: 2,
                wcet_l: 3,
                wcet_h: 3,
                offset: 2,
                period: 5,
            }),
            2,
            2,
        );

        let mut simulator = Simulator {
            tasks: vec![task1, task2],
            random_execution_time: false,
        };
        let (tasks, events) = simulator.run(10);

        assert_eq!(tasks, None);
        assert_eq!(
            events,
            vec![SimulatorEvent::ModeChange(
                crate::simulator::SimulatorMode::HMode,
                1
            )]
        );
    }

    #[test]
    fn non_feasible_exceed_wcet_h() {
        let task1 = SimulatorTask::new(
            &super::task::Task::HTask(TaskProps {
                id: 1,
                wcet_l: 2,
                wcet_h: 3,
                offset: 0,
                period: 4,
            }),
            1,
            4,
        );
        let mut simulator = Simulator {
            tasks: vec![task1],
            random_execution_time: false,
        };
        let (tasks, events) = simulator.run(10);

        assert_eq!(tasks, None);
        assert_eq!(
            events,
            vec![SimulatorEvent::ModeChange(
                crate::simulator::SimulatorMode::HMode,
                1
            )]
        );
    }
}

use self::task::{SimulatorTask, Task, TaskId, TimeUnit};
use crate::agent::SimulatorAgent;
use std::{cell::RefCell, rc::Rc};

pub mod task;
pub mod validation;

#[derive(Debug, PartialEq, Copy, Clone)]
struct SimulatorJob {
    task_id: TaskId,
    running_for: TimeUnit,
    remaining: TimeUnit,
}

impl SimulatorJob {
    fn task<'a>(&self, simulator: &'a Simulator) -> &'a SimulatorTask {
        simulator
            .tasks
            .iter()
            .find(|t| t.task.props().id == self.task_id)
            .unwrap()
    }
}

#[derive(Debug, PartialEq, Copy, Clone)]
pub enum SimulatorMode {
    LMode,
    HMode,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SimulatorEvent {
    Start(TaskId, TimeUnit),
    TaskKill(TaskId, TimeUnit),
    ModeChange(SimulatorMode, TimeUnit),
    EndSimulation,
}

impl SimulatorEvent {
    pub fn task<'a>(&self, simulator: &'a Simulator) -> Option<&'a SimulatorTask> {
        match self {
            SimulatorEvent::Start(task_id, _) | SimulatorEvent::TaskKill(task_id, _) => Some(
                simulator
                    .tasks
                    .iter()
                    .find(|t| t.task.props().id == *task_id)
                    .unwrap(),
            ),
            _ => None,
        }
    }
}

pub struct Simulator {
    pub tasks: Vec<SimulatorTask>,
    pub random_execution_time: bool,
    pub agent: Option<Rc<RefCell<SimulatorAgent>>>,
}

impl Simulator {
    pub fn run(
        &mut self,
        duration: TimeUnit,
    ) -> (Option<Vec<Option<TaskId>>>, Vec<SimulatorEvent>) {
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
                    .map(move |activation_time| {
                        SimulatorEvent::Start(task.task.props().id, activation_time)
                    })
            })
            .collect::<Vec<_>>();

        // Run the simulation.
        for time in 0..duration {
            // Determine tasks starting now, according to the current mode
            let new_jobs = task_start_events
                .iter()
                .filter(|event| match event {
                    SimulatorEvent::Start(_, instant) => {
                        time == *instant
                            && (matches!(current_mode, SimulatorMode::LMode)
                                || matches!(event.task(self).unwrap().task, Task::HTask(_)))
                    }
                    _ => unreachable!(),
                })
                .map(|event| match event {
                    SimulatorEvent::Start(_, _) => {
                        let task = event.task(self).unwrap();
                        SimulatorJob {
                            task_id: task.task.props().id,
                            running_for: 0,
                            remaining: if self.random_execution_time {
                                Task::sample_execution_time(
                                    task.acet,
                                    task.bcet,
                                    task.task.props().wcet_h,
                                )
                            } else {
                                task.acet
                            },
                        }
                    }
                    _ => unreachable!(),
                })
                .collect::<Vec<_>>();

            // Add new jobs to current jobs
            current_running_jobs.extend(new_jobs);
            current_running_jobs.sort_by(|a, b| a.task(self).priority.cmp(&b.task(self).priority));

            // Check for multiple jobs for the same tasks.
            // This can't happen since we're assuming D=T.
            for job in &current_running_jobs {
                if current_running_jobs
                    .iter()
                    .filter(|j| j.task(self).task.props().id == job.task(self).task.props().id)
                    .count()
                    > 1
                {
                    println!("Multiple jobs of the same task; the system is not schedulable.");
                    return (None, simulator_events_history);
                }
            }

            // Update the most prioritary job for this instant.
            if let Some(job) = current_running_jobs.first_mut() {
                if job.running_for == 0 {
                    // Schedule the next start event.
                    let this_task_start_event = task_start_events
                        .iter_mut()
                        .find(|e| match e {
                            SimulatorEvent::Start(id, _) => *id == job.task(self).task.props().id,
                            _ => unreachable!(),
                        })
                        .unwrap();
                    let previous_instant = match this_task_start_event {
                        SimulatorEvent::Start(_, instant) => instant,
                        _ => unreachable!(),
                    };
                    let new_instant = *previous_instant + job.task(self).task.props().period;
                    *this_task_start_event =
                        SimulatorEvent::Start(job.task(self).task.props().id, new_instant);

                    simulator_events_history
                        .push(SimulatorEvent::Start(job.task(self).task.props().id, time));

                    if self.agent.is_some() {
                        // Signal task start to agent.
                        self.agent.as_ref().unwrap().borrow_mut().push_event(
                            SimulatorEvent::Start(job.task(self).task.props().id, time),
                        );
                    }
                }

                println!(
                    "Running task {} for instant {}",
                    job.task(self).task.props().id,
                    time
                );

                job.running_for += 1;
                job.remaining -= 1;
                run_history.push(Some(job.task(self).task.props().id));
            }

            if let Some(job) = current_running_jobs.first().cloned() {
                if job.remaining == 0 {
                    // Job has ended. Remove it.
                    current_running_jobs
                        .retain(|j| j.task(self).task.props().id != job.task(self).task.props().id);
                } else if matches!(current_mode, SimulatorMode::HMode)
                    && job.running_for >= job.task(self).task.props().wcet_h
                {
                    // Task has surpassed its worst case execution time in H-mode.
                    // The system is not schedulable.
                    println!(
                        "Task {} has surpassed its worst case execution time in H-mode.",
                        job.task(self).task.props().id
                    );
                    return (None, simulator_events_history);
                } else if matches!(current_mode, SimulatorMode::LMode)
                    && job.running_for >= job.task(self).task.props().wcet_l
                {
                    // Task has surpassed its worst case execution time in L-mode
                    if matches!(job.task(self).task, Task::HTask(_)) {
                        // This is a HTask. We must switch mode immediately.
                        println!("Task {} has surpassed its worst case execution time in L-mode. Switching to H-mode.", job.task(self).task.props().id);
                        current_mode = SimulatorMode::HMode;
                        current_running_jobs
                            .retain(|job| matches!(job.task(self).task, Task::HTask(_)));
                        simulator_events_history
                            .push(SimulatorEvent::ModeChange(SimulatorMode::HMode, time));

                        // Inform the agent of the mode change.
                        if self.agent.is_some() {
                            self.agent
                                .as_ref()
                                .unwrap()
                                .borrow_mut()
                                .push_event(SimulatorEvent::ModeChange(SimulatorMode::HMode, time));
                        }
                    } else {
                        // We could go about this in some ways, but in an attempt to try to preserve
                        // more LTasks, we only kill the current job.
                        println!("Task {} has surpassed its worst case execution time in L-mode. Killing task.", job.task(self).task.props().id);
                        current_running_jobs.retain(|j| {
                            j.task(self).task.props().id != job.task(self).task.props().id
                        });
                        simulator_events_history.push(SimulatorEvent::TaskKill(
                            job.task(self).task.props().id,
                            time,
                        ));

                        if self.agent.is_some() {
                            // Inform the agent of the task kill.
                            self.agent.as_ref().unwrap().borrow_mut().push_event(
                                SimulatorEvent::TaskKill(job.task(self).task.props().id, time),
                            );
                        }
                    }
                }
            } else {
                // No task is running. We can run the agent.
                // This will trigger event processing
                // and tasks props changes.
                println!("No task is running. Running agent.");

                if self.agent.is_some() {
                    let agent = self.agent.take().unwrap();
                    agent.borrow_mut().activate(self, &run_history);
                    self.agent = Some(agent);
                }

                if current_mode != SimulatorMode::LMode {
                    current_mode = SimulatorMode::LMode;
                    simulator_events_history
                        .push(SimulatorEvent::ModeChange(SimulatorMode::LMode, time));

                    if self.agent.is_some() {
                        // Inform the agent of the mode change.
                        self.agent
                            .as_ref()
                            .unwrap()
                            .borrow_mut()
                            .push_event(SimulatorEvent::ModeChange(SimulatorMode::LMode, time));
                    }
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

    fn assert_events_eq(events: Vec<SimulatorEvent>, expected: Vec<SimulatorEvent>) {
        let events_with_stripped_start = events
            .iter()
            .filter(|e| !matches!(e, SimulatorEvent::Start(_, _)))
            .cloned()
            .collect::<Vec<_>>();
        assert_eq!(events_with_stripped_start, expected);
    }

    #[test]

    fn same_criticality() {
        let task1 = SimulatorTask::new_with_custom_priority(
            super::task::Task::LTask(TaskProps {
                id: 1,
                wcet_l: 1,
                wcet_h: 1,
                offset: 1,
                period: 4,
            }),
            1,
            1,
        );
        let task2 = SimulatorTask::new_with_custom_priority(
            super::task::Task::LTask(TaskProps {
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
            agent: None,
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

        assert_events_eq(events, vec![]);
    }

    #[test]

    fn same_criticality_2() {
        let task1 = SimulatorTask::new_with_custom_priority(
            super::task::Task::LTask(TaskProps {
                id: 1,
                wcet_l: 2,
                wcet_h: 2,
                offset: 1,
                period: 5,
            }),
            2,
            2,
        );
        let task2 = SimulatorTask::new_with_custom_priority(
            super::task::Task::LTask(TaskProps {
                id: 2,
                wcet_l: 2,
                wcet_h: 2,
                offset: 0,
                period: 5,
            }),
            3,
            2,
        );
        let task3 = SimulatorTask::new_with_custom_priority(
            super::task::Task::LTask(TaskProps {
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
            agent: None,
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

        assert_events_eq(events, vec![]);
    }

    #[test]
    fn different_criticality() {
        let task1 = SimulatorTask::new_with_custom_priority(
            super::task::Task::HTask(TaskProps {
                id: 1,
                wcet_l: 1,
                wcet_h: 1,
                offset: 1,
                period: 3,
            }),
            1,
            1,
        );
        let task2 = SimulatorTask::new_with_custom_priority(
            super::task::Task::LTask(TaskProps {
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
            agent: None,
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

        assert_events_eq(events, vec![]);
    }

    #[test]
    fn different_criticality_task_kill() {
        let task1 = SimulatorTask::new_with_custom_priority(
            super::task::Task::LTask(TaskProps {
                id: 1,
                wcet_l: 2,
                wcet_h: 0,
                offset: 0,
                period: 5,
            }),
            1,
            3,
        );
        let task2 = SimulatorTask::new_with_custom_priority(
            super::task::Task::HTask(TaskProps {
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
            tasks: vec![task1.clone(), task2.clone()],
            random_execution_time: false,
            agent: None,
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

        assert_events_eq(
            events,
            vec![
                SimulatorEvent::TaskKill(task1.task.props().id, 1),
                SimulatorEvent::TaskKill(task1.task.props().id, 6),
                SimulatorEvent::TaskKill(task1.task.props().id, 11),
            ],
        );
    }

    #[test]
    fn different_criticality_mode_change() {
        let task1 = SimulatorTask::new_with_custom_priority(
            super::task::Task::HTask(TaskProps {
                id: 1,
                wcet_l: 2,
                wcet_h: 3,
                offset: 0,
                period: 5,
            }),
            1,
            3,
        );
        let task2 = SimulatorTask::new_with_custom_priority(
            super::task::Task::LTask(TaskProps {
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
            agent: None,
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

        assert_events_eq(
            events,
            vec![
                SimulatorEvent::ModeChange(crate::simulator::SimulatorMode::HMode, 1),
                SimulatorEvent::ModeChange(crate::simulator::SimulatorMode::LMode, 3),
                SimulatorEvent::ModeChange(crate::simulator::SimulatorMode::HMode, 6),
                SimulatorEvent::ModeChange(crate::simulator::SimulatorMode::LMode, 8),
                SimulatorEvent::ModeChange(crate::simulator::SimulatorMode::HMode, 11),
            ],
        );
    }

    #[test]
    fn non_feasible_simple() {
        let task1 = SimulatorTask::new_with_custom_priority(
            super::task::Task::LTask(TaskProps {
                id: 1,
                wcet_l: 1,
                wcet_h: 1,
                offset: 1,
                period: 3,
            }),
            1,
            1,
        );
        let task2 = SimulatorTask::new_with_custom_priority(
            super::task::Task::LTask(TaskProps {
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
            agent: None,
        };
        let (tasks, events) = simulator.run(10);

        assert_eq!(tasks, None);
        assert_events_eq(events, vec![]);
    }

    #[test]
    fn non_feasible_mode_change() {
        let task1 = SimulatorTask::new_with_custom_priority(
            super::task::Task::HTask(TaskProps {
                id: 1,
                wcet_l: 2,
                wcet_h: 3,
                offset: 0,
                period: 4,
            }),
            1,
            3,
        );
        let task2 = SimulatorTask::new_with_custom_priority(
            super::task::Task::HTask(TaskProps {
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
            agent: None,
        };
        let (tasks, events) = simulator.run(10);

        assert_eq!(tasks, None);
        assert_events_eq(
            events,
            vec![SimulatorEvent::ModeChange(
                crate::simulator::SimulatorMode::HMode,
                1,
            )],
        );
    }

    #[test]
    fn non_feasible_exceed_wcet_h() {
        let task1 = SimulatorTask::new_with_custom_priority(
            super::task::Task::HTask(TaskProps {
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
            agent: None,
        };
        let (tasks, events) = simulator.run(10);

        assert_eq!(tasks, None);
        assert_events_eq(
            events,
            vec![SimulatorEvent::ModeChange(
                crate::simulator::SimulatorMode::HMode,
                1,
            )],
        );
    }
}

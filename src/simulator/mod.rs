use memory_stats::memory_stats;
use probability::source;

use self::task::{SimulatorTask, Task, TaskId, TimeUnit};
use crate::agent::SimulatorAgent;
use std::{
    cell::RefCell,
    collections::{BinaryHeap, HashMap},
    rc::Rc,
    time,
};

pub mod task;
pub mod validation;

#[derive(Debug, Clone)]
struct SimulatorJob {
    task: Rc<RefCell<SimulatorTask>>,
    exec_time: TimeUnit,
    run_time: TimeUnit,
    event: Rc<RefCell<SimulatorEvent>>,
}

#[derive(Debug, PartialEq, Copy, Clone)]
pub enum SimulatorMode {
    LMode,
    HMode,
}

#[derive(Debug, Clone)]
pub enum SimulatorEvent {
    Start(Rc<RefCell<SimulatorTask>>, TimeUnit),
    End(Rc<RefCell<SimulatorTask>>, TimeUnit),
    TaskKill(Rc<RefCell<SimulatorTask>>, TimeUnit),
    ModeChange(SimulatorMode, TimeUnit),
}

impl SimulatorEvent {
    pub fn priority(&self, simulator: &Simulator) -> Option<TimeUnit> {
        match self {
            SimulatorEvent::Start(task, time)
            | SimulatorEvent::End(task, time)
            | SimulatorEvent::TaskKill(task, time) => {
                let task = task.borrow();
                Some(
                    time * simulator.tasks.len() as TimeUnit + simulator.tasks.len() as TimeUnit
                        - task.task.props().id,
                )
            }
            _ => None,
        }
    }
}

pub struct Simulator {
    pub tasks: Vec<Rc<RefCell<SimulatorTask>>>,
    random_execution_time: bool,
    agent: Option<Rc<RefCell<SimulatorAgent>>>,
    elapsed_times: Vec<time::Duration>,
    memory_usage: Vec<(usize, usize)>,

    // Needed during simulation.
    // Inited during constructor; should not reuse the same simulator for multiple simulations.
    random_source: source::Xorshift128Plus, // TODO: make seed configurable
    jobs: HashMap<TaskId, SimulatorJob>,    // max 1 job per task
    running_job: Option<Rc<RefCell<SimulatorJob>>>,
    ready_jobs_queue: BinaryHeap<Rc<RefCell<SimulatorJob>>>, // except the one that is currently running
    event_queue: BinaryHeap<Rc<RefCell<SimulatorEvent>>>,    // only start and end events
    event_history: Vec<Rc<RefCell<SimulatorEvent>>>, // all events, used if RETURN_FULL_HISTORY is set
}

impl Simulator {
    pub fn new(
        tasks: Vec<SimulatorTask>,
        random_execution_time: bool,
        agent: Option<Rc<RefCell<SimulatorAgent>>>,
    ) -> Self {
        let tasks = tasks
            .into_iter()
            .map(|task| {
                Rc::new(RefCell::new({
                    // In the simulator, the task id is used to determine the order of execution (priority).
                    let mut task = task;
                    task.task.props_mut().id = task.task.props().id + task.task.props().period;
                    task
                }))
            })
            .collect::<Vec<_>>();
        Self {
            tasks,
            random_execution_time,
            agent: None,
            elapsed_times: vec![],
            memory_usage: vec![],
            random_source: source::default(42),
            jobs: HashMap::new(),
            running_job: None,
            ready_jobs_queue: BinaryHeap::new(),
            event_queue: BinaryHeap::new(),
            event_history: vec![],
        }
    }

    fn init_event_queue(&mut self) {
        for task in &self.tasks {
            // Generate the first arrival event.
            let event = Rc::new(RefCell::new(SimulatorEvent::Start(
                task.clone(),
                task.borrow().task.props().offset,
            )));

            // Create a job for the task.
            let job = Rc::new(RefCell::new(SimulatorJob {
                task: task.clone(),
                exec_time: 0,
                run_time: 0,
                event,
            }));

            // Push the job to the ready queue.
            self.ready_jobs_queue.push(job.clone());
        }
    }

    pub fn fire<const RETURN_FULL_HISTORY: bool>(
        &mut self,
        duration: TimeUnit,
    ) -> (Option<Vec<Option<TaskId>>>, Vec<SimulatorEvent>) {
        (todo!(), self.event_history)
    }
}

#[cfg(test)]
mod tests {
    use std::{cell::RefCell, rc::Rc};

    use crate::simulator::SimulatorEvent;

    use super::{task::TaskProps, Simulator, SimulatorTask};

    fn assert_events_eq(events: Vec<SimulatorEvent>, expected: Vec<SimulatorEvent>) {
        let events_with_stripped_start_end = events
            .iter()
            .filter(|e| !matches!(e, SimulatorEvent::Start(_, _) | SimulatorEvent::End(_, _)))
            .cloned()
            .collect::<Vec<_>>();
        for (event, expected) in events_with_stripped_start_end.iter().zip(expected.iter()) {
            match (event, expected) {
                (
                    SimulatorEvent::TaskKill(task, time),
                    SimulatorEvent::TaskKill(expected_task, expected_time),
                ) => {
                    assert_eq!(
                        task.borrow().task.props().id,
                        expected_task.borrow().task.props().id
                    );
                    assert_eq!(time, expected_time);
                }
                (
                    SimulatorEvent::ModeChange(mode, time),
                    SimulatorEvent::ModeChange(expected_mode, expected_time),
                ) => {
                    assert_eq!(mode, expected_mode);
                    assert_eq!(time, expected_time);
                }
                _ => panic!("Events do not match"),
            }
        }
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

        let mut simulator = Simulator::new(vec![task1, task2], false, None);
        let (tasks, events) = simulator.fire::<true>(10);

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

        let mut simulator = Simulator::new(vec![task1, task2, task3], false, None);
        let (tasks, events) = simulator.fire::<true>(10);

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

        let mut simulator = Simulator::new(vec![task1, task2], false, None);
        let (tasks, events) = simulator.fire::<true>(8);

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

        let mut simulator = Simulator::new(vec![task1.clone(), task2.clone()], false, None);
        let (tasks, events) = simulator.fire::<true>(12);

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
                SimulatorEvent::TaskKill(Rc::new(RefCell::new(task1)), 1),
                SimulatorEvent::TaskKill(Rc::new(RefCell::new(task1)), 6),
                SimulatorEvent::TaskKill(Rc::new(RefCell::new(task1)), 11),
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

        let mut simulator = Simulator::new(vec![task1, task2], false, None);
        let (tasks, events) = simulator.fire::<true>(12);

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

        let mut simulator = Simulator::new(vec![task1, task2], false, None);
        let (tasks, events) = simulator.fire::<true>(10);

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

        let mut simulator = Simulator::new(vec![task1, task2], false, None);
        let (tasks, events) = simulator.fire::<true>(10);

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
        let mut simulator = Simulator::new(vec![task1], false, None);
        let (tasks, events) = simulator.fire::<true>(10);

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

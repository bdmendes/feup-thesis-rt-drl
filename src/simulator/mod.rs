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

pub mod handlers;
pub mod task;
pub mod validation;

#[derive(Debug, PartialEq, Copy, Clone)]
pub enum SimulatorJobState {
    Ready,
    Running,
}

#[derive(Debug, Clone)]
struct SimulatorJob {
    task: Rc<RefCell<SimulatorTask>>,
    exec_time: TimeUnit,
    run_time: TimeUnit,
    event: Rc<RefCell<SimulatorEvent>>,
    state: SimulatorJobState,
    is_agent: bool,
}

impl PartialEq for SimulatorJob {
    fn eq(&self, other: &Self) -> bool {
        self.task.borrow().task.props().id == other.task.borrow().task.props().id
    }
}

impl Eq for SimulatorJob {}

impl Ord for SimulatorJob {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.task
            .borrow()
            .task
            .props()
            .id
            .cmp(&other.task.borrow().task.props().id)
    }
}

impl PartialOrd for SimulatorJob {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
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
    pub fn task(&self) -> Rc<RefCell<SimulatorTask>> {
        match self {
            SimulatorEvent::Start(task, _) | SimulatorEvent::End(task, _) => task.clone(),
            _ => unimplemented!("should not be called"),
        }
    }
}

impl PartialEq for SimulatorEvent {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (SimulatorEvent::Start(task1, time1), SimulatorEvent::Start(task2, time2))
            | (SimulatorEvent::End(task1, time1), SimulatorEvent::End(task2, time2))
            | (SimulatorEvent::TaskKill(task1, time1), SimulatorEvent::TaskKill(task2, time2)) => {
                task1.borrow().task.props().id == task2.borrow().task.props().id && time1 == time2
            }
            _ => false,
        }
    }
}

impl Eq for SimulatorEvent {}

impl Ord for SimulatorEvent {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match (self, other) {
            (SimulatorEvent::Start(_, time1), SimulatorEvent::End(_, time2)) => {
                if time1 < time2 {
                    std::cmp::Ordering::Less
                } else {
                    // Even if times are equal, we want to prioritize the end event.
                    std::cmp::Ordering::Greater
                }
            }
            (SimulatorEvent::End(_, time1), SimulatorEvent::Start(_, time2)) => {
                if time1 > time2 {
                    std::cmp::Ordering::Greater
                } else {
                    // Same as above.
                    std::cmp::Ordering::Less
                }
            }
            (SimulatorEvent::Start(task1, time1), SimulatorEvent::Start(task2, time2))
            | (SimulatorEvent::End(task1, time1), SimulatorEvent::End(task2, time2)) => {
                if time1 < time2 {
                    std::cmp::Ordering::Less
                } else if time1 > time2 {
                    std::cmp::Ordering::Greater
                } else {
                    task1
                        .borrow()
                        .task
                        .props()
                        .id
                        .cmp(&task2.borrow().task.props().id)
                }
            }
            _ => std::cmp::Ordering::Equal,
        }
    }
}

impl PartialOrd for SimulatorEvent {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl SimulatorEvent {
    pub fn priority(&self, simulator: &Simulator) -> Option<TimeUnit> {
        match self {
            SimulatorEvent::Start(task, time)
            | SimulatorEvent::End(task, time)
            | SimulatorEvent::TaskKill(task, time) => {
                let task = task.borrow();
                if let Some(custom_priority) = task.custom_priority {
                    // The priority is based on the custom priority.
                    Some(
                        time * simulator.tasks.len() as TimeUnit
                            + simulator.tasks.len() as TimeUnit
                            - custom_priority,
                    )
                } else {
                    // The priority is based on the task id.
                    Some(
                        time * simulator.tasks.len() as TimeUnit
                            + simulator.tasks.len() as TimeUnit
                            - task.task.props().id,
                    )
                }
            }
            _ => None,
        }
    }

    pub fn time(&self) -> TimeUnit {
        match self {
            SimulatorEvent::Start(_, time)
            | SimulatorEvent::End(_, time)
            | SimulatorEvent::TaskKill(_, time)
            | SimulatorEvent::ModeChange(_, time) => *time,
        }
    }

    pub fn handle(&self, simulator: &mut Simulator) {
        match self {
            SimulatorEvent::Start(task, time) => {
                handlers::handle_start_event(task.clone(), *time, simulator);
            }
            SimulatorEvent::End(task, time) => {
                handlers::handle_end_event(task.clone(), *time, simulator);
            }
            _ => unimplemented!("should not be called"),
        }
    }
}

pub struct Simulator {
    pub tasks: Vec<Rc<RefCell<SimulatorTask>>>,
    random_execution_time: bool,
    agent: Option<Rc<RefCell<SimulatorAgent>>>,
    _elapsed_times: Vec<time::Duration>,
    _memory_usage: Vec<(usize, usize)>,

    // Needed during simulation.
    // Inited during constructor; should not reuse the same simulator for multiple simulations.
    random_source: source::Xorshift128Plus, // TODO: make seed configurable
    jobs: HashMap<TaskId, Rc<RefCell<SimulatorJob>>>, // max 1 job per task
    running_job: Option<Rc<RefCell<SimulatorJob>>>,
    ready_jobs_queue: BinaryHeap<Rc<RefCell<SimulatorJob>>>, // except the one that is currently running
    event_queue: BinaryHeap<Rc<RefCell<SimulatorEvent>>>,    // only start and end events
    event_history: Vec<Rc<RefCell<SimulatorEvent>>>, // all events, used if RETURN_FULL_HISTORY is set
    last_context_switch: TimeUnit,
    now: TimeUnit,
    mode: SimulatorMode,
}

impl Simulator {
    pub fn new(
        tasks: Vec<SimulatorTask>,
        random_execution_time: bool,
        agent: Option<Rc<RefCell<SimulatorAgent>>>,
    ) -> Self {
        let mut tasks = tasks.clone();
        let tasks_size = tasks.len();
        for task in &mut tasks {
            if let Some(custom_priority) = task.custom_priority {
                // The priority is based on the custom priority.
                task.task.props_mut().id = custom_priority;
            } else {
                // Default to rate monotonic priority.
                task.task.props_mut().id =
                    task.task.props().id + task.task.props().period * tasks_size as TaskId;
            }
        }

        Self {
            tasks: tasks
                .iter()
                .map(|t| Rc::new(RefCell::new(t.clone())).clone())
                .collect(),
            random_execution_time,
            agent,
            _elapsed_times: vec![],
            _memory_usage: vec![],
            random_source: source::default(42),
            jobs: HashMap::new(),
            running_job: None,
            ready_jobs_queue: BinaryHeap::new(),
            event_queue: BinaryHeap::new(),
            event_history: vec![],
            last_context_switch: 0,
            now: 0,
            mode: SimulatorMode::LMode,
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
                state: SimulatorJobState::Ready,
                is_agent: false,
            }));

            // Push the job to the ready queue.
            self.ready_jobs_queue.push(job.clone());
        }

        // TODO: Create job for the H-mode agent.
        // Should it be a task too?
    }

    pub fn push_event(&mut self, event: Rc<RefCell<SimulatorEvent>>) {
        self.event_history.push(event.clone());
        if self.agent.is_some() {
            let event_cpy = match &*event.borrow() {
                SimulatorEvent::Start(task, time) => SimulatorEvent::Start(task.clone(), *time),
                SimulatorEvent::End(task, time) => SimulatorEvent::End(task.clone(), *time),
                SimulatorEvent::TaskKill(task, time) => {
                    SimulatorEvent::TaskKill(task.clone(), *time)
                }
                SimulatorEvent::ModeChange(mode, time) => SimulatorEvent::ModeChange(*mode, *time),
            };
            self.agent
                .as_ref()
                .unwrap()
                .borrow_mut()
                .push_event(event_cpy);
        }
    }

    pub fn fire<const RETURN_FULL_HISTORY: bool>(
        &mut self,
        duration: TimeUnit,
    ) -> (Option<Vec<Option<TaskId>>>, Vec<SimulatorEvent>) {
        self.init_event_queue();

        while self.now < duration {
            let event = self.event_queue.pop().unwrap();
            self.now = event.borrow().time();
            event.borrow().handle(self);
        }

        (
            None,
            self.event_history
                .iter()
                .map(|e| e.borrow().clone())
                .collect(),
        )
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
                SimulatorEvent::TaskKill(Rc::new(RefCell::new(task1.clone())), 1),
                SimulatorEvent::TaskKill(Rc::new(RefCell::new(task1.clone())), 6),
                SimulatorEvent::TaskKill(Rc::new(RefCell::new(task1.clone())), 11),
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

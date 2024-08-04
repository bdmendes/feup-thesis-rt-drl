use probability::source;
use task::TaskProps;

use self::task::{SimulatorTask, TaskId, TimeUnit};
use crate::{agent::SimulatorAgent, generator::Runnable};
use std::{
    cell::RefCell,
    collections::{BinaryHeap, HashMap},
    rc::Rc,
    time,
};

pub mod handlers;
pub mod task;
pub mod validation;

const MAX_TASKS_SIZE: usize = 1000;

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

impl SimulatorJob {
    // As the id is changed to accomodate priorities, we need to return the real id for debugging purposes.
    pub fn real_id(&self) -> TaskId {
        let task = self.task.borrow();
        if let Some(custom_priority) = task.custom_priority {
            // The priority is based on the custom priority.
            task.task.props().id
                - custom_priority * MAX_TASKS_SIZE as TaskId
                - MAX_TASKS_SIZE as TaskId
        } else {
            // Default to rate monotonic priority.
            task.task.props().id
                - task.task.props().period * MAX_TASKS_SIZE as TaskId
                - MAX_TASKS_SIZE as TaskId
        }
    }
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
            .reverse()
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

#[derive(Debug, PartialEq, Copy, Clone)]
pub enum EndReason {
    JobCompletion,
    BudgetExceedance,
}

#[derive(Debug, Clone)]
pub enum SimulatorEvent {
    Start(Rc<RefCell<SimulatorTask>>, TimeUnit),
    End(Rc<RefCell<SimulatorTask>>, TimeUnit, EndReason),
    TaskKill(Rc<RefCell<SimulatorTask>>, TimeUnit),
    ModeChange(SimulatorMode, TimeUnit),
}

impl SimulatorEvent {
    pub fn task(&self) -> Rc<RefCell<SimulatorTask>> {
        match self {
            SimulatorEvent::Start(task, _) | SimulatorEvent::End(task, _, _) => task.clone(),
            _ => unimplemented!("should not be called"),
        }
    }
}

impl PartialEq for SimulatorEvent {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (SimulatorEvent::Start(task1, time1), SimulatorEvent::Start(task2, time2))
            | (SimulatorEvent::End(task1, time1, _), SimulatorEvent::End(task2, time2, _))
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
            (SimulatorEvent::Start(_, time1), SimulatorEvent::End(_, time2, _)) => {
                if time1 < time2 {
                    std::cmp::Ordering::Greater
                } else {
                    // Even if times are equal, we want to prioritize the end event.
                    std::cmp::Ordering::Less
                }
            }
            (SimulatorEvent::End(_, time1, _), SimulatorEvent::Start(_, time2)) => {
                if time1 > time2 {
                    std::cmp::Ordering::Less
                } else {
                    // Same as above.
                    std::cmp::Ordering::Greater
                }
            }
            (SimulatorEvent::Start(task1, time1), SimulatorEvent::Start(task2, time2))
            | (SimulatorEvent::End(task1, time1, _), SimulatorEvent::End(task2, time2, _)) =>
            {
                #[allow(clippy::comparison_chain)]
                if time1 < time2 {
                    std::cmp::Ordering::Greater
                } else if time1 > time2 {
                    std::cmp::Ordering::Less
                } else {
                    task1
                        .borrow()
                        .task
                        .props()
                        .id
                        .cmp(&task2.borrow().task.props().id)
                        .reverse()
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
            | SimulatorEvent::End(task, time, _)
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
            | SimulatorEvent::End(_, time, _)
            | SimulatorEvent::TaskKill(_, time)
            | SimulatorEvent::ModeChange(_, time) => *time,
        }
    }

    pub fn handle(&self, simulator: &mut Simulator) {
        match self {
            SimulatorEvent::Start(task, time) => {
                handlers::handle_start_event(task.clone(), *time, simulator);
            }
            SimulatorEvent::End(task, time, reason) => {
                handlers::handle_end_event(task.clone(), *time, *reason, simulator);
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
    event_history: Vec<Rc<RefCell<SimulatorEvent>>>,         // all events
    last_context_switch: TimeUnit,
    now: TimeUnit,
    mode: SimulatorMode,
    running_history: Vec<Option<TaskId>>, // used if we want to return the full history
}

impl Simulator {
    pub fn new(
        tasks: Vec<SimulatorTask>,
        random_execution_time: bool,
        agent: Option<Rc<RefCell<SimulatorAgent>>>,
    ) -> Self {
        let mut tasks = tasks.clone();
        for task in &mut tasks {
            if let Some(custom_priority) = task.custom_priority {
                // The priority is based on the custom priority.
                task.task.props_mut().id = custom_priority * MAX_TASKS_SIZE as TaskId
                    + MAX_TASKS_SIZE as TaskId
                    + task.task.props().id;
            } else {
                // Default to rate monotonic priority.
                task.task.props_mut().id = task.task.props().id
                    + task.task.props().period * MAX_TASKS_SIZE as TaskId
                    + MAX_TASKS_SIZE as TaskId;
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
            running_history: vec![],
        }
    }

    fn init_event_queue(&mut self) {
        for task in &self.tasks {
            // Generate the first arrival event.
            let event = Rc::new(RefCell::new(SimulatorEvent::Start(
                task.clone(),
                task.borrow().task.props().offset,
            )));
            self.event_queue.push(event.clone());

            // Create a job for the task.
            let job = Rc::new(RefCell::new(SimulatorJob {
                task: task.clone(),
                exec_time: 0,
                run_time: 0,
                event,
                state: SimulatorJobState::Ready,
                is_agent: false,
            }));

            // Add the job to the jobs map.
            self.jobs.insert(task.borrow().task.props().id, job);
        }

        if self.agent.is_some() {
            // Create a task for the agent.
            let task = Rc::new(RefCell::new(SimulatorTask::new_with_custom_priority(
                task::Task::HTask(TaskProps {
                    id: self.tasks.len() as TaskId + 1,
                    wcet_l: 0,
                    wcet_h: Runnable::duration_to_time_unit(time::Duration::from_micros(1)),
                    offset: 0,
                    period: Runnable::duration_to_time_unit(time::Duration::from_micros(10)),
                }),
                self.tasks.len() as TaskId + 1,
                self.tasks.len() as TaskId + 1,
            )));
            self.tasks.push(task.clone());

            // Create an arrival event for the agent.
            let event = Rc::new(RefCell::new(SimulatorEvent::Start(task.clone(), 0)));
            self.event_queue.push(event.clone());

            // Create a job for the agent.
            let job = Rc::new(RefCell::new(SimulatorJob {
                task: task.clone(),
                exec_time: 0,
                run_time: 0,
                event,
                state: SimulatorJobState::Ready,
                is_agent: true,
            }));

            // Add the job to the jobs map.
            self.jobs.insert(task.borrow().task.props().id, job);
        }
    }

    pub fn push_event(&mut self, event: Rc<RefCell<SimulatorEvent>>) {
        self.event_history.push(event.clone());
        if self.agent.is_some() {
            let event_cpy = match &*event.borrow() {
                SimulatorEvent::Start(task, time) => SimulatorEvent::Start(task.clone(), *time),
                SimulatorEvent::End(task, time, reason) => {
                    SimulatorEvent::End(task.clone(), *time, *reason)
                }
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

    fn change_back_task_ids(&mut self) {
        // Change the task ids back to their original values.
        for task in &self.tasks {
            let real_id = self
                .jobs
                .get(&task.borrow().task.props().id)
                .unwrap()
                .borrow()
                .real_id();
            task.borrow_mut().task.props_mut().id = real_id;
        }
    }

    pub fn fire<const RETURN_FULL_HISTORY: bool>(
        &mut self,
        duration: TimeUnit,
    ) -> (Vec<Option<TaskId>>, Vec<SimulatorEvent>) {
        self.init_event_queue();

        while self.now < duration {
            println!("------------------");
            println!(
                "instant: {}; events in queue: {}; ready jobs queue: {:?}\n",
                self.event_queue.peek().unwrap().borrow().time(),
                self.event_queue.len(),
                self.ready_jobs_queue
                    .iter()
                    .map(|j| j.borrow().real_id())
                    .collect::<Vec<_>>()
            );
            for event in &self.event_queue {
                println!(
                    "event: {:?} at instant: {}",
                    event.borrow(),
                    event.borrow().time()
                );
            }

            let event = self.event_queue.pop().unwrap();
            println!("\nPopped event: {:?}", event.borrow());

            if RETURN_FULL_HISTORY {
                for _ in self.now..(event.borrow().time()) {
                    self.running_history
                        .push(self.running_job.as_ref().map(|job| job.borrow().real_id()));
                }
            }

            self.now = event.borrow().time();
            event.borrow().handle(self);
        }

        self.change_back_task_ids();

        (
            self.running_history.clone(),
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
            .filter(|e| {
                !matches!(
                    e,
                    SimulatorEvent::Start(_, _) | SimulatorEvent::End(_, _, _)
                )
            })
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

    fn same_criticality_1() {
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
            tasks,
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
            tasks,
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
    fn different_criticality_1() {
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
            tasks,
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
            tasks,
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
                SimulatorEvent::TaskKill(Rc::new(RefCell::new(task1.clone())), 2),
                SimulatorEvent::TaskKill(Rc::new(RefCell::new(task1.clone())), 7),
                SimulatorEvent::TaskKill(Rc::new(RefCell::new(task1.clone())), 12),
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
            tasks,
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
                SimulatorEvent::ModeChange(crate::simulator::SimulatorMode::HMode, 2),
                SimulatorEvent::ModeChange(crate::simulator::SimulatorMode::LMode, 2),
                SimulatorEvent::ModeChange(crate::simulator::SimulatorMode::HMode, 7),
                SimulatorEvent::ModeChange(crate::simulator::SimulatorMode::LMode, 7),
                SimulatorEvent::ModeChange(crate::simulator::SimulatorMode::HMode, 12),
                SimulatorEvent::ModeChange(crate::simulator::SimulatorMode::LMode, 12),
            ],
        );
    }
}

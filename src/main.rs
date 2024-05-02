use std::{cell::RefCell, rc::Rc};

use agent::{
    dqn::ActivationFunction, SimulatorAgent, DEFAULT_GAMMA, DEFAULT_LEARNING_RATE,
    DEFAULT_MEM_SIZE, DEFAULT_MIN_MEM_SIZE, DEFAULT_UPDATE_FREQ,
};
use simulator::{
    task::{Task, TaskProps},
    Simulator,
};

use crate::{
    agent::DEFAULT_SAMPLE_BATCH_SIZE,
    simulator::{validation::feasible_schedule_design_time, SimulatorTask},
};

pub mod agent;
pub mod ml;
pub mod simulator;

fn main() {
    // This task will overrun its WCET_L in L mode.
    let task0 = SimulatorTask::new(
        Task::HTask(TaskProps {
            id: 0,
            wcet_l: 2,
            wcet_h: 3,
            period: 6,
            offset: 2,
        }),
        0,
        3,
    );

    // This task will behave as expected.
    let task1 = SimulatorTask::new(
        Task::LTask(TaskProps {
            id: 1,
            wcet_l: 2,
            wcet_h: 0,
            period: 6,
            offset: 0,
        }),
        2,
        2,
    );

    let tasks = vec![task0, task1];

    assert!(feasible_schedule_design_time(&tasks));

    let mut sim = Simulator {
        tasks: tasks.clone(),
        random_execution_time: false,
        agent: Some(Rc::new(RefCell::new(SimulatorAgent::new(
            DEFAULT_MEM_SIZE,
            DEFAULT_MIN_MEM_SIZE,
            DEFAULT_GAMMA,
            DEFAULT_UPDATE_FREQ,
            DEFAULT_LEARNING_RATE,
            16,
            DEFAULT_SAMPLE_BATCH_SIZE,
            ActivationFunction::ReLU,
            SimulatorAgent::number_of_actions(&tasks),
            SimulatorAgent::number_of_features(&tasks),
        )))),
    };

    let _results = sim.run(4100);
    //println!("Run history: {:?}", results);
}

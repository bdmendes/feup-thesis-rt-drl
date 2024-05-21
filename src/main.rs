use std::{cell::RefCell, rc::Rc};

use agent::{
    dqn::ActivationFunction, SimulatorAgent, DEFAULT_GAMMA, DEFAULT_HIDDEN_SIZE,
    DEFAULT_LEARNING_RATE, DEFAULT_MEM_SIZE, DEFAULT_MIN_MEM_SIZE, DEFAULT_SAMPLE_BATCH_SIZE,
    DEFAULT_UPDATE_FREQ,
};
use generator::{
    generate_tasks, AVG_AVG_RUNNABLES_PER_TASK, AVG_MAX_RUNNABLES_PER_TASK, AVG_TOTAL_RUNNABLES,
};
use simulator::Simulator;

use crate::generator::AVG_MIN_RUNNABLES_PER_TASK;

pub mod agent;
pub mod generator;
pub mod ml;
pub mod simulator;

fn main() {
    let (tasks, _) = generate_tasks(
        0.8,
        AVG_MIN_RUNNABLES_PER_TASK,
        AVG_AVG_RUNNABLES_PER_TASK,
        AVG_MAX_RUNNABLES_PER_TASK,
        AVG_TOTAL_RUNNABLES / AVG_MAX_RUNNABLES_PER_TASK,
    );

    for task in &tasks {
        println!("{:?}", task);
    }

    let number_of_actions = SimulatorAgent::number_of_actions(&tasks);
    let number_of_features = SimulatorAgent::number_of_features(&tasks);
    let mut simulator = Simulator {
        tasks,
        random_execution_time: true,
        agent: Some(Rc::new(RefCell::new(agent::SimulatorAgent::new(
            DEFAULT_MEM_SIZE,
            DEFAULT_MIN_MEM_SIZE,
            DEFAULT_GAMMA,
            DEFAULT_UPDATE_FREQ,
            DEFAULT_LEARNING_RATE,
            DEFAULT_HIDDEN_SIZE,
            DEFAULT_SAMPLE_BATCH_SIZE,
            ActivationFunction::ReLU,
            number_of_actions,
            number_of_features,
        )))),
    };

    simulator.run(500_000_000_000);
}

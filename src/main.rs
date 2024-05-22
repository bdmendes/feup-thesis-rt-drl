use crate::simulator::{validation::feasible_schedule_design_time, SimulatorMode};
use agent::{
    dqn::ActivationFunction, SimulatorAgent, DEFAULT_GAMMA, DEFAULT_HIDDEN_SIZE,
    DEFAULT_LEARNING_RATE, DEFAULT_MEM_SIZE, DEFAULT_MIN_MEM_SIZE, DEFAULT_SAMPLE_BATCH_SIZE,
    DEFAULT_UPDATE_FREQ,
};
use generator::generate_tasks;
use simulator::Simulator;
use std::{cell::RefCell, io::Write, rc::Rc};

pub mod agent;
pub mod generator;
pub mod ml;
pub mod simulator;

fn main() {
    let instants: u64 = 300_000_000;
    let tasks = generate_tasks(0.2, 20, 10);
    let tasks_cpy = tasks.clone();
    let tasks_cpy_2 = tasks.clone();

    for task in &tasks {
        println!("{:?}", task);
    }

    assert!(feasible_schedule_design_time(&tasks));

    let agent = Rc::new(RefCell::new(SimulatorAgent::new(
        DEFAULT_MEM_SIZE,
        DEFAULT_MIN_MEM_SIZE,
        DEFAULT_GAMMA,
        DEFAULT_UPDATE_FREQ,
        DEFAULT_LEARNING_RATE,
        DEFAULT_HIDDEN_SIZE,
        DEFAULT_SAMPLE_BATCH_SIZE,
        ActivationFunction::Sigmoid,
        SimulatorAgent::number_of_actions(&tasks),
        SimulatorAgent::number_of_features(&tasks),
    )));

    let mut simulator = Simulator {
        tasks,
        random_execution_time: true,
        agent: Some(agent.clone()),
    };

    simulator.run::<false>(instants);

    println!("Cumulative reward: {}", agent.borrow().cumulative_reward());

    // write cum reward to file
    let mut file = std::fs::File::create("out/reward.txt").unwrap();
    file.write_all(agent.borrow().cumulative_reward().to_string().as_bytes())
        .unwrap();

    agent.borrow_mut().quit_training();

    let mut simulator = Simulator {
        tasks: tasks_cpy,
        random_execution_time: true,
        agent: Some(agent.clone()),
    };

    simulator.run::<false>(instants);

    println!(
        "Cumulative reward 2: {}",
        agent.borrow().cumulative_reward()
    );

    // write cum reward to file
    let mut file = std::fs::File::create("out/reward2.txt").unwrap();
    file.write_all(agent.borrow().cumulative_reward().to_string().as_bytes())
        .unwrap();

    agent.borrow_mut().placebo_mode();

    let mut simulator = Simulator {
        tasks: tasks_cpy_2,
        random_execution_time: true,
        agent: Some(agent.clone()),
    };

    simulator.run::<false>(instants);

    println!("Cumulative reward: {}", agent.borrow().cumulative_reward());

    // write cum reward to file
    let mut file = std::fs::File::create("out/reward3.txt").unwrap();
    file.write_all(agent.borrow().cumulative_reward().to_string().as_bytes())
        .unwrap();
}

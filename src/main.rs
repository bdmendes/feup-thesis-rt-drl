use std::{cell::RefCell, rc::Rc};

use agent::{
    dqn::ActivationFunction, SimulatorAgent, DEFAULT_GAMMA, DEFAULT_LEARNING_RATE,
    DEFAULT_MEM_SIZE, DEFAULT_MIN_MEM_SIZE, DEFAULT_UPDATE_FREQ,
};
use compute::distributions::{Distribution, Normal};
use simulator::{
    task::{Task, TaskProps},
    Simulator,
};

use crate::simulator::{validation::feasible_schedule_design_time, SimulatorTask};

pub mod agent;
pub mod ml;
pub mod simulator;

fn main() {
    let average_utilization = 0.1;
    let average_period = 600.0;
    let average_wcet_multiply_low = 1.2;
    let average_wcet_multiply_high = 1.5;
    let number_of_tasks = 50;
    let average_wcet = average_period * average_utilization / number_of_tasks as f64;

    let normal_dist_wcet_l_multiply =
        Normal::new(average_wcet * average_wcet_multiply_low, average_wcet * 0.2);
    let normal_dist_wcet_h_multiply = Normal::new(
        average_wcet * average_wcet_multiply_high,
        average_wcet * average_wcet_multiply_high * 0.2,
    );
    let period_dist = Normal::new(average_period, average_period * 0.2);
    let actual_execution_time_dist = Normal::new(average_wcet, average_wcet * 0.2);

    let tasks = (0..number_of_tasks)
        .map(|i| {
            let actual_execution_time = (actual_execution_time_dist.sample() as u32).max(1);
            let wcet_l = actual_execution_time.max(normal_dist_wcet_l_multiply.sample() as u32);
            let wcet_h = actual_execution_time
                .max(wcet_l)
                .max(normal_dist_wcet_h_multiply.sample() as u32);

            let props = TaskProps {
                id: i,
                wcet_l,
                wcet_h,
                offset: i,
                period: period_dist.sample() as u32,
            };

            let task = if i % 2 == 0 {
                Task::LTask(props)
            } else {
                Task::HTask(props)
            };

            SimulatorTask::new(task, i % 4, actual_execution_time)
        })
        .collect::<Vec<SimulatorTask>>();

    for task in &tasks {
        println!("{:?}", task);
    }

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
            ActivationFunction::ReLU,
            SimulatorAgent::number_of_actions(&tasks),
            SimulatorAgent::number_of_features(&tasks),
        )))),
    };

    let results = sim.run(60);
    println!("Run history: {:?}", results);
}

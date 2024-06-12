use crate::simulator::validation::feasible_schedule_design_time;
use agent::{
    dqn::ActivationFunction, SimulatorAgent, DEFAULT_GAMMA, DEFAULT_LEARNING_RATE,
    DEFAULT_MEM_SIZE, DEFAULT_MIN_MEM_SIZE, DEFAULT_SAMPLE_BATCH_SIZE, DEFAULT_UPDATE_FREQ,
};
use generator::generate_tasks;
use simulator::Simulator;
use std::{cell::RefCell, io::Write, rc::Rc};

pub mod agent;
pub mod generator;
pub mod ml;
pub mod simulator;

fn run_in_3modes() {
    let instants: u64 = 300_000_000; // 3 seconds

    #[allow(unused_assignments)]
    let mut tasks = Vec::new();
    loop {
        tasks = generate_tasks(0.7, 100, generator::TimeSampleDistribution::Pert);
        for task in &tasks {
            println!("{:?} {}", task, task.task.props().utilization());
        }

        if !feasible_schedule_design_time(&tasks) {
            println!("Not feasible");
            continue;
        }

        break;
    }
    let tasks_cpy = tasks.clone();
    let tasks_cpy_2 = tasks.clone();

    let agent = Rc::new(RefCell::new(SimulatorAgent::new(
        DEFAULT_MEM_SIZE,
        DEFAULT_MIN_MEM_SIZE,
        DEFAULT_GAMMA,
        DEFAULT_UPDATE_FREQ,
        DEFAULT_LEARNING_RATE,
        vec![tasks.len() * 2, tasks.len()],
        DEFAULT_SAMPLE_BATCH_SIZE,
        ActivationFunction::Sigmoid,
        SimulatorAgent::number_of_actions(&tasks),
        SimulatorAgent::number_of_features(&tasks),
    )));

    ////////// Training //////////
    {
        let mut simulator = Simulator {
            tasks,
            random_execution_time: true,
            agent: Some(agent.clone()),
            elapsed_times: vec![],
        };
        simulator.run::<false>(instants);
        let contents = format!(
            "Cumulative reward: {}; mode changes to H: {}; mode changes to L: {}; task kills: {}, task starts: {}",
            agent.borrow().cumulative_reward(),
            agent.borrow().mode_changes_to_hmode(),
            agent.borrow().mode_changes_to_lmode(),
            agent.borrow().task_kills(),
            agent.borrow().task_starts()
        );
        let mut file = std::fs::File::create("out/reward.txt").unwrap();
        file.write_all(contents.as_bytes()).unwrap();
    }

    ////////// Reactive //////////
    {
        agent.borrow_mut().quit_training();
        let mut simulator = Simulator {
            tasks: tasks_cpy,
            random_execution_time: true,
            agent: Some(agent.clone()),
            elapsed_times: vec![],
        };
        simulator.run::<false>(instants);
        let contents = format!(
            "Cumulative reward: {}; mode changes to H: {}; mode changes to L: {}; task kills: {}, task starts: {}",
            agent.borrow().cumulative_reward(),
            agent.borrow().mode_changes_to_hmode(),
            agent.borrow().mode_changes_to_lmode(),
            agent.borrow().task_kills(),
            agent.borrow().task_starts()
        );
        let mut file = std::fs::File::create("out/reward2.txt").unwrap();
        file.write_all(contents.as_bytes()).unwrap();
        let mut times_file = std::fs::File::create("out/times.txt").unwrap();
        for time in simulator.elapsed_times.iter() {
            times_file
                .write_all(format!("{}\n", time.as_nanos()).as_bytes())
                .unwrap();
        }
    }

    ////////// Placebo //////////
    {
        agent.borrow_mut().placebo_mode();
        let mut simulator = Simulator {
            tasks: tasks_cpy_2,
            random_execution_time: true,
            agent: Some(agent.clone()),
            elapsed_times: vec![],
        };

        simulator.run::<false>(instants);
        let contents = format!(
            "Cumulative reward: {}; mode changes to H: {}; mode changes to L: {}; task kills: {}, task starts: {}",
            agent.borrow().cumulative_reward(),
            agent.borrow().mode_changes_to_hmode(),
            agent.borrow().mode_changes_to_lmode(),
            agent.borrow().task_kills(),
            agent.borrow().task_starts()
        );
        let mut file = std::fs::File::create("out/reward3.txt").unwrap();
        file.write_all(contents.as_bytes()).unwrap();
    }
}

fn main() {
    run_in_3modes();
}

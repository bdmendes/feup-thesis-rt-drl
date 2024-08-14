use crate::simulator::validation::feasible_schedule_design_time;
use agent::{
    dqn::ActivationFunction, SimulatorAgent, DEFAULT_GAMMA, DEFAULT_LEARNING_RATE,
    DEFAULT_MEM_SIZE, DEFAULT_MIN_MEM_SIZE, DEFAULT_SAMPLE_BATCH_SIZE, DEFAULT_UPDATE_FREQ,
};
use generator::{generate_tasks, Runnable};
use simulator::{task::SimulatorTask, Simulator};
use std::{cell::RefCell, io::Write, rc::Rc, sync::mpsc::channel, thread::sleep, time::Duration};

pub mod agent;
pub mod generator;
pub mod ml;
pub mod simulator;

fn write_result(agent: &SimulatorAgent, file: &mut std::fs::File) {
    let contents = format!(
        "Cumulative reward: {}; mode changes to H: {}; mode changes to L: {}; task kills: {}, task starts: {}\n",
        agent.cumulative_reward(),
        agent.mode_changes_to_hmode(),
        agent.mode_changes_to_lmode(),
        agent.task_kills(),
        agent.task_starts()
    );
    file.write_all(contents.as_bytes()).unwrap();
}

fn tune(tasks: Vec<SimulatorTask>) {
    let train_instants: u64 = Runnable::duration_to_time_unit(Duration::from_secs(
        std::env::var("TRAIN_INSTANTS")
            .expect("TRAIN_INSTANTS not set")
            .parse::<u64>()
            .unwrap(),
    ));
    let test_instants: u64 = Runnable::duration_to_time_unit(Duration::from_secs(
        std::env::var("TEST_INSTANTS")
            .expect("TEST_INSTANTS not set")
            .parse::<u64>()
            .unwrap(),
    ));
    let number_test_simulations = std::env::var("NUMBER_TEST_SIMULATIONS")
        .expect("NUMBER_TEST_SIMULATIONS not set")
        .parse::<u64>()
        .unwrap();
    sleep(Duration::from_secs(5));
    let mut hyper_iteration = 0;

    let pool = threadpool::ThreadPool::new(
        std::env::var("THREAD_POOL_SIZE")
            .expect("THREAD_POOL_SIZE not set")
            .parse::<usize>()
            .unwrap(),
    );
    let (tx, rx) = channel::<()>(); // so that we can wait for all threads to finish

    {
        ////////// Placebo //////////
        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .create(true)
            .open("out/placebo.txt")
            .unwrap();
        file.set_len(0).unwrap();
        file.write_all(
            format!(
                "parameters: NUMBER_TEST_SIMULATIONS: {}; TRAIN_INSTANTS: {}; TEST_INSTANTS: {}\n",
                number_test_simulations,
                train_instants / 100000000,
                test_instants / 100000000
            )
            .as_bytes(),
        )
        .unwrap();

        for _ in 0..number_test_simulations {
            let agent = Rc::new(RefCell::new(SimulatorAgent::new(
                DEFAULT_MEM_SIZE,
                DEFAULT_MIN_MEM_SIZE,
                DEFAULT_GAMMA,
                DEFAULT_UPDATE_FREQ,
                DEFAULT_LEARNING_RATE,
                vec![8],
                DEFAULT_SAMPLE_BATCH_SIZE,
                ActivationFunction::Sigmoid,
                &tasks,
            )));
            agent.borrow_mut().placebo_mode();
            let mut simulator = Simulator::new(tasks.clone(), true, Some(agent.clone()));
            simulator.fire::<false>(test_instants);
            write_result(&agent.borrow(), &mut file);
        }
    }

    // Hyperparameter tuning: train and test
    for sample_batch_size in [
        DEFAULT_SAMPLE_BATCH_SIZE,
        DEFAULT_SAMPLE_BATCH_SIZE / 2,
        DEFAULT_SAMPLE_BATCH_SIZE * 2,
    ] {
        for hidden_sizes in [
            vec![tasks.len() / 2],
            vec![tasks.len(), tasks.len() / 2],
            vec![tasks.len(), tasks.len() / 2, tasks.len() / 4],
        ] {
            for activation_function in [
                ActivationFunction::Sigmoid,
                ActivationFunction::ReLU,
                ActivationFunction::Tanh,
            ] {
                hyper_iteration += 1;
                let tasks = tasks.clone();
                let hidden_sizes = hidden_sizes.clone();
                let tx = tx.clone();

                pool.execute(move || {
                    let agent = Rc::new(RefCell::new(SimulatorAgent::new(
                        DEFAULT_MEM_SIZE,
                        DEFAULT_MIN_MEM_SIZE,
                        DEFAULT_GAMMA,
                        DEFAULT_UPDATE_FREQ,
                        DEFAULT_LEARNING_RATE,
                        hidden_sizes.clone(),
                        sample_batch_size,
                        activation_function,
                        &tasks,
                    )));

                    ////////// Training //////////
                    {
                        let mut simulator =
                            Simulator::new(tasks.clone(), true, Some(agent.clone()));
                        simulator.fire::<false>(train_instants);
                    }

                    ////////// Testing //////////
                    {
                        let mut file = std::fs::OpenOptions::new()
                        .append(true)
                        .create(true)
                        .open(format!("out/test_{hyper_iteration}.txt"))
                        .unwrap();
                    file.set_len(0).unwrap();
                    file.write_all(format!("hidden sizes: {:?}; sample batch size: {}; activation function: {:?}\n", hidden_sizes, sample_batch_size, activation_function).as_bytes()).unwrap();

                    for _ in 0..number_test_simulations {
                        agent.borrow_mut().quit_training();
                        let mut simulator = Simulator::new(tasks.clone(), true, Some(agent.clone()));
                        simulator.fire::<false>(test_instants);
                        write_result(&agent.borrow(), &mut file);
                    }

                    tx.send(()).unwrap();
                }});
            }
        }
    }

    for _ in 0..27 {
        rx.recv().unwrap();
    }
}

pub fn hp_tuning(number_runnables: usize) {
    std::fs::create_dir_all("out").unwrap();
    loop {
        let set = generate_tasks(number_runnables);
        if feasible_schedule_design_time(&set) {
            tune(set.clone());
            return;
        }
        println!("Infeasible schedule, retrying...\n");
        sleep(Duration::from_millis(100));
    }
}

fn main() {
    hp_tuning(
        std::env::var("NUMBER_RUNNABLES")
            .expect("NUMBER_RUNNABLES not set")
            .parse::<usize>()
            .unwrap(),
    );
}

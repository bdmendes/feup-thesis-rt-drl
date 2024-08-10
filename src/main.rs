use crate::simulator::validation::feasible_schedule_design_time;
use agent::{
    dqn::ActivationFunction, SimulatorAgent, DEFAULT_GAMMA, DEFAULT_LEARNING_RATE,
    DEFAULT_MEM_SIZE, DEFAULT_MIN_MEM_SIZE, DEFAULT_SAMPLE_BATCH_SIZE, DEFAULT_UPDATE_FREQ,
};
use generator::{generate_tasks, Runnable};
use simulator::{task::SimulatorTask, Simulator};
use std::{
    cell::RefCell,
    io::Write,
    rc::Rc,
    sync::{Arc, Mutex},
    thread::{self, sleep},
    time::Duration,
};

pub mod agent;
pub mod generator;
pub mod ml;
pub mod simulator;

fn bad_events(agent: &SimulatorAgent) -> f64 {
    0.6 * (agent.mode_changes_to_hmode() as f64) + 0.4 * (agent.task_kills() as f64)
}

fn write_result(agent: &SimulatorAgent, file: &mut std::fs::File) {
    let contents = format!(
        "Cumulative reward: {}; mode changes to H: {}; mode changes to L: {}; task kills: {}, task starts: {}",
        agent.cumulative_reward(),
        agent.mode_changes_to_hmode(),
        agent.mode_changes_to_lmode(),
        agent.task_kills(),
        agent.task_starts()
    );
    file.write_all(contents.as_bytes()).unwrap();
}

fn tune(tasks: Vec<SimulatorTask>) {
    let train_instants: u64 = Runnable::duration_to_time_unit(Duration::from_secs(10));
    let test_instants: u64 = Runnable::duration_to_time_unit(Duration::from_secs(2));

    let hyper_events = Arc::new(Mutex::new(vec![]));
    let mut hyper_iteration = 0;

    {
        ////////// Placebo //////////
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
        let mut file = std::fs::File::create("out/placebo.txt").unwrap();
        write_result(&agent.borrow(), &mut file);
    }

    // Hyperparameter tuning: train and test
    let hidden_sizes_set = [
        vec![tasks.len() / 2],
        vec![tasks.len(), tasks.len() / 2],
        vec![tasks.len(), tasks.len() / 2, tasks.len() / 4],
    ];
    thread::scope(|scope| {
        for sample_batch_size in [
            DEFAULT_SAMPLE_BATCH_SIZE,
            DEFAULT_SAMPLE_BATCH_SIZE / 2,
            DEFAULT_SAMPLE_BATCH_SIZE * 2,
        ] {
            for hidden_sizes in &hidden_sizes_set {
                for activation_function in [
                    ActivationFunction::Sigmoid,
                    ActivationFunction::ReLU,
                    ActivationFunction::Tanh,
                ] {
                    hyper_iteration += 1;
                    let tasks = tasks.clone();
                    let hyper_events = hyper_events.clone();

                    scope.spawn(move || {
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
                            agent.borrow_mut().quit_training();
                            let mut simulator = Simulator::new(tasks, true, Some(agent.clone()));
                            simulator.fire::<false>(test_instants);
                            let mut file =
                                std::fs::File::create(format!("out/test_{hyper_iteration}.txt"))
                                    .unwrap();
                            write_result(&agent.borrow(), &mut file);
                            hyper_events.lock().unwrap().push((
                                bad_events(&agent.borrow()),
                                DEFAULT_UPDATE_FREQ,
                                sample_batch_size,
                                hidden_sizes.clone(),
                                activation_function,
                            ));
                        }
                    });
                }
            }
        }
    });
}

fn generate_sets(size: usize) -> Vec<Vec<SimulatorTask>> {
    let mut task_sets = vec![];
    while task_sets.len() < size {
        let set = generate_tasks();
        if !feasible_schedule_design_time(&set) {
            continue;
        }
        task_sets.push(set);
    }
    task_sets
}

pub fn hp_tuning() {
    std::fs::create_dir_all("out").unwrap();
    loop {
        let set = generate_sets(1);
        if feasible_schedule_design_time(&set[0]) {
            println!("Feasible schedule found, tuning hyperparameters...");
            sleep(Duration::from_millis(100));
            tune(set[0].clone());
            return;
        }
        println!("Infeasible schedule, retrying...\n");
        sleep(Duration::from_millis(100));
    }
}

pub fn testing_fast() {
    let tasks = generate_tasks();
    assert!(feasible_schedule_design_time(&tasks));
    let mut simulator = Simulator::new(
        tasks.clone(),
        true,
        Some(Rc::new(RefCell::new(SimulatorAgent::new(
            DEFAULT_MEM_SIZE,
            DEFAULT_MIN_MEM_SIZE,
            DEFAULT_GAMMA,
            DEFAULT_UPDATE_FREQ,
            DEFAULT_LEARNING_RATE,
            vec![tasks.len(), tasks.len() / 2],
            DEFAULT_SAMPLE_BATCH_SIZE,
            ActivationFunction::ReLU,
            &tasks,
        )))),
    );
    simulator.fire::<false>(Runnable::duration_to_time_unit(Duration::from_secs(10)));
    println!(
        "Cumulative reward: {}",
        simulator
            .agent
            .as_ref()
            .unwrap()
            .borrow()
            .cumulative_reward()
    );
    println!(
        "Task kills: {}",
        simulator.agent.as_ref().unwrap().borrow().task_kills()
    );
    println!(
        "Mode changes to H: {}",
        simulator
            .agent
            .as_ref()
            .unwrap()
            .borrow()
            .mode_changes_to_hmode()
    );
    println!(
        "Task starts: {}",
        simulator.agent.as_ref().unwrap().borrow().task_starts()
    );
}

fn main() {
    // hp_tuning();
    testing_fast();
}

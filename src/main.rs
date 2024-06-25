use crate::simulator::validation::feasible_schedule_design_time;
use agent::{
    dqn::ActivationFunction, SimulatorAgent, DEFAULT_GAMMA, DEFAULT_LEARNING_RATE,
    DEFAULT_MEM_SIZE, DEFAULT_MIN_MEM_SIZE, DEFAULT_SAMPLE_BATCH_SIZE, DEFAULT_UPDATE_FREQ,
};
use generator::{generate_tasks, Runnable};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use simulator::{task::SimulatorTask, Simulator};
use std::{
    cell::RefCell,
    io::Write,
    rc::Rc,
    sync::{Arc, Mutex},
    thread,
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
            SimulatorAgent::number_of_actions(&tasks),
            SimulatorAgent::number_of_features(&tasks),
        )));
        agent.borrow_mut().placebo_mode();
        let mut simulator = Simulator {
            tasks: tasks.clone(),
            random_execution_time: true,
            agent: Some(agent.clone()),
            elapsed_times: vec![],
            memory_usage: vec![],
        };

        simulator.run::<false>(test_instants);
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
                    let number_of_actions = SimulatorAgent::number_of_actions(&tasks);
                    let number_of_features = SimulatorAgent::number_of_features(&tasks);

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
                            number_of_actions,
                            number_of_features,
                        )));

                        ////////// Training //////////
                        {
                            let mut simulator = Simulator {
                                tasks: tasks.clone(),
                                random_execution_time: true,
                                agent: Some(agent.clone()),
                                elapsed_times: vec![],
                                memory_usage: vec![],
                            };
                            simulator.run::<false>(train_instants);
                        }

                        ////////// Testing //////////
                        {
                            agent.borrow_mut().quit_training();
                            let mut simulator = Simulator {
                                tasks: tasks.clone(),
                                random_execution_time: true,
                                agent: Some(agent.clone()),
                                elapsed_times: vec![],
                                memory_usage: vec![],
                            };
                            simulator.run::<false>(test_instants);
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

fn generate_sets(size: usize, number_runnables: usize) -> Vec<Vec<SimulatorTask>> {
    let mut task_sets = vec![];
    while task_sets.len() < size {
        let set = generate_tasks(
            0.7,
            number_runnables,
            generator::TimeSampleDistribution::Pert,
        );
        if !feasible_schedule_design_time(&set) {
            continue;
        }
        task_sets.push(set);
    }
    task_sets
}

fn simulate_placebo(tasks: Vec<SimulatorTask>, secs: usize) -> (usize, usize) {
    let test_instants: u64 = Runnable::duration_to_time_unit(Duration::from_secs(secs as u64));

    let agent = Rc::new(RefCell::new(SimulatorAgent::new(
        DEFAULT_MEM_SIZE,
        DEFAULT_MIN_MEM_SIZE,
        DEFAULT_GAMMA,
        DEFAULT_UPDATE_FREQ,
        DEFAULT_LEARNING_RATE,
        vec![8],
        DEFAULT_SAMPLE_BATCH_SIZE,
        ActivationFunction::Sigmoid,
        SimulatorAgent::number_of_actions(&tasks),
        SimulatorAgent::number_of_features(&tasks),
    )));
    agent.borrow_mut().placebo_mode();
    let mut simulator = Simulator {
        tasks: tasks.clone(),
        random_execution_time: true,
        agent: Some(agent.clone()),
        elapsed_times: vec![],
        memory_usage: vec![],
    };

    simulator.run::<false>(test_instants);
    let mode_changes_to_hmode = agent.borrow().mode_changes_to_hmode();
    let task_kills = agent.borrow().task_kills();
    (mode_changes_to_hmode, task_kills)
}

pub fn hp_tuning() {
    std::fs::create_dir_all("out").unwrap();
    let set = generate_sets(1, 25);
    tune(set[0].clone());
}

pub fn placebo() {
    std::fs::create_dir_all("out").unwrap();
    let set_25 = generate_sets(50, 25);
    let changes_kills_25 = set_25
        .par_iter()
        .map(|tasks| {
            let (mode_changes_to_h, task_kills) = simulate_placebo(tasks.clone(), 1);
            (mode_changes_to_h, task_kills)
        })
        .collect::<Vec<_>>();
    let mut file_25 = std::fs::File::create("out/changes_kills_25.txt").unwrap();
    for (mode_changes_to_h, task_kills) in changes_kills_25 {
        file_25
            .write_all(
                format!(
                    "Mode changes to H: {}; Task kills: {}\n",
                    mode_changes_to_h, task_kills
                )
                .as_bytes(),
            )
            .unwrap();
    }
    let set_100 = generate_sets(50, 100);
    let changes_kills_100 = set_100
        .par_iter()
        .map(|tasks| {
            let (mode_changes_to_h, task_kills) = simulate_placebo(tasks.clone(), 1);
            (mode_changes_to_h, task_kills)
        })
        .collect::<Vec<_>>();
    let mut file_100 = std::fs::File::create("out/changes_kills_100.txt").unwrap();
    for (mode_changes_to_h, task_kills) in changes_kills_100 {
        file_100
            .write_all(
                format!(
                    "Mode changes to H: {}; Task kills: {}\n",
                    mode_changes_to_h, task_kills
                )
                .as_bytes(),
            )
            .unwrap();
    }
}

fn activation_time_size(hidden_sizes: Vec<usize>, set: &[Vec<SimulatorTask>]) {
    let agent = Rc::new(RefCell::new(SimulatorAgent::new(
        DEFAULT_MEM_SIZE,
        DEFAULT_MIN_MEM_SIZE,
        DEFAULT_GAMMA,
        DEFAULT_UPDATE_FREQ,
        DEFAULT_LEARNING_RATE,
        hidden_sizes.clone(),
        DEFAULT_SAMPLE_BATCH_SIZE,
        ActivationFunction::Sigmoid,
        SimulatorAgent::number_of_actions(&set[0]),
        SimulatorAgent::number_of_features(&set[0]),
    )));
    agent.borrow_mut().placebo_mode();
    agent.borrow_mut().skip_tracking();
    let mut simulator = Simulator {
        tasks: set[0].clone(),
        random_execution_time: true,
        agent: Some(agent.clone()),
        elapsed_times: vec![],
        memory_usage: vec![],
    };
    simulator.run::<false>(Runnable::duration_to_time_unit(Duration::from_secs(2)));

    // write activation times
    let mut file =
        std::fs::File::create(format!("out/activation_times_{}.txt", hidden_sizes.len())).unwrap();
    let _ = file.write_all(
        simulator
            .elapsed_times
            .iter()
            .map(|x| x.as_micros().to_string())
            .collect::<Vec<_>>()
            .join("\n")
            .to_string()
            .as_bytes(),
    );

    // write memory usage
    let mut file =
        std::fs::File::create(format!("out/memory_usage_{}.txt", hidden_sizes.len())).unwrap();
    let _ = file.write_all(
        simulator
            .memory_usage
            .iter()
            .map(|x| format!("{:.2} {:.2}", x.0, x.1))
            .collect::<Vec<_>>()
            .join("\n")
            .to_string()
            .as_bytes(),
    );
}

pub fn activation_time() {
    std::fs::create_dir_all("out").unwrap();
    let set = generate_sets(1, 100);

    activation_time_size(vec![set[0].len() / 2], &set);
    activation_time_size(vec![set[0].len(), set[0].len() / 2, set[0].len() / 4], &set);
}

fn main() {
    //hp_tuning();
    activation_time();
}

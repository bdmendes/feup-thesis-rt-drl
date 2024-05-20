use self::dqn::{Policy, ReplayMemory};
use crate::agent::dqn::Transition;
use crate::ml::tensor::{mean_squared_error, TensorStorage};
use crate::ml::ComputeModel;
use crate::simulator::validation::feasible_schedule_online;
use crate::simulator::{task::TaskId, Simulator, SimulatorEvent};
use crate::simulator::{SimulatorMode, SimulatorTask};
use rand::Rng;
use tch::Tensor;

pub mod dqn;

pub const DEFAULT_MEM_SIZE: usize = 200;
pub const DEFAULT_MIN_MEM_SIZE: usize = 10;
pub const DEFAULT_GAMMA: f32 = 0.99;
pub const DEFAULT_UPDATE_FREQ: usize = 10;
pub const DEFAULT_LEARNING_RATE: f32 = 0.00005;
pub const DEFAULT_SAMPLE_BATCH_SIZE: usize = 8;

#[derive(Debug)]
pub enum SimulatorAction {
    WcetIncrease(TaskId),
    WcetDecrease(TaskId),
    None,
}

impl SimulatorAction {
    fn task_id(&self) -> TaskId {
        match self {
            SimulatorAction::WcetIncrease(id) => *id,
            SimulatorAction::WcetDecrease(id) => *id,
            SimulatorAction::None => panic!("No task id for None action"),
        }
    }

    pub fn apply(&self, tasks: &mut [SimulatorTask]) {
        if matches!(self, SimulatorAction::None) {
            return;
        }

        let task_to_change = tasks
            .iter_mut()
            .find(|t| t.task.props().id == self.task_id())
            .unwrap();
        match self {
            SimulatorAction::WcetIncrease(_) => {
                task_to_change.task.props_mut().wcet_l =
                    task_to_change.task.props().wcet_l.saturating_add(1);
            }
            SimulatorAction::WcetDecrease(_) => {
                task_to_change.task.props_mut().wcet_l =
                    task_to_change.task.props().wcet_l.saturating_sub(1);
            }
            SimulatorAction::None => unreachable!(),
        }
    }

    pub fn reverse(&self) -> SimulatorAction {
        match self {
            SimulatorAction::WcetIncrease(id) => SimulatorAction::WcetDecrease(*id),
            SimulatorAction::WcetDecrease(id) => SimulatorAction::WcetIncrease(*id),
            SimulatorAction::None => SimulatorAction::None,
        }
    }
}

#[derive(Debug, PartialEq)]
enum SimulatorAgentStage {
    // In the data collection stage, we fill the replay memory
    // with transitions collected from experience, to be sampled
    // later for training.
    DataCollection,

    // In the training stage, we
    Training,

    // In the reactive stage, we use the simply use policy network to
    // make decisions based on the current state of the simulator.
    Reactive,
}

pub struct SimulatorAgent {
    // The agent is informed periodically about the state of the simulator.
    pending_events: Vec<SimulatorEvent>,

    // DQN parameters.
    sample_batch_size: usize,
    gamma: f32,
    update_freq: usize,
    learning_rate: f32,
    stage: SimulatorAgentStage,

    // DQN model
    /// The policy network is the one that is being trained.
    /// It receives the state as input and outputs the Q-values for each action.
    policy_network: Policy,

    /// The target network is a snapshot of the policy network that is
    /// used to compute the loss and update the policy network via backpropagation.
    /// DQN uses this to stabilize the learning process.
    target_network: Policy,

    /// The replay memory is a collection of tuples (state, action, reward, state')
    /// stored from experience.
    replay_memory: ReplayMemory,

    memory_policy: TensorStorage,
    memory_target: TensorStorage,
    epsilon: f32,
    reward_history: Vec<f32>,

    buffered_action: Option<SimulatorAction>,
    buffered_state: Option<Tensor>,
    buffered_reward: Option<f64>,
}

impl SimulatorAgent {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        mem_size: usize,
        min_mem_size: usize,
        gamma: f32,
        update_freq: usize,
        learning_rate: f32,
        hidden_size: usize,
        sample_batch_size: usize,
        activation: dqn::ActivationFunction,
        number_of_actions: usize,
        number_of_features: usize,
    ) -> Self {
        let replay_memory = ReplayMemory::new(mem_size, min_mem_size);
        let mut memory_policy = TensorStorage::default();
        let policy_network = Policy::new(
            &mut memory_policy,
            number_of_features,
            number_of_actions,
            hidden_size,
            activation,
        );
        let mut memory_target = TensorStorage::default();
        let target_network = Policy::new(
            &mut memory_target,
            number_of_features,
            number_of_actions,
            hidden_size,
            activation,
        );
        memory_target.copy(&memory_policy);

        Self {
            pending_events: Vec::new(),
            gamma,
            update_freq,
            learning_rate,
            sample_batch_size,
            stage: SimulatorAgentStage::DataCollection,
            policy_network,
            target_network,
            replay_memory,
            memory_policy,
            memory_target,
            epsilon: 1.0,
            reward_history: Vec::new(),
            buffered_action: None,
            buffered_state: None,
            buffered_reward: None,
        }
    }

    pub fn push_event(&mut self, event: SimulatorEvent) {
        self.pending_events.push(event);
    }

    pub fn activate(&mut self, simulator: &mut Simulator, history: &[Option<TaskId>]) {
        println!("\nActivating agent.");

        // Build a state tensor from the simulator's state.
        let state = Self::history_to_input(history, simulator);

        // Get a new action from the policy.
        let action = Self::epsilon_greedy(
            &self.memory_policy,
            &self.policy_network,
            self.epsilon,
            &state,
            simulator,
        );
        println!("Got action: {:?}", action);

        // Apply it to the simulator.
        // If this is not valid, revert the action and ignore it.
        action.apply(&mut simulator.tasks);
        if !feasible_schedule_online(&simulator.tasks) {
            println!("Invalid action {:?}, reverting.", action);
            let reverse_action = action.reverse();
            reverse_action.apply(&mut simulator.tasks);
            self.buffered_reward = Some(-0.25);
        } else {
            println!("Applied action {:?}", action);
            println!("Tasks: {:?}", simulator.tasks);
            self.buffered_reward = Some(0.25);
        }

        if let Some(buffered_action) = &self.buffered_action {
            println!("Last buffered action: {:?}", buffered_action);

            // We had taken an action previously, and are now receiving the reward.
            let reward = self
                .pending_events
                .pop()
                .map_or(0.0, |e| Self::event_to_reward(&e, simulator))
                + self.buffered_reward.unwrap_or(0.0);
            self.reward_history.push(reward as f32);
            println!("Got reward: {}", reward);

            let transition = Transition::new(
                self.buffered_state.as_ref().unwrap(),
                Self::action_to_index(buffered_action, simulator) as i64,
                reward as f32,
                &state,
            );

            println!("Pushing transition to replay memory: {:?}", transition);

            match self.stage {
                SimulatorAgentStage::DataCollection => {
                    if self.replay_memory.add_initial(transition) {
                        // The replay memory is now filled with the minimum number of transitions.
                        self.stage = SimulatorAgentStage::Training;
                    }
                }
                SimulatorAgentStage::Training => {
                    self.replay_memory.add(transition);
                }
                _ => {}
            }
        }

        // Store this action and state to generate a transition later.
        self.buffered_action = Some(action);
        self.buffered_state = Some(state);

        // If we are not training, do nothing else.
        if self.stage != SimulatorAgentStage::Training {
            println!("Not training. Skipping NN activity.");
            return;
        }

        println!("Training.");

        let (b_state, b_action, b_reward, b_state_) =
            self.replay_memory.sample_batch(self.sample_batch_size);
        let qvalues = self
            .policy_network
            .forward(&self.memory_policy, &b_state)
            .gather(1, &b_action, false);

        let target_values: Tensor =
            tch::no_grad(|| self.target_network.forward(&self.memory_target, &b_state_));
        let max_target_values = target_values.max_dim(1, true).0;
        let expected_values = b_reward + self.gamma * (&max_target_values);

        let loss = mean_squared_error(&qvalues, &expected_values);
        loss.backward();
        self.memory_policy.apply_grads_adam(self.learning_rate);

        // We update the target network every `update_freq` steps.
        // This allows for a more stable learning process.
        if self.reward_history.len() % self.update_freq == 0 {
            println!("Updating target network.");
            self.memory_target.copy(&self.memory_policy);

            self.epsilon = (self.epsilon * 0.95).max(0.3);
            println!("Updated epsilon: {}", self.epsilon);
        }
    }

    pub fn quit_training(&mut self) {
        self.stage = SimulatorAgentStage::Reactive;
    }

    pub fn event_to_reward(event: &SimulatorEvent, _simulator: &Simulator) -> f64 {
        match event {
            SimulatorEvent::Start(_, _) => 0.25,
            SimulatorEvent::TaskKill(_, _) => -1.0,
            SimulatorEvent::ModeChange(SimulatorMode::HMode, _) => -2.0,
            _ => 0.0,
        }
    }

    pub fn history_to_input(history: &[Option<TaskId>], simulator: &Simulator) -> Tensor {
        let mut input = Vec::with_capacity(Self::number_of_features(&simulator.tasks));

        for task in &simulator.tasks {
            let wcet_l = task.task.props().wcet_l as f32;
            let deadline = task.task.props().period as f32;
            let offset = task.task.props().offset as f32;
            let priority = task.priority as f32;

            // Assuming there are no overlapping jobs of the same task in a period,
            // the last execution time is the number of instants of the task
            // since the last one + period.
            let last_executed_distance = history.iter().rev().position(|t| {
                if let Some(t) = t {
                    *t == task.task.props().id
                } else {
                    false
                }
            });
            let last_job_execution_time = match last_executed_distance {
                Some(instant) if (instant + task.task.props().period as usize) < history.len() => {
                    history
                        .iter()
                        .rev()
                        .skip(instant)
                        .take(task.task.props().period as usize)
                        .fold(0.0, |acc, t| {
                            if let Some(t) = t {
                                if *t == task.task.props().id {
                                    acc + 1.0
                                } else {
                                    acc
                                }
                            } else {
                                acc
                            }
                        })
                }
                _ => -1.0,
            };

            input.push(wcet_l);
            input.push(deadline);
            input.push(offset);
            input.push(priority);
            input.push(last_job_execution_time);

            println!(
                "Task {}: WCET_L: {}, Deadline: {}, Offset: {}, Priority: {}, Last job execution time: {}",
                task.task.props().id, wcet_l, deadline, offset, priority, last_job_execution_time);
        }

        Tensor::from_slice(input.as_slice())
    }

    pub fn sample_simulator_action(simulator: &Simulator) -> SimulatorAction {
        let index = rand::random::<usize>() % (simulator.tasks.len() + 1);

        if index == simulator.tasks.len() {
            return SimulatorAction::None;
        }

        let action_variant = rand::random::<usize>() % 2;
        match action_variant {
            0 => SimulatorAction::WcetIncrease(simulator.tasks[index].task.props().id),
            1 => SimulatorAction::WcetDecrease(simulator.tasks[index].task.props().id),
            _ => unreachable!(),
        }
    }

    pub fn epsilon_greedy(
        storage: &TensorStorage,
        policy: &dyn ComputeModel,
        epsilon: f32,
        environment: &Tensor,
        simulator: &Simulator,
    ) -> SimulatorAction {
        let mut rng = rand::thread_rng();
        let random_number: f32 = rng.gen::<f32>();
        if random_number > epsilon {
            println!("Using policy.");
            let value = tch::no_grad(|| policy.forward(storage, environment));
            println!("Q-values: {}", value);
            let action_index = value.argmax(1, false).int64_value(&[]) as usize;
            Self::index_to_action(action_index, simulator)
        } else {
            println!("Using random action.");
            Self::sample_simulator_action(simulator)
        }
    }

    pub fn number_of_actions(tasks: &[SimulatorTask]) -> usize {
        // Increase and decrease WCET for each task + No action.
        tasks.len() * 2 + 1
    }

    pub fn number_of_features(tasks: &[SimulatorTask]) -> usize {
        // We'll place the tasks from task to bottom.
        // Each task has 5 features: WCET_L, deadline, offset, priority, last job execution time.
        tasks.len() * 5
    }

    fn index_to_action(index: usize, simulator: &Simulator) -> SimulatorAction {
        let number_of_actions = Self::number_of_actions(&simulator.tasks);
        if index == number_of_actions - 1 {
            SimulatorAction::None
        } else {
            let task_idx = index / 2;
            let task_id = simulator.tasks.get(task_idx).unwrap().task.props().id;
            if index % 2 == 0 {
                SimulatorAction::WcetIncrease(task_id)
            } else {
                SimulatorAction::WcetDecrease(task_id)
            }
        }
    }

    fn action_to_index(action: &SimulatorAction, simulator: &Simulator) -> usize {
        let number_of_actions = Self::number_of_actions(&simulator.tasks);
        match action {
            SimulatorAction::None => number_of_actions - 1,
            SimulatorAction::WcetIncrease(id) => {
                let task_idx = simulator
                    .tasks
                    .iter()
                    .position(|t| t.task.props().id == *id)
                    .unwrap();
                task_idx * 2
            }
            SimulatorAction::WcetDecrease(id) => {
                let task_idx = simulator
                    .tasks
                    .iter()
                    .position(|t| t.task.props().id == *id)
                    .unwrap();
                task_idx * 2 + 1
            }
        }
    }
}

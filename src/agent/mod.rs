use self::dqn::{Policy, ReplayMemory};
use crate::agent::dqn::Transition;
use crate::ml::tensor::{mean_squared_error, TensorStorage};
use crate::ml::ComputeModel;
use crate::simulator::task::{SimulatorTask, TimeUnit};
use crate::simulator::validation::feasible_schedule_online;
use crate::simulator::SimulatorMode;
use crate::simulator::{task::TaskId, Simulator, SimulatorEvent};
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
    WcetIncreaseSmall(TaskId),
    WcetIncreaseMedium(TaskId),
    WcetIncreaseLarge(TaskId),
    WcetDecreaseSmall(TaskId),
    WcetDecreaseMedium(TaskId),
    WcetDecreaseLarge(TaskId),
    None,
}

impl SimulatorAction {
    fn task_id(&self) -> TaskId {
        match self {
            SimulatorAction::WcetIncreaseSmall(id)
            | SimulatorAction::WcetIncreaseMedium(id)
            | SimulatorAction::WcetIncreaseLarge(id)
            | SimulatorAction::WcetDecreaseSmall(id)
            | SimulatorAction::WcetDecreaseMedium(id)
            | SimulatorAction::WcetDecreaseLarge(id) => *id,
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

        let amount =
            (task_to_change.task.props().wcet_h as f32
                * match self {
                    SimulatorAction::WcetIncreaseSmall(_)
                    | SimulatorAction::WcetDecreaseSmall(_) => 0.1,
                    SimulatorAction::WcetIncreaseMedium(_)
                    | SimulatorAction::WcetDecreaseMedium(_) => 0.25,
                    SimulatorAction::WcetIncreaseLarge(_)
                    | SimulatorAction::WcetDecreaseLarge(_) => 0.5,
                    _ => unreachable!(),
                }) as TimeUnit;

        match self {
            SimulatorAction::WcetIncreaseSmall(_)
            | SimulatorAction::WcetIncreaseMedium(_)
            | SimulatorAction::WcetIncreaseLarge(_) => {
                task_to_change.task.props_mut().wcet_l =
                    task_to_change.task.props().wcet_l.saturating_add(amount);
            }
            SimulatorAction::WcetDecreaseSmall(_)
            | SimulatorAction::WcetDecreaseMedium(_)
            | SimulatorAction::WcetDecreaseLarge(_) => {
                task_to_change.task.props_mut().wcet_l =
                    task_to_change.task.props().wcet_l.saturating_sub(amount);
            }
            SimulatorAction::None => unreachable!(),
        }
    }

    pub fn reverse(&self) -> SimulatorAction {
        match self {
            SimulatorAction::WcetIncreaseSmall(id) => SimulatorAction::WcetDecreaseSmall(*id),
            SimulatorAction::WcetIncreaseMedium(id) => SimulatorAction::WcetDecreaseMedium(*id),
            SimulatorAction::WcetIncreaseLarge(id) => SimulatorAction::WcetDecreaseLarge(*id),
            SimulatorAction::WcetDecreaseSmall(id) => SimulatorAction::WcetIncreaseSmall(*id),
            SimulatorAction::WcetDecreaseMedium(id) => SimulatorAction::WcetIncreaseMedium(*id),
            SimulatorAction::WcetDecreaseLarge(id) => SimulatorAction::WcetIncreaseLarge(*id),
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

    // In placebo mode, the agent does nothing and just collects rewards.
    // Used for testing.
    Placebo,
}

pub struct SimulatorAgent {
    // The agent is informed periodically about the state of the simulator.
    events_history: Vec<SimulatorEvent>,
    cumulative_reward: f64,
    mode_changes_to_hmode: usize,
    mode_changes_to_lmode: usize,
    task_kills: usize,

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
        hidden_sizes: Vec<usize>,
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
            hidden_sizes.clone(),
            activation,
        );
        let mut memory_target = TensorStorage::default();
        let target_network = Policy::new(
            &mut memory_target,
            number_of_features,
            number_of_actions,
            hidden_sizes,
            activation,
        );
        memory_target.copy(&memory_policy);

        Self {
            events_history: Vec::new(),
            cumulative_reward: 0.0,
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
            mode_changes_to_hmode: 0,
            mode_changes_to_lmode: 0,
            task_kills: 0,
        }
    }

    pub fn cumulative_reward(&self) -> f64 {
        self.cumulative_reward
    }

    pub fn task_kills(&self) -> usize {
        self.task_kills
    }

    pub fn mode_changes_to_hmode(&self) -> usize {
        self.mode_changes_to_hmode
    }

    pub fn mode_changes_to_lmode(&self) -> usize {
        self.mode_changes_to_lmode
    }

    pub fn push_event(&mut self, event: SimulatorEvent) {
        self.events_history.push(event);
    }

    pub fn activate(&mut self, simulator: &mut Simulator) {
        println!("\nActivating agent.");

        // Build a state tensor from the simulator's state.
        let state = Self::history_to_input(&self.events_history, simulator);

        // Get a new action from the policy.
        let action = match self.stage {
            SimulatorAgentStage::Placebo => SimulatorAction::None,
            _ => Self::epsilon_greedy(
                &self.memory_policy,
                &self.policy_network,
                self.epsilon,
                &state,
                simulator,
            ),
        };
        println!("Got action: {:?}", action);

        // Apply it to the simulator.
        // If this is not valid, revert the action and ignore it.
        action.apply(&mut simulator.tasks);
        if !matches!(action, SimulatorAction::None) {
            let task_changed = simulator
                .tasks
                .iter()
                .find(|t| t.task.props().id == action.task_id())
                .unwrap();
            if !feasible_schedule_online(&simulator.tasks)
                || task_changed.task.props().wcet_l > 2 * task_changed.task.props().wcet_h
            {
                println!("Invalid action {:?}, reverting.", action);
                let reverse_action = action.reverse();
                reverse_action.apply(&mut simulator.tasks);
                self.buffered_reward = Some(-0.05);
            } else {
                println!("Applied action {:?}", action);
                self.buffered_reward = Some(0.05);
            }
        } else {
            self.buffered_reward = Some(0.0);
        }

        if let Some(buffered_action) = &self.buffered_action {
            // We had taken an action previously, and are now receiving the reward.
            let reward = self
                .events_history
                .iter()
                .map(|e| Self::event_to_reward(e, simulator))
                .sum::<f64>()
                + self.buffered_reward.unwrap_or(0.0);
            self.task_kills += self
                .events_history
                .iter()
                .filter(|e| matches!(e, SimulatorEvent::TaskKill(_, _)))
                .count();
            self.mode_changes_to_hmode += self
                .events_history
                .iter()
                .filter(|e| matches!(e, SimulatorEvent::ModeChange(SimulatorMode::HMode, _)))
                .count();
            self.mode_changes_to_lmode += self
                .events_history
                .iter()
                .filter(|e| matches!(e, SimulatorEvent::ModeChange(SimulatorMode::LMode, _)))
                .count();
            self.events_history.clear();
            self.cumulative_reward += reward;
            println!("Cumulative reward: {}", self.cumulative_reward);
            self.reward_history.push(reward as f32);

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
        self.cumulative_reward = 0.0;
        self.task_kills = 0;
        self.mode_changes_to_hmode = 0;
        self.mode_changes_to_lmode = 0;
        self.events_history.clear();
        self.target_network.free(&mut self.memory_target);
    }

    pub fn placebo_mode(&mut self) {
        self.stage = SimulatorAgentStage::Placebo;
        self.cumulative_reward = 0.0;
        self.task_kills = 0;
        self.mode_changes_to_hmode = 0;
        self.mode_changes_to_lmode = 0;
        self.events_history.clear();
    }

    pub fn event_to_reward(event: &SimulatorEvent, _simulator: &Simulator) -> f64 {
        match event {
            SimulatorEvent::Start(_, _) => 0.1,
            SimulatorEvent::TaskKill(_, _) => -1.0,
            SimulatorEvent::ModeChange(SimulatorMode::HMode, _) => -2.0,
            _ => 0.0,
        }
    }

    fn last_task_execution_time(history: &[SimulatorEvent], id: TaskId) -> Option<TimeUnit> {
        let last_end_event_offset = history.iter().rev().position(|e| match e {
            SimulatorEvent::End(e_id, _) => *e_id == id,
            _ => false,
        });

        if let Some(last_end_event_offset) = last_end_event_offset {
            let previous_start_event = history
                .iter()
                .rev()
                .skip(last_end_event_offset)
                .find(|e| match e {
                    SimulatorEvent::Start(e_id, _) => *e_id == id,
                    _ => false,
                })
                .unwrap();
            let end_time = match history.iter().rev().nth(last_end_event_offset).unwrap() {
                SimulatorEvent::End(_, time) => time,
                _ => unreachable!(),
            };
            let start_time = match previous_start_event {
                SimulatorEvent::Start(_, time) => *time,
                _ => unreachable!(),
            };
            return Some((end_time - start_time) as TimeUnit);
        }

        None
    }

    pub fn history_to_input(event_history: &[SimulatorEvent], simulator: &Simulator) -> Tensor {
        let mut input = Vec::with_capacity(Self::number_of_features(&simulator.tasks));

        for task in &simulator.tasks {
            let wcet_l = task.task.props().wcet_l as f32;
            let last_job_execution_time = if let Some(diff_time) =
                Self::last_task_execution_time(event_history, task.task.props().id)
            {
                diff_time as f32
            } else {
                -1.0
            };

            input.push(wcet_l);
            input.push(last_job_execution_time);

            println!(
                "Task {}: WCET_L: {}, Last job execution time: {}",
                task.task.props().id,
                wcet_l,
                last_job_execution_time
            );
        }

        Tensor::from_slice(input.as_slice())
    }

    pub fn sample_simulator_action(simulator: &Simulator) -> SimulatorAction {
        let index = rand::random::<usize>() % (simulator.tasks.len() + 1);

        if index == simulator.tasks.len() {
            return SimulatorAction::None;
        }

        let action_variant = rand::random::<usize>() % 6;
        match action_variant {
            0 => SimulatorAction::WcetIncreaseLarge(simulator.tasks[index].task.props().id),
            1 => SimulatorAction::WcetIncreaseMedium(simulator.tasks[index].task.props().id),
            2 => SimulatorAction::WcetIncreaseSmall(simulator.tasks[index].task.props().id),
            3 => SimulatorAction::WcetDecreaseLarge(simulator.tasks[index].task.props().id),
            4 => SimulatorAction::WcetDecreaseMedium(simulator.tasks[index].task.props().id),
            5 => SimulatorAction::WcetDecreaseSmall(simulator.tasks[index].task.props().id),
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
        // Increase and decrease WCET for each task * 3 diffs + No action.
        tasks.len() * 2 * 3 + 1
    }

    pub fn number_of_features(tasks: &[SimulatorTask]) -> usize {
        // We'll place the tasks from task to bottom.
        // Each task has 2 features: WCET_L and last job execution time.
        tasks.len() * 2
    }

    fn index_to_action(index: usize, simulator: &Simulator) -> SimulatorAction {
        let number_of_actions = Self::number_of_actions(&simulator.tasks);
        if index == number_of_actions - 1 {
            SimulatorAction::None
        } else {
            let task_idx = index / 6;
            let task_id = simulator.tasks.get(task_idx).unwrap().task.props().id;
            match index % 6 {
                0 => SimulatorAction::WcetIncreaseSmall(task_id),
                1 => SimulatorAction::WcetIncreaseMedium(task_id),
                2 => SimulatorAction::WcetIncreaseLarge(task_id),
                3 => SimulatorAction::WcetDecreaseSmall(task_id),
                4 => SimulatorAction::WcetDecreaseMedium(task_id),
                5 => SimulatorAction::WcetDecreaseLarge(task_id),
                _ => unreachable!(),
            }
        }
    }

    fn action_to_index(action: &SimulatorAction, simulator: &Simulator) -> usize {
        let number_of_actions = Self::number_of_actions(&simulator.tasks);
        if matches!(action, SimulatorAction::None) {
            return number_of_actions - 1;
        }

        let id = action.task_id();
        let task_idx = simulator
            .tasks
            .iter()
            .position(|t| t.task.props().id == id)
            .unwrap();

        match action {
            SimulatorAction::None => unreachable!(),
            SimulatorAction::WcetIncreaseSmall(_) => task_idx * 6,
            SimulatorAction::WcetIncreaseMedium(_) => task_idx * 6 + 1,
            SimulatorAction::WcetIncreaseLarge(_) => task_idx * 6 + 2,
            SimulatorAction::WcetDecreaseSmall(_) => task_idx * 6 + 3,
            SimulatorAction::WcetDecreaseMedium(_) => task_idx * 6 + 4,
            SimulatorAction::WcetDecreaseLarge(_) => task_idx * 6 + 5,
        }
    }
}

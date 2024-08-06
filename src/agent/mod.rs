use std::cell::RefCell;
use std::rc::Rc;

use self::dqn::{Policy, ReplayMemory};
use crate::agent::dqn::Transition;
use crate::ml::tensor::{mean_squared_error, TensorStorage};
use crate::ml::ComputeModel;
use crate::simulator::task::{SimulatorTask, TaskProps, TimeUnit};
use crate::simulator::validation::feasible_schedule_online;
use crate::simulator::SimulatorMode;
use crate::simulator::{task::TaskId, Simulator, SimulatorEvent};
use rand::Rng;
use tch::Tensor;

pub mod dqn;

pub const DEFAULT_MEM_SIZE: usize = 200;
pub const DEFAULT_MIN_MEM_SIZE: usize = 20;
pub const DEFAULT_GAMMA: f32 = 0.99;
pub const DEFAULT_UPDATE_FREQ: usize = 5;
pub const DEFAULT_LEARNING_RATE: f32 = 0.00005;
pub const DEFAULT_SAMPLE_BATCH_SIZE: usize = 6;

pub type SimulatorAction = (
    SimulatorActionPart,
    SimulatorActionPart,
    SimulatorActionPart,
);

#[derive(Debug, Eq, PartialEq, Copy, Clone)]
pub enum SimulatorActionPart {
    WcetIncrease(TaskId), // 10% increase
    WcetDecrease(TaskId), // 5% decrease
    None,
}

impl SimulatorActionPart {
    fn task_id(&self) -> TaskId {
        match self {
            SimulatorActionPart::WcetIncrease(id) | SimulatorActionPart::WcetDecrease(id) => *id,
            SimulatorActionPart::None => panic!("No task id for None action"),
        }
    }

    pub fn apply(&self, tasks: &mut [Rc<RefCell<SimulatorTask>>]) {
        if matches!(self, SimulatorActionPart::None) {
            return;
        }

        let task_to_change = tasks
            .iter_mut()
            .find(|t| t.borrow().task.props().id == self.task_id())
            .unwrap();

        let amount = (task_to_change.borrow().task.props().wcet_h as f32
            * match self {
                SimulatorActionPart::WcetIncrease(_) => 0.1,
                SimulatorActionPart::WcetDecrease(_) => 0.05,
                _ => unreachable!(),
            }) as TimeUnit;

        let wcet_l = task_to_change.borrow_mut().task.props().wcet_l;
        match self {
            SimulatorActionPart::WcetIncrease(_) => {
                task_to_change.borrow_mut().task.props_mut().wcet_l = wcet_l.saturating_add(amount);
            }
            SimulatorActionPart::WcetDecrease(_) => {
                task_to_change.borrow_mut().task.props_mut().wcet_l = wcet_l.saturating_sub(amount);
            }
            SimulatorActionPart::None => unreachable!(),
        }
    }

    pub fn reverse(&self) -> SimulatorActionPart {
        match self {
            SimulatorActionPart::WcetIncrease(id) => SimulatorActionPart::WcetDecrease(*id),
            SimulatorActionPart::WcetDecrease(id) => SimulatorActionPart::WcetIncrease(*id),
            SimulatorActionPart::None => SimulatorActionPart::None,
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
    task_starts: usize,
    last_processed_event_index: usize,
    track: bool,
    number_of_features: usize,
    _number_of_actions: usize,
    number_of_tasks: usize,

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
        task_set: &[SimulatorTask],
    ) -> Self {
        let number_of_features = Self::number_of_features(task_set);
        let number_of_actions = Self::number_of_actions(task_set);

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
            track: true,
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
            mode_changes_to_hmode: 0,
            mode_changes_to_lmode: 0,
            task_kills: 0,
            task_starts: 0,
            last_processed_event_index: 0,
            number_of_features,
            _number_of_actions: number_of_actions,
            number_of_tasks: task_set.len(),
        }
    }

    pub fn cumulative_reward(&self) -> f64 {
        self.cumulative_reward
    }

    pub fn task_kills(&self) -> usize {
        self.task_kills
    }

    pub fn task_starts(&self) -> usize {
        self.task_starts
    }

    pub fn mode_changes_to_hmode(&self) -> usize {
        self.mode_changes_to_hmode
    }

    pub fn mode_changes_to_lmode(&self) -> usize {
        self.mode_changes_to_lmode
    }

    pub fn push_event(&mut self, event: SimulatorEvent) {
        if self.events_history.len() > self.replay_memory.capacity - 1 {
            self.events_history.remove(0);
        }
        self.events_history.push(event);
    }

    pub fn skip_tracking(&mut self) {
        self.track = false;
    }

    pub fn activate(&mut self, simulator: &mut Simulator) {
        println!("\nActivating agent.");

        // Build a state tensor from the simulator's state.
        let state = Self::history_to_input(self, &self.events_history, simulator);

        // Get a new action from the policy.
        let action = match self.stage {
            SimulatorAgentStage::Placebo => vec![SimulatorActionPart::None],
            _ => self
                .epsilon_greedy(
                    &self.memory_policy,
                    &self.policy_network,
                    self.epsilon,
                    &state,
                    simulator,
                )
                .map_or(vec![SimulatorActionPart::None], |(a, b, c)| vec![a, b, c]),
        };
        println!("Got action: {:?}", action);

        // Apply it to the simulator.
        // If this is not valid, revert the action and ignore it.
        action.iter().for_each(|a| a.apply(&mut simulator.tasks));
        if !matches!(action[0], SimulatorActionPart::None) {
            if !feasible_schedule_online(&simulator.tasks) {
                println!("Invalid action {:?}, reverting.", action);
                let reverse_action = action.iter().map(|a| a.reverse()).collect::<Vec<_>>();
                reverse_action
                    .iter()
                    .for_each(|a| a.apply(&mut simulator.tasks));
            } else {
                println!("Applied action {:?}", action);
            }
        }

        // Track events.
        if self.track {
            self.task_kills += self
                .events_history
                .iter()
                .skip(self.last_processed_event_index)
                .filter(|e| matches!(e, SimulatorEvent::TaskKill(_, _)))
                .count();
            self.mode_changes_to_hmode += self
                .events_history
                .iter()
                .skip(self.last_processed_event_index)
                .filter(|e| matches!(e, SimulatorEvent::ModeChange(SimulatorMode::HMode, _)))
                .count();
            self.mode_changes_to_lmode += self
                .events_history
                .iter()
                .skip(self.last_processed_event_index)
                .filter(|e| matches!(e, SimulatorEvent::ModeChange(SimulatorMode::LMode, _)))
                .count();
            self.task_starts += self
                .events_history
                .iter()
                .skip(self.last_processed_event_index)
                .filter(|e| matches!(e, SimulatorEvent::Start(_, _)))
                .count();
        }
        let reward = self
            .events_history
            .iter()
            .skip(self.last_processed_event_index)
            .map(|e| Self::event_to_reward(e, simulator))
            .sum::<f64>();
        self.cumulative_reward += reward;
        println!("Cumulative reward: {}", self.cumulative_reward);
        self.reward_history.push(reward as f32);
        self.last_processed_event_index = self.events_history.len();

        if let Some(buffered_action) = &self.buffered_action {
            // We had taken an action previously, and are now receiving the reward.
            let transition = Transition::new(
                self.buffered_state.as_ref().unwrap(),
                self.action_to_index(Some(buffered_action), simulator) as i64,
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
        self.buffered_action = if action == vec![SimulatorActionPart::None] {
            None
        } else {
            Some((action[0], action[1], action[2]))
        };
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
        self.reward_history.clear();
        self.task_kills = 0;
        self.task_starts = 0;
        self.mode_changes_to_hmode = 0;
        self.mode_changes_to_lmode = 0;
        self.events_history.clear();
        self.last_processed_event_index = 0;
        self.buffered_action = None;
    }

    pub fn placebo_mode(&mut self) {
        self.stage = SimulatorAgentStage::Placebo;
        self.cumulative_reward = 0.0;
        self.reward_history.clear();
        self.task_kills = 0;
        self.task_starts = 0;
        self.mode_changes_to_hmode = 0;
        self.mode_changes_to_lmode = 0;
        self.events_history.clear();
        self.last_processed_event_index = 0;
        self.buffered_action = None;
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
        // FIXME: This is not efficient, and does not take into account preemptions.

        let last_end_event_offset = history.iter().rev().position(|e| match e {
            SimulatorEvent::End(task, _, _) => task.borrow().task.props().id == id,
            _ => false,
        });

        if let Some(last_end_event_offset) = last_end_event_offset {
            let end_time = match history.iter().rev().nth(last_end_event_offset).unwrap() {
                SimulatorEvent::End(_, time, _) => time,
                _ => unreachable!(),
            };
            let previous_start_event =
                history
                    .iter()
                    .rev()
                    .skip(last_end_event_offset)
                    .find(|e| match e {
                        SimulatorEvent::Start(task, _) => task.borrow().task.props().id == id,
                        _ => false,
                    });
            let start_time = match previous_start_event {
                Some(SimulatorEvent::Start(_, time)) => *time,
                _ => *end_time,
            };
            return Some((end_time - start_time) as TimeUnit);
        }

        None
    }

    pub fn history_to_input(
        &self,
        event_history: &[SimulatorEvent],
        simulator: &Simulator,
    ) -> Tensor {
        let mut input = Vec::with_capacity(self.number_of_features);

        for task in simulator.tasks.iter().take(self.number_of_tasks) {
            let wcet_l = task.borrow().task.props().wcet_l as f32;
            let last_job_execution_time = if let Some(diff_time) =
                Self::last_task_execution_time(event_history, task.borrow().task.props().id)
            {
                diff_time as f32
            } else {
                -1.0
            };

            input.push(wcet_l);
            input.push(last_job_execution_time);
        }

        Tensor::from_slice(input.as_slice())
    }

    pub fn sample_simulator_action(&self, simulator: &Simulator) -> Option<SimulatorAction> {
        let actions = Self::generate_actions(
            simulator
                .tasks
                .iter()
                .take(self.number_of_tasks)
                .map(|t| t.borrow().task.props())
                .collect::<Vec<_>>()
                .as_slice(),
        );
        let mut rng = rand::thread_rng();
        let action_index = rng.gen_range(0..actions.len() + 1);
        if action_index == actions.len() {
            return None;
        }
        Some(actions[action_index])
    }

    pub fn epsilon_greedy(
        &self,
        storage: &TensorStorage,
        policy: &dyn ComputeModel,
        epsilon: f32,
        environment: &Tensor,
        simulator: &Simulator,
    ) -> Option<SimulatorAction> {
        let mut rng = rand::thread_rng();
        let random_number: f32 = rng.gen::<f32>();
        if random_number > epsilon {
            println!("Using policy.");
            let value = tch::no_grad(|| policy.forward(storage, environment));
            let action_index = value.argmax(1, false).int64_value(&[]) as usize;
            self.index_to_action(action_index, simulator)
        } else {
            println!("Using random action.");
            self.sample_simulator_action(simulator)
        }
    }

    pub fn number_of_actions(tasks: &[SimulatorTask]) -> usize {
        if tasks.len() < 3 {
            return 1; // Only the None action is available.
        }
        Self::generate_actions(
            tasks
                .iter()
                .map(|t| t.task.props())
                .collect::<Vec<_>>()
                .as_slice(),
        )
        .len()
            + 1
    }

    pub fn number_of_features(tasks: &[SimulatorTask]) -> usize {
        // We'll place the tasks from task to bottom.
        // Each task has 2 features: WCET_L and last job execution time.
        tasks.len() * 2
    }

    fn generate_actions(tasks: &[TaskProps]) -> Vec<SimulatorAction> {
        // Actions are tiples (increase(i), decrease(j), decrease(k))
        // where i, j, k are the ids of the tasks.
        let mut actions = Vec::new();

        for prop in tasks {
            let increase_first = SimulatorActionPart::WcetIncrease(prop.id);
            let mut decrease_pairs = vec![];

            for second_prop in tasks {
                if second_prop.id == prop.id {
                    continue;
                }
                let decrease_second = SimulatorActionPart::WcetDecrease(second_prop.id);
                for third_prop in tasks {
                    if third_prop.id == prop.id || third_prop.id == second_prop.id {
                        continue;
                    }
                    let decrease_third = SimulatorActionPart::WcetDecrease(third_prop.id);

                    // Avoid duplicate actions.
                    if decrease_pairs
                        .iter()
                        .any(|(s, t)| *s == decrease_third && *t == decrease_second)
                    {
                        continue;
                    }

                    decrease_pairs.push((decrease_second, decrease_third));
                    actions.push((increase_first, decrease_second, decrease_third));
                }
            }
        }

        actions
    }

    fn index_to_action(&self, index: usize, simulator: &Simulator) -> Option<SimulatorAction> {
        let actions = Self::generate_actions(
            simulator
                .tasks
                .iter()
                .take(self.number_of_tasks)
                .map(|t| t.borrow().task.props())
                .collect::<Vec<_>>()
                .as_slice(),
        );
        if index >= actions.len() {
            return None;
        }
        Some(actions[index])
    }

    fn action_to_index(&self, action: Option<&SimulatorAction>, simulator: &Simulator) -> usize {
        let actions = Self::generate_actions(
            simulator
                .tasks
                .iter()
                .take(self.number_of_tasks)
                .map(|t| t.borrow().task.props())
                .collect::<Vec<_>>()
                .as_slice(),
        );

        if action.is_none() {
            return actions.len(); // None is the last action.
        }
        actions
            .iter()
            .position(|a| a == action.unwrap())
            .expect("Action not found.")
    }
}

#[cfg(test)]
mod tests {
    use crate::simulator::task::TaskProps;

    #[test]
    fn generate_actions() {
        let props = vec![
            TaskProps::new_empty(0),
            TaskProps::new_empty(1),
            TaskProps::new_empty(2),
            TaskProps::new_empty(3),
            TaskProps::new_empty(4),
            TaskProps::new_empty(5),
        ];
        let actions = super::SimulatorAgent::generate_actions(&props);
        for action in &actions {
            println!("{:?}", action);
        }

        let expected_number = 6 * (5 * 4) / 2;
        assert_eq!(actions.len(), expected_number);
    }
}

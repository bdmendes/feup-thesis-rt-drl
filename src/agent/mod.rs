use self::dqn::{Policy, ReplayMemory};
use crate::ml::tensor::TensorStorage;
use crate::ml::ComputeModel;
use crate::simulator::SimulatorTask;
use crate::simulator::{
    task::{Task, TaskId},
    Simulator, SimulatorEvent,
};
use rand::Rng;
use tch::Tensor;

pub mod dqn;

pub const DEFAULT_MEM_SIZE: usize = 30000;
pub const DEFAULT_MIN_MEM_SIZE: usize = 5000;
pub const DEFAULT_GAMMA: f32 = 0.99;
pub const DEFAULT_UPDATE_FREQ: i64 = 50;
pub const DEFAULT_LEARNING_RATE: f32 = 0.00005;

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
        let task_to_change = tasks
            .iter_mut()
            .find(|t| t.task.props().id == self.task_id())
            .unwrap();
        match self {
            SimulatorAction::WcetIncrease(_) => {
                task_to_change.task.props().wcet_l += 1;
            }
            SimulatorAction::WcetDecrease(_) => {
                task_to_change.task.props().wcet_l -= 1;
            }
            SimulatorAction::None => (),
        }
    }
}

pub struct SimulatorAgent {
    // The agent is informed periodically about the state of the simulator.
    pending_events: Vec<SimulatorEvent>,
    training: bool,
    last_processed_history_index: usize,

    // DQN parameters
    mem_size: usize,
    min_mem_size: usize,
    gamma: f32,
    update_freq: i64,
    learning_rate: f32,

    // DQN model
    policy_network: Policy,
    target_network: Policy,
    memory_replay: ReplayMemory,
    memory_policy: TensorStorage,
    memory_target: TensorStorage,
    epsilon: f32,
    ret: f32,
}

impl SimulatorAgent {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        mem_size: usize,
        min_mem_size: usize,
        gamma: f32,
        update_freq: i64,
        learning_rate: f32,
        hidden_size: u64,
        activation: dqn::ActivationFunction,
        simulator: &Simulator,
    ) -> Self {
        let number_of_actions = Self::number_of_actions(simulator);
        let number_of_features = Self::number_of_features(simulator);

        let memory_replay = ReplayMemory::new(mem_size, min_mem_size);
        let mut memory_policy = TensorStorage::default();
        let policy_network = Policy::new(
            &mut memory_policy,
            number_of_features as u64,
            number_of_actions as u64,
            hidden_size,
            activation,
        );
        let mut memory_target = TensorStorage::default();
        let target_network = Policy::new(
            &mut memory_target,
            number_of_features as u64,
            number_of_actions as u64,
            hidden_size,
            activation,
        );
        memory_target.copy(&memory_policy);

        Self {
            pending_events: Vec::new(),
            training: false,
            last_processed_history_index: 0,
            mem_size,
            min_mem_size,
            gamma,
            update_freq,
            learning_rate,
            policy_network,
            target_network,
            memory_replay,
            memory_policy,
            memory_target,
            ret: 0.0,
            epsilon: 1.0,
        }
    }

    pub fn push_event(&mut self, event: SimulatorEvent) {
        self.pending_events.push(event);
    }

    pub fn activate(&mut self, simulator: &mut Simulator, history: &[Option<TaskId>]) {
        // We take the slice of new/unknown task runs to build a state tensor.
        let history_slice = &history[self.last_processed_history_index..];
        self.last_processed_history_index = history.len();
        let state = Self::history_slice_to_input(history_slice, simulator);

        // Get a new action from the policy.
        let action = Self::epsilon_greedy(
            &self.memory_policy,
            &self.policy_network,
            self.epsilon,
            &state,
            simulator,
        );

        // Validate the action and apply it to the simulator.
        action.apply(&mut simulator.tasks);
    }

    pub fn event_to_reward(event: &SimulatorEvent, simulator: &Simulator) -> f64 {
        todo!("Event to tensor input")
    }

    pub fn history_slice_to_input(
        history_slice: &[Option<TaskId>],
        simulator: &Simulator,
    ) -> Tensor {
        todo!("History slice to tensor input")
    }

    pub fn sample_simulator_action(simulator: &Simulator) -> SimulatorAction {
        let ltasks = simulator
            .tasks
            .iter()
            .filter(|t| matches!(t.task, Task::LTask(_)))
            .collect::<Vec<_>>();
        let index = rand::random::<usize>() % ltasks.len();
        let action_variant = rand::random::<usize>() % 2;
        match action_variant {
            0 => SimulatorAction::WcetIncrease(ltasks[index].task.props().id),
            1 => SimulatorAction::WcetDecrease(ltasks[index].task.props().id),
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
        let number_of_actions = Self::number_of_actions(simulator);

        let mut rng = rand::thread_rng();
        let random_number: f32 = rng.gen::<f32>();
        if random_number > epsilon {
            let value = tch::no_grad(|| policy.forward(storage, environment));
            let action_index = value.argmax(1, false).int64_value(&[]) as usize;
            if action_index == number_of_actions - 1 {
                SimulatorAction::None
            } else {
                let task_idx = action_index / 2;
                let task_id = simulator.tasks.get(task_idx).unwrap().task.props().id;
                if action_index % 2 == 0 {
                    SimulatorAction::WcetIncrease(task_id)
                } else {
                    SimulatorAction::WcetDecrease(task_id)
                }
            }
        } else {
            Self::sample_simulator_action(simulator)
        }
    }

    pub fn epsilon_update(
        cur_reward: f32,
        min_reward: f32,
        max_reward: f32,
        min_eps: f32,
        max_eps: f32,
    ) -> f32 {
        if cur_reward < min_reward {
            return max_eps;
        }
        let reward_range = max_reward - min_reward;
        let eps_range = max_eps - min_eps;
        let min_update = eps_range / reward_range;
        let new_eps = (max_reward - cur_reward) * min_update;
        if new_eps < min_eps {
            min_eps
        } else {
            new_eps
        }
    }

    pub fn number_of_actions(simulator: &Simulator) -> usize {
        // Increase and decrease WCET for each task + No action.
        simulator.tasks.iter().count() * 2 + 1
    }

    pub fn number_of_events(simulator: &Simulator) -> usize {
        // Task kill for each LTask + mode change + No event.
        simulator
            .tasks
            .iter()
            .filter(|t| matches!(t.task, Task::LTask(_)))
            .count()
            + 2
    }

    pub fn number_of_features(simulator: &Simulator) -> usize {
        unimplemented!("Number of features")
    }
}

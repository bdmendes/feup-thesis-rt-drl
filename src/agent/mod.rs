use rand::Rng;
use tch::Tensor;

use crate::{
    ml::{tensor::TensorStorage, ComputeModel},
    simulator::{
        task::{Task, TaskId},
        Simulator, SimulatorEvent, SimulatorTask,
    },
};

pub mod dqn;

pub enum SimulatorAction {
    WcetIncrease(TaskId),
    WcetDecrease(TaskId),
    None,
}

#[derive(Default, Clone, PartialEq, Debug)]
pub struct SimulatorAgent {
    pending_events: Vec<SimulatorEvent>,
    training: bool,
}

impl SimulatorAgent {
    pub fn push_event(&mut self, event: SimulatorEvent) {
        self.pending_events.push(event);
    }

    pub fn activate(&mut self, simulator: &mut Simulator) {
        todo!("Activate agent");
    }

    pub fn event_to_tensor_input(event: &SimulatorEvent, simulator: &Simulator) -> Tensor {
        todo!("Event to tensor input")
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
        event_input: &Tensor,
        simulator: &Simulator,
    ) -> SimulatorAction {
        let number_of_actions = Self::number_of_actions(simulator);

        let mut rng = rand::thread_rng();
        let random_number: f32 = rng.gen::<f32>();
        if random_number > epsilon {
            let value = tch::no_grad(|| policy.forward(storage, event_input));
            let action_index = value.argmax(1, false).int64_value(&[]) as usize;
            if action_index == number_of_actions - 1 {
                SimulatorAction::None
            } else {
                let task_idx = action_index / 2;
                let task_id = simulator
                    .tasks
                    .iter()
                    .filter(|t| matches!(t.task, Task::LTask(_)))
                    .nth(task_idx)
                    .unwrap()
                    .task
                    .props()
                    .id;
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
        // Increase and decrease WCET for each LTask + No action.
        simulator
            .tasks
            .iter()
            .filter(|t| matches!(t.task, Task::LTask(_)))
            .count()
            * 2
            + 1
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
}

use std::collections::VecDeque;

use rand::Rng;
use tch::Tensor;

use crate::ml::{linear::LinearLayer, tensor::TensorStorage, ComputeModel};

#[derive(Debug, Clone, Copy)]
pub enum ActivationFunction {
    Tanh,
    ReLU,
    Sigmoid,
}

pub struct Policy {
    l1: LinearLayer,
    l2: LinearLayer,
    activation: ActivationFunction,
}

impl Policy {
    pub fn new(
        storage: &mut TensorStorage,
        number_features: usize,
        number_actions: usize,
        hidden_size: usize,
        activation: ActivationFunction,
    ) -> Policy {
        let l1 = LinearLayer::new(storage, number_features as i64, hidden_size as i64);
        let l2 = LinearLayer::new(storage, hidden_size as i64, number_actions as i64);
        Self { l1, l2, activation }
    }
}

impl ComputeModel for Policy {
    fn forward(&self, storage: &TensorStorage, input: &Tensor) -> Tensor {
        let mut o = self.l1.forward(storage, input);
        o = match self.activation {
            ActivationFunction::Tanh => o.tanh(),
            ActivationFunction::ReLU => o.relu(),
            ActivationFunction::Sigmoid => o.sigmoid(),
        };
        o = self.l2.forward(storage, &o);
        o
    }
}

#[derive(Debug)]
pub struct Transition {
    state: Tensor,
    action: i64,
    reward: f32,
    state_: Tensor,
}

impl Transition {
    pub fn new(state: &Tensor, action: i64, reward: f32, state_: &Tensor) -> Self {
        Self {
            state: state.shallow_clone(),
            action,
            reward,
            state_: state_.shallow_clone(),
        }
    }
}

pub struct ReplayMemory {
    transitions: VecDeque<Transition>,
    capacity: usize,
    min_size: usize,
}

impl ReplayMemory {
    pub fn new(capacity: usize, min_size: usize) -> Self {
        Self {
            transitions: VecDeque::new(),
            capacity,
            min_size,
        }
    }

    pub fn add(&mut self, transition: Transition) {
        self.transitions.push_back(transition);
        if self.transitions.len() > self.capacity {
            self.transitions.pop_front();
        }
    }

    pub fn add_initial(&mut self, transition: Transition) -> bool {
        if self.transitions.len() < self.min_size {
            self.add(transition);
        }
        self.transitions.len() >= self.min_size
    }

    pub fn sample_batch(&self, size: usize) -> (Tensor, Tensor, Tensor, Tensor) {
        let index: Vec<usize> = (0..size)
            .map(|_| rand::thread_rng().gen_range(0..self.transitions.len()))
            .collect();
        let mut states: Vec<Tensor> = Vec::new();
        let mut actions: Vec<i64> = Vec::new();
        let mut rewards: Vec<f32> = Vec::new();
        let mut states_: Vec<Tensor> = Vec::new();
        index.iter().for_each(|i| {
            let transition = self.transitions.get(*i).unwrap();
            states.push(transition.state.shallow_clone());
            actions.push(transition.action);
            rewards.push(transition.reward);
            states_.push(transition.state_.shallow_clone());
        });
        (
            Tensor::stack(&states, 0),
            Tensor::from_slice(actions.as_slice()).unsqueeze(1),
            Tensor::from_slice(rewards.as_slice()).unsqueeze(1),
            Tensor::stack(&states_, 0),
        )
    }
}

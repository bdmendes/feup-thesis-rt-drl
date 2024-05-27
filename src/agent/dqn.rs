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
    layers: Vec<LinearLayer>,
    activation: ActivationFunction,
}

impl Policy {
    pub fn new(
        storage: &mut TensorStorage,
        number_features: usize,
        number_actions: usize,
        hidden_sizes: Vec<usize>,
        activation: ActivationFunction,
    ) -> Policy {
        assert!(!hidden_sizes.is_empty());
        let mut layers = Vec::new();

        for (i, size) in hidden_sizes.iter().enumerate() {
            let input_size = if i == 0 {
                number_features
            } else {
                hidden_sizes[i - 1]
            };
            let output_size = *size;
            layers.push(LinearLayer::new(
                storage,
                input_size as i64,
                output_size as i64,
            ));
        }

        layers.push(LinearLayer::new(
            storage,
            hidden_sizes[hidden_sizes.len() - 1] as i64,
            number_actions as i64,
        ));

        Self { layers, activation }
    }
}

impl ComputeModel for Policy {
    fn forward(&self, storage: &TensorStorage, input: &Tensor) -> Tensor {
        let mut o = self.layers.first().unwrap().forward(storage, input);

        for i in 0..self.layers.len() - 1 {
            if i > 0 {
                o = self.layers[i].forward(storage, &o);
            }
            o = match self.activation {
                ActivationFunction::Tanh => o.tanh(),
                ActivationFunction::ReLU => o.relu(),
                ActivationFunction::Sigmoid => o.sigmoid(),
            };
        }

        o = self.layers.last().unwrap().forward(storage, &o);
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

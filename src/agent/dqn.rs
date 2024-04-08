use std::collections::VecDeque;

use rand::Rng;
use tch::Tensor;

use crate::ml::{tensor::TensorStorage, ComputeModel};

pub fn epsilon_greedy(
    storage: &TensorStorage,
    policy: &dyn ComputeModel,
    epsilon: f32,
    obs: &Tensor,
) -> i64 {
    let mut rng = rand::thread_rng();
    let random_number: f32 = rng.gen::<f32>();
    if random_number > epsilon {
        let value = tch::no_grad(|| policy.forward(storage, obs));
        value.argmax(1, false).int64_value(&[])
    } else {
        todo!("sample action");
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

pub struct Transition {
    state: Tensor,
    action: i64,
    reward: f32,
    done: Tensor,
    state_: Tensor,
}

impl Transition {
    pub fn new(state: &Tensor, action: i64, reward: f32, done: bool, state_: &Tensor) -> Self {
        Self {
            state: state.shallow_clone(),
            action,
            reward,
            done: Tensor::from(done as i32 as f32),
            state_: state_.shallow_clone(),
        }
    }
}

pub struct RunningStat<T> {
    values: VecDeque<T>,
    capacity: usize,
}

impl<T> RunningStat<T> {
    pub fn new(capacity: usize) -> Self {
        Self {
            values: VecDeque::new(),
            capacity,
        }
    }

    pub fn add(&mut self, val: T) {
        self.values.push_back(val);
        if self.values.len() > self.capacity {
            self.values.pop_front();
        }
    }

    pub fn sum(&self) -> T
    where
        T: std::iter::Sum,
        T: Clone,
    {
        self.values.iter().cloned().sum()
    }

    pub fn average(&self) -> f32
    where
        T: std::iter::Sum,
        T: std::ops::Div<f32, Output = T>,
        T: Clone,
        T: Into<f32>,
    {
        let sum = self.sum();
        (sum / (self.capacity as f32)).into()
    }
}

pub struct ReplayMemory {
    transitions: VecDeque<Transition>,
    capacity: usize,
}

impl ReplayMemory {
    pub fn new(capacity: usize) -> Self {
        Self {
            transitions: VecDeque::new(),
            capacity,
        }
    }

    pub fn add(&mut self, transition: Transition) {
        self.transitions.push_back(transition);
        if self.transitions.len() > self.capacity {
            self.transitions.pop_front();
        }
    }

    pub fn sample_batch(&self, size: usize) -> (Tensor, Tensor, Tensor, Tensor, Tensor) {
        let index: Vec<usize> = (0..size)
            .map(|_| rand::thread_rng().gen_range(0..self.transitions.len()))
            .collect();
        let mut states: Vec<Tensor> = Vec::new();
        let mut actions: Vec<i64> = Vec::new();
        let mut rewards: Vec<f32> = Vec::new();
        let mut dones: Vec<Tensor> = Vec::new();
        let mut states_: Vec<Tensor> = Vec::new();
        index.iter().for_each(|i| {
            let transition = self.transitions.get(*i).unwrap();
            states.push(transition.state.shallow_clone());
            actions.push(transition.action);
            rewards.push(transition.reward);
            dones.push(transition.done.shallow_clone());
            states_.push(transition.state_.shallow_clone());
        });
        (
            Tensor::stack(&states, 0),
            Tensor::from_slice(actions.as_slice()).unsqueeze(1),
            Tensor::from_slice(rewards.as_slice()).unsqueeze(1),
            Tensor::stack(&dones, 0).unsqueeze(1),
            Tensor::stack(&states_, 0),
        )
    }

    // pub fn init(&mut self) {
    //     let mut state = todo!("reset");
    //     let stepskip = 4;
    //     for s in 0..(self.minsize * stepskip) {
    //         let action = todo!("sample discrete action");
    //         let (state_, reward, done) = todo!("step(action)");
    //         if s % stepskip == 0 {
    //             let t = Transition::new(&state, action, reward, done, &state_);
    //             self.add(t);
    //         }
    //         if done {
    //             state = todo!("reset");
    //         } else {
    //             state = state_;
    //         }
    //     }
    // }
}

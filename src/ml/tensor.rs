use tch::{Kind, Tensor};

use super::DEVICE;

#[derive(Default, Debug)]
pub struct TensorStorage {
    values: Vec<Tensor>,
}

impl ToString for TensorStorage {
    fn to_string(&self) -> String {
        self.values
            .iter()
            .map(|t| {
                if t.requires_grad() {
                    format!("{:?} ", t.size())
                } else {
                    "".to_string()
                }
            })
            .reduce(|acc, s| acc + &s)
            .unwrap_or("".to_string())
    }
}

impl TensorStorage {
    pub fn copy(&mut self, source_storage: &TensorStorage) {
        self.values = source_storage
            .values
            .iter()
            .map(|t| t.copy().set_requires_grad(false))
            .collect();
    }

    pub fn size(&self) -> usize {
        self.values.len()
    }

    pub fn push_tensor(&mut self, value: Tensor) -> usize {
        self.values.push(value);
        self.values.len() - 1
    }

    pub fn push(&mut self, size: &[i64], requires_grad: bool) -> usize {
        let t = Tensor::randn(size, (Kind::Float, DEVICE)).requires_grad_(requires_grad);
        self.push_tensor(t)
    }

    pub fn get(&self, index: usize) -> &Tensor {
        &self.values[index]
    }

    pub fn set(&mut self, index: usize, value: Tensor) {
        self.values[index] = value;
    }

    pub fn apply_grads_sgd(&mut self, learning_rate: f32) {
        let mut g = Tensor::new();
        self.values.iter_mut().for_each(|t| {
            if t.requires_grad() {
                g = t.grad();
                t.set_data(&(t.data() - learning_rate * &g));
                t.zero_grad();
            }
        });
    }
}

pub fn mean_squared_error(target: &Tensor, pred: &Tensor) -> Tensor {
    pred.smooth_l1_loss(target, tch::Reduction::Mean, 0.0)
}

pub fn cross_entropy(target: &Tensor, pred: &Tensor) -> Tensor {
    pred.log_softmax(-1, Kind::Float).nll_loss(target)
}

pub fn accuracy(target: &Tensor, pred: &Tensor) -> f64 {
    let yhat = pred.argmax(1, true).squeeze();
    let eq = target.eq_tensor(&yhat);
    let accuracy: f64 = (eq.sum(Kind::Int64) / target.size()[0]).double_value(&[]);
    accuracy
}

pub fn zeros(size: &[i64]) -> Tensor {
    Tensor::zeros(size, (Kind::Float, DEVICE))
}

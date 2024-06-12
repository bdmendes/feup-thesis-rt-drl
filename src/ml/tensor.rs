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

    fn push_tensor(&mut self, value: Tensor) -> usize {
        self.values.push(value);
        self.values.len() - 1
    }

    pub fn push(&mut self, size: &[i64], requires_grad: bool) -> usize {
        let t = Tensor::randn(size, (Kind::Float, DEVICE)).requires_grad_(requires_grad);
        self.push_tensor(t)
    }

    pub fn free_at(&mut self, index: usize) {
        self.values[index] = Tensor::new();
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

    pub fn apply_grads_adam(&mut self, learning_rate: f32) {
        let mut g = Tensor::new();
        const BETA: f32 = 0.9;

        let mut velocity = zeros(&[self.size() as i64]).split(1, 0);
        let mut mom = zeros(&[self.size() as i64]).split(1, 0);
        let mut vel_corr = zeros(&[self.size() as i64]).split(1, 0);
        let mut mom_corr = zeros(&[self.size() as i64]).split(1, 0);
        let mut counter = 0;

        self.values.iter_mut().for_each(|t| {
            if t.requires_grad() {
                g = t.grad();
                g = g.clamp(-1, 1);
                mom[counter] = BETA * &mom[counter] + (1.0 - BETA) * &g;
                velocity[counter] =
                    BETA * &velocity[counter] + (1.0 - BETA) * (&g.pow(&Tensor::from(2)));
                mom_corr[counter] =
                    &mom[counter] / (Tensor::from(1.0 - BETA).pow(&Tensor::from(2)));
                vel_corr[counter] =
                    &velocity[counter] / (Tensor::from(1.0 - BETA).pow(&Tensor::from(2)));

                t.set_data(
                    &(t.data()
                        - learning_rate
                            * (&mom_corr[counter] / (&velocity[counter].sqrt() + 0.0000001))),
                );
                t.zero_grad();
            }
            counter += 1;
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

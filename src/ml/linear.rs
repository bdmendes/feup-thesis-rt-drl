use std::collections::HashMap;

use tch::Tensor;

use super::{tensor::TensorStorage, ComputeModel};

pub struct Linear {
    params: HashMap<String, usize>,
}

impl Linear {
    pub fn new(mem: &mut TensorStorage, ninputs: i64, noutputs: i64) -> Self {
        let mut p = HashMap::new();
        p.insert("W".to_string(), mem.push(&[ninputs, noutputs], true));
        p.insert("b".to_string(), mem.push(&[1, noutputs], true));
        Self { params: p }
    }
}

impl ComputeModel for Linear {
    fn forward(&self, mem: &TensorStorage, input: &Tensor) -> Tensor {
        let w = mem.get(*self.params.get(&"W".to_string()).unwrap());
        let b = mem.get(*self.params.get(&"b".to_string()).unwrap());
        input.matmul(w) + b
    }
}

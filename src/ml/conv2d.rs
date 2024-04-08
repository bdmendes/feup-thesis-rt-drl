use std::collections::HashMap;

use tch::{Kind, Tensor};

use super::{tensor::TensorStorage, ComputeModel, DEVICE};

pub struct Conv2d {
    params: HashMap<String, usize>,
    stride: i64,
}

impl Conv2d {
    pub fn new(
        mem: &mut TensorStorage,
        kernel_size: i64,
        in_channel: i64,
        out_channel: i64,
        stride: i64,
    ) -> Self {
        let mut p = HashMap::new();
        p.insert(
            "kernel".to_string(),
            mem.push(&[out_channel, in_channel, kernel_size, kernel_size], true),
        );
        p.insert(
            "bias".to_string(),
            mem.push_tensor(
                Tensor::full([out_channel], 0.0, (Kind::Float, DEVICE)).requires_grad_(true),
            ),
        );
        Self { params: p, stride }
    }
}

impl ComputeModel for Conv2d {
    fn forward(&self, mem: &TensorStorage, input: &Tensor) -> Tensor {
        let kernel = mem.get(*self.params.get(&"kernel".to_string()).unwrap());
        let bias = mem.get(*self.params.get(&"bias".to_string()).unwrap());
        input.conv2d(kernel, Some(bias), [self.stride], 0, [1], 1)
    }
}

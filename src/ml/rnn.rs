use std::collections::HashMap;

use tch::Tensor;

use super::{
    tensor::{zeros, TensorStorage},
    ComputeModel,
};

pub struct RNN {
    params: HashMap<String, usize>,
    out_seq_len: i64,
    linear_layer: bool,
}

impl RNN {
    pub fn new(
        mem: &mut TensorStorage,
        input_size: i64,
        hidden_size: i64,
        linear_layer: bool,
        linear_out_size: i64,
        out_seq_len: i64,
    ) -> Self {
        let mut p = HashMap::new();
        p.insert(
            "Wxh".to_string(),
            mem.push(&[input_size, hidden_size], true),
        );
        p.insert(
            "Whh".to_string(),
            mem.push(&[hidden_size, hidden_size], true),
        );
        p.insert("bh".to_string(), mem.push(&[hidden_size], true));
        if linear_layer {
            p.insert(
                "W".to_string(),
                mem.push(&[hidden_size, linear_out_size], true),
            );
            p.insert("b".to_string(), mem.push(&[1, linear_out_size], true));
        }
        Self {
            params: p,
            out_seq_len,
            linear_layer,
        }
    }

    pub fn set_h0(&self, mem: &mut TensorStorage, h0: Tensor) {
        let h0_addr = self.params["h0"];
        mem.set(h0_addr, h0);
    }
}

impl ComputeModel for RNN {
    fn forward(&self, mem: &TensorStorage, input: &Tensor) -> Tensor {
        let wxh = mem.get(*self.params.get(&"Wxh".to_string()).unwrap());
        let whh = mem.get(*self.params.get(&"Whh".to_string()).unwrap());
        let bh = mem.get(*self.params.get(&"bh".to_string()).unwrap());
        let mut w = &Tensor::from(0.0);
        let mut b = &Tensor::from(0.0);
        if self.linear_layer {
            w = mem.get(*self.params.get(&"W".to_string()).unwrap());
            b = mem.get(*self.params.get(&"b".to_string()).unwrap());
        }
        let batchsize = input.size()[0]; // input = datapoints x timesteps x features
        let timesteps = input.size()[1];
        let out_start = timesteps - self.out_seq_len;

        let mut h: Tensor;
        if let Some(h0_index) = self.params.get(&"h0".to_string()) {
            h = mem.get(*h0_index).copy();
        } else {
            h = zeros(&[batchsize, bh.size()[0]]);
        };
        let mut out: Vec<Tensor> = Vec::new();
        let mut out_h: Vec<Tensor> = Vec::new();
        for i in 0..timesteps {
            let row = input.narrow(1, i, 1).squeeze_dim(1);
            h = (row.matmul(wxh) + h.matmul(whh) + bh).tanh();
            out_h.push(h.copy());
            if self.linear_layer {
                out.push(h.matmul(w) + b);
            }
        }
        let output: &Vec<Tensor> = if self.linear_layer { &out } else { &out_h };
        let res = Tensor::concat(output.as_slice(), 1).reshape([batchsize, timesteps, -1]);
        res.narrow(1, out_start, timesteps - out_start)
    }
}

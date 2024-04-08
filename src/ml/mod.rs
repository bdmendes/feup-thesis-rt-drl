use tch::{Device, Kind, Tensor};

use self::tensor::TensorStorage;

pub mod conv2d;
pub mod linear;
pub mod rnn;
pub mod tensor;

pub const DEVICE: Device = Device::Cpu;

pub trait ComputeModel {
    fn forward(&self, storage: &TensorStorage, input: &Tensor) -> Tensor;
}

fn get_batches(
    x: &Tensor,
    y: &Tensor,
    batch_size: i64,
    shuffle: bool,
) -> impl Iterator<Item = (Tensor, Tensor)> {
    let num_rows = x.size()[0];
    let num_batches = (num_rows + batch_size - 1) / batch_size;

    let indices = if shuffle {
        Tensor::randperm(num_rows, (Kind::Int64, DEVICE))
    } else {
        let rng = (0..num_rows).collect::<Vec<i64>>();
        Tensor::from_slice(&rng)
    };
    let x = x.index_select(0, &indices);
    let y = y.index_select(0, &indices);

    (0..num_batches).map(move |i| {
        let start = i * batch_size;
        let end = (start + batch_size).min(num_rows);
        let batchx: Tensor = x.narrow(0, start, end - start);
        let batchy: Tensor = y.narrow(0, start, end - start);
        (batchx, batchy)
    })
}

#[allow(clippy::too_many_arguments)]
pub fn train<F>(
    mem: &mut TensorStorage,
    x: &Tensor,
    y: &Tensor,
    model: &dyn ComputeModel,
    epochs: i64,
    batch_size: i64,
    error_func: F,
    learning_rate: f32,
) where
    F: Fn(&Tensor, &Tensor) -> Tensor,
{
    for epoch in 0..epochs {
        let mut batch_error = Tensor::from(0.0);
        for (batchx, batchy) in get_batches(x, y, batch_size, true) {
            let pred = model.forward(mem, &batchx);
            let error = error_func(&batchy, &pred);
            batch_error += error.detach();
            error.backward();
            mem.apply_grads_sgd(learning_rate);
        }

        println!("Epoch: {:?} Error: {:?}", epoch, batch_error / batch_size);
    }
}

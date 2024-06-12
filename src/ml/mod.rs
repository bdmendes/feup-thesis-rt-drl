use tch::{Device, Tensor};

use self::tensor::TensorStorage;

pub mod linear;
pub mod tensor;

pub const DEVICE: Device = Device::Cpu;

pub trait ComputeModel {
    fn forward(&self, storage: &TensorStorage, input: &Tensor) -> Tensor;
}

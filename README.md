# feup-thesis-rt-drl
Code for my dissertation at FEUP, related to optimizing task scheduling on a real-time system. Supervised by professors Pedro Souto and Pedro Diniz.

You can read the dissertation with a detailed explanation of the work [here](/doc/thesis.pdf).

The methodology was improved for the conference paper draft, which can be read [here](/doc/article.pdf).

## Compilation

Make sure you have an updated Rust toolchain in your machine. Then, simply:

```sh
cargo build --release
```

## Testing

```sh
cargo test --release
```

## Executing

First, set the environment variables:

- NUMBER_RUNNABLES: number of (automotive) runnables to be generated in each task set
- TRAIN_INSTANTS: number of simulated seconds to train each model
- TEST_INSTANS: number of simulated seconds to test each model
- NUMBER_TEST_SIMULATIONS: number of test simulations for testing each model
- THREAD_POOL_SIZE: number of models to be trained simultaneously

Then, you can test the system for 1 task set by simply running the program:

```sh
cargo run --release
```
If you want to collect statistics for more task sets, use the bundled Python scripts.
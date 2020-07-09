#[macro_use]
extern crate log;

use mnist_classification::infrastructure::logging;
use mnist_classification::prelude::NeuralNetwork;


fn main() {
    logging::init();

    let _x = NeuralNetwork::from(vec![5, 4, 3, 2, 1]);
}

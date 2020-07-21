extern crate log;


use log::*;


use mnist_classification::infrastructure::logging;
use mnist_classification::infrastructure::mnist_loader::*;
use mnist_classification::prelude::NeuralNetwork;

fn main() {
    logging::init();
    let labels_path = "resources/t10k-labels-idx1-ubyte.gz";
    let images_path = "resources/t10k-images-idx3-ubyte.gz";

    let mnist_labels = MnistGzFileLoader::load_labels(labels_path).unwrap();
    let mnist_images = MnistGzFileLoader::load_images(images_path).unwrap();

    info!("{}", mnist_labels);
    info!("{}", mnist_images);

    let _x = NeuralNetwork::from(vec![5, 4, 3, 2, 1]);
}

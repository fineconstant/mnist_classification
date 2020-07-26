use log::*;

use mnist_classification::infrastructure::logging;
use mnist_classification::infrastructure::mnist_loader::dataset::MnistImage;
use mnist_classification::infrastructure::mnist_loader::*;
use mnist_classification::prelude::NeuralNetwork;

fn main() {
    logging::init();
    let images_path = "resources/train-images-idx3-ubyte.gz";
    let labels_path = "resources/train-labels-idx1-ubyte.gz";
    let test_images_path = "resources/t10k-images-idx3-ubyte.gz";
    let test_labels_path = "resources/t10k-labels-idx1-ubyte.gz";

    let mnist_images = MnistGzFileLoader::load_images(images_path).unwrap();
    let mnist_labels = MnistGzFileLoader::load_labels(labels_path).unwrap();
    info!("Training images: {}", mnist_images);
    info!("Training labels: {}", mnist_labels);

    let test_mnist_images = MnistGzFileLoader::load_images(test_images_path).unwrap();
    let test_mnist_labels = MnistGzFileLoader::load_labels(test_labels_path).unwrap();
    info!("Test images: {}", test_mnist_images);
    info!("Test labels: {}", test_mnist_labels);

    let dataset = &mut MnistImage::new(mnist_images, mnist_labels).unwrap();
    let test_dataset = &mut MnistImage::new(test_mnist_images, test_mnist_labels).unwrap();

    let mut network = NeuralNetwork::new(&[784, 100, 10]);
    network.stochastic_gradient_descend(100, 10, 3.0, dataset, test_dataset);
}

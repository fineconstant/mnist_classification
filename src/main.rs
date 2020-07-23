use log::*;

use mnist_classification::infrastructure::logging;
use mnist_classification::infrastructure::mnist_loader::*;
use mnist_classification::prelude::NeuralNetwork;
use mnist_classification::infrastructure::mnist_loader::dataset::MnistImage;

fn main() {
    logging::init();
    let images_path = "resources/t10k-images-idx3-ubyte.gz";
    let labels_path = "resources/t10k-labels-idx1-ubyte.gz";
    // let images_path = "resources/train-images-idx3-ubyte.gz";
    // let labels_path = "resources/train-labels-idx1-ubyte.gz";


    let mnist_images = MnistGzFileLoader::load_images(images_path).unwrap();
    let mnist_labels = MnistGzFileLoader::load_labels(labels_path).unwrap();

    info!("{}", mnist_images);
    info!("{}", mnist_labels);

    let dataset = MnistImage::new(mnist_images, mnist_labels).unwrap();


    let _x = NeuralNetwork::from(vec![784, 20, 10]);

    println!("{:?}", dataset.len())
}

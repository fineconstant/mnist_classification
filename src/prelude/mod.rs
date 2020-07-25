use std::borrow::BorrowMut;

use log::*;
use ndarray::prelude::*;
use ndarray_rand::rand::rngs::ThreadRng;
use ndarray_rand::rand::seq::SliceRandom;
use ndarray_rand::rand::thread_rng;

use arrays_random::StandardNormalDistribution;

use crate::algebra::Algebra;
use crate::infrastructure::logging;
use crate::infrastructure::mnist_loader::dataset::MnistImage;

mod arrays_random;

pub struct NeuralNetwork {
    layers_number: usize,
    layers_sizes: Vec<usize>,
    weights: Vec<Array2<f32>>,
    biases: Vec<Array1<f32>>,
    rng: ThreadRng,
}

impl NeuralNetwork {
    pub fn from(layers_sizes: Vec<usize>) -> NeuralNetwork {
        info!("Creating network with layers: {:?}", layers_sizes);

        let layers_number = layers_sizes.len();

        let tail = layers_sizes[1..].iter();
        let drop_last = layers_sizes[..layers_number - 1].iter();

        let weights = tail
            .zip(drop_last)
            // weights (neuron, weight)
            .map(|(x, y)| StandardNormalDistribution::from(&[*x, *y]))
            // weights (weight, neuron)
            // .map(|(x, y)| StandardNormalDistribution::from(&[*y, *x]))
            .map(|array| {
                array.into_dimensionality::<Ix2>()
                    .expect("Failed converting weights into 2D array")
            })
            .collect::<Vec<_>>();

        let biases = layers_sizes[1..]
            .iter()
            .map(|x| StandardNormalDistribution::from(&[*x]))
            .map(|array| {
                array.into_dimensionality::<Ix1>()
                    .expect("Failed biases weights into 1D array")
            })
            .collect::<Vec<_>>();

        NeuralNetwork {
            layers_number,
            layers_sizes,
            weights,
            biases,
            rng: thread_rng(),
        }
    }

    /// Input is an (m, 1) array.
    ///
    /// Output is an(n, 1) array where n is number of neurons is passed through,
    ///
    /// Uses ndarray dot function: https://docs.rs/ndarray/0.13.1/ndarray/struct.ArrayBase.html#method.dot
    fn feed_forward(self, input: Array1<f32>) -> Array1<f32> {
        self.weights
            .iter()
            .zip(self.biases.iter())
            .fold(input, |acc, (weights, biases)| {
                debug!("Acc: {:?}", acc);
                debug!("Weights:\n{:?}", weights);
                debug!("Biases: {:?}", biases);
                Algebra::sigmoid(&(weights.dot(&acc) + biases))
            })
    }

    /// Trains the neural network using mini-batch stochastic gradient descent.
    ///
    /// `epochs` - defines for how many epochs to train the network.
    ///
    /// `mini_batch_size` - defines size of a batch used when sampling.
    ///
    /// `eta` (η) - is the learning rate - the step size when performing gradient descent.
    ///
    /// `training_data` - is a list of tuples (x, y) representing the training inputs and
    /// corresponding desired outputs.
    ///
    /// If the optional argument `test_data` is supplied, then the program will evaluate
    /// the network after each epoch of training, and print out partial progress.
    /// This is useful for tracking progress, but slows things down substantially
    fn stochastic_gradient_descend(
        &mut self,
        epochs: u32,
        mini_batch_size: usize,
        eta: u32,
        training_data: &mut Vec<MnistImage>,
        test_data_option: Option<Vec<MnistImage>>,
    ) {
        let training_data_size = training_data.len();
        info!("Training data size: {}", training_data_size);
        info!("Training epochs: {}", epochs);

        for epoch in 0..epochs {
            training_data.shuffle(&mut self.rng);

            for mini_batch in training_data.chunks(mini_batch_size) {
                self.update_mini_batch(mini_batch, eta);
            }

            match &test_data_option {
                Some(test_data) => {
                    // todo: implement 4 - logging
                    // info!("Completed learning epoch: {}, {}", epoch,test_data.len());
                    self.evaluate(test_data)
                }
                None => {}
            };

            info!("Completed learning epoch: {}", epoch);
        }
    }

    // todo: logging
    fn update_mini_batch(&mut self, mini_batch: &[MnistImage], eta: u32) {
        let mut nabla_weights = self
            .weights
            .iter()
            .map(|w| Array2::zeros(w.dim()))
            .collect::<Vec<Array2<f32>>>();

        let mut nabla_biases = self
            .biases
            .iter()
            .map(|b| Array1::zeros(b.dim()))
            .collect::<Vec<Array1<f32>>>();

        for batch_item in mini_batch {
            let (delta_nabla_weights, delta_nabla_biases) = self.back_propagate(batch_item);
            nabla_weights = nabla_weights.iter().zip(delta_nabla_weights.iter())
                .map(|(nw, dnw)| nw + dnw).collect::<Vec<_>>();
            nabla_biases = nabla_biases.iter().zip(delta_nabla_biases.iter())
                .map(|(nb, dnb)| nb + dnb).collect::<Vec<_>>();
        }

        self.weights = self.weights.iter().zip(nabla_weights.iter())
            .map(|(w, nw)| w.sub(eta as f32 / mini_batch.len() as f32) * nw)
            .collect::<Vec<_>>();

        self.biases = self.biases.iter().zip(nabla_biases.iter())
            .map(|(b, nb)| b.sub(eta as f32 / mini_batch.len() as f32) * nb)
            .collect::<Vec<_>>();
    }

    fn back_propagate(&mut self, x: f32, y: f32) -> (Array2<f32>, Array1<f32>) {
        println!("x: {}, y: {}", x, y);
        (array![[1., 2.], [3., 4.]], array![1., 2., 3., 4.])
    }

    /// Evaluate the network and print out partial progress.
    ///
    /// Warning: slows down learning substantially!
    fn evaluate(&mut self, test_dataset: Vec<MnistImage>) {
        // test_dataset.iter().map(|test_data| {
        //     let output = self.feed_forward(test_data.data);
        //     // todo: find index of max element
        // });

        // for data in test_data {
        //   let output = self.feed_forward(data.data);
        //
        // };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wip() {
        let layers_sizes = vec![8, 5, 3];
        let mut network = NeuralNetwork::from(layers_sizes.clone());

        let input = array![-2., -1., 0., 1., 2., 3., 4., 5.];

        // network.stochastic_gradient_descend(
        //     10,
        //     30,
        //     1,
        //     &mut vec![TestData {
        //         data: array![-2., -1., 0., 1., 2., 3., 4., 5.],
        //         label: 1,
        //     }],
        //     Option::None,
        // )
    }

    #[test]
    fn initializes_neural_network() {
        let layers_sizes = vec![1, 2, 3, 4, 5];

        let actual = NeuralNetwork::from(layers_sizes.clone());

        let expected_layers_len = layers_sizes.len();
        assert_eq!(actual.layers_number, expected_layers_len);
        assert_eq!(actual.layers_sizes, layers_sizes);
        assert_eq!(actual.weights.len(), expected_layers_len - 1);
        assert_eq!(actual.biases.len(), expected_layers_len - 1)
    }

    #[test]
    fn does_not_initialise_biases_for_input_layer() {
        let layers_sizes = vec![1, 2, 3, 4, 5];

        let actual_biases = NeuralNetwork::from(layers_sizes.clone()).biases;

        for (bias_array, expected_len) in actual_biases.iter().zip(layers_sizes[1..].iter()) {
            assert_eq!(bias_array.len(), *expected_len)
        }
    }

    #[test]
    fn does_not_initialise_weights_for_input_layer() {
        let layers_sizes = vec![1, 2, 3, 4, 5];

        let actual_weights = NeuralNetwork::from(layers_sizes.clone()).weights;

        for (idx, (weights_array, expected_len)) in actual_weights
            .iter()
            .zip(layers_sizes[1..].iter())
            .enumerate()
        {
            // weights (neuron, weight)
            assert_eq!(weights_array.len_of(Axis(0)), *expected_len);
            assert_eq!(weights_array.len_of(Axis(1)), layers_sizes[idx]);

            // weights (weight, neuron)
            // assert_eq!(weights_array.len_of(Axis(0)), layers_sizes[idx]);
            // assert_eq!(weights_array.len_of(Axis(1)), *expected_len);
        }
    }

    #[test]
    fn outputs_value_for_each_neuron_in_last_layer() {
        let layers_sizes = vec![5, 3, 3];
        let network = NeuralNetwork::from(layers_sizes.clone());

        let input = array![-1., 0., 1., 2., 3.];
        let actual = network.feed_forward(input);

        assert_eq!(actual.len(), *layers_sizes.last().unwrap());
    }

    #[test]
    fn outputs_values_within_sigmoid_range() {
        let layers_sizes = vec![8, 5, 100];
        let network = NeuralNetwork::from(layers_sizes.clone());

        let input = array![-2., -1., 0., 1., 2., 3., 4., 5.];
        let actual = network.feed_forward(input);

        for actual_value in actual.into_raw_vec().iter() {
            assert!(*actual_value < 1.0);
            assert!(*actual_value > -1.0);
        }
    }
}

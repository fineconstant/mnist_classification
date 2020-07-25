use std::ops::Sub;

use log::*;
use ndarray::prelude::*;
use ndarray_rand::rand::rngs::ThreadRng;
use ndarray_rand::rand::seq::SliceRandom;
use ndarray_rand::rand::thread_rng;

use arrays_random::StandardNormalDistribution;

use crate::algebra::Algebra;
use crate::infrastructure::mnist_loader::dataset::MnistImage;
use ndarray_stats::QuantileExt;

mod arrays_random;

pub struct NeuralNetwork {
    layers_number: usize,
    _layers_sizes: Vec<usize>,
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
            .map(|(x, y)| StandardNormalDistribution::from(&[*x, *y]))
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
            _layers_sizes: layers_sizes,
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
    fn feed_forward(&mut self, input: &Array1<f32>) -> Array1<f32> {
        self.weights
            .iter()
            .zip(self.biases.iter())
            .fold(input.to_owned(), |acc, (weights, biases)| {
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
    pub fn stochastic_gradient_descend(
        &mut self,
        epochs: u32,
        mini_batch_size: usize,
        eta: f32,
        training_data: &mut Vec<MnistImage>,
        test_data_option: Option<&Vec<MnistImage>>,
    ) -> &mut NeuralNetwork {
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
                    let evaluation = self.evaluate(test_data);
                    info!("Evaluation: {}/{}", evaluation, test_data.len());
                }
                None => {
                    info!("Finished learning epoch: {}/{}", epoch + 1, epochs);
                }
            };
        }

        self
    }

    fn update_mini_batch(&mut self, mini_batch: &[MnistImage], eta: f32) {
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
            .map(|(w, nw)| w.sub(eta / mini_batch.len() as f32) * nw)
            .collect::<Vec<_>>();

        self.biases = self.biases.iter().zip(nabla_biases.iter())
            .map(|(b, nb)| b.sub(eta / mini_batch.len() as f32) * nb)
            .collect::<Vec<_>>();
    }

    fn back_propagate(&mut self, batch_item: &MnistImage) -> (Vec<Array2<f32>>, Vec<Array1<f32>>) {
        let mut nabla_weights = self.weights.iter()
            .map(|w| Array2::zeros(w.dim()))
            .collect::<Vec<Array2<f32>>>();

        let mut nabla_biases = self.biases.iter()
            .map(|b| Array1::zeros(b.dim()))
            .collect::<Vec<Array1<f32>>>();

        let mut activation = batch_item.image.to_owned();
        let mut activations = vec![activation.to_owned()];
        let mut zs = Vec::<Array1<f32>>::with_capacity(self.weights.len());

        for (weights, biases) in self.weights.iter().zip(self.biases.iter()) {
            let z = weights.dot(&activation) + biases;
            activation = Algebra::sigmoid(&z);
            activations.push(activation.to_owned());
            zs.push(z);
        }

        // backward pass
        let mut delta = {
            let output_activations = activations.last().unwrap();
            let last_z = zs.last().unwrap();
            self.cost_derivative(output_activations, &batch_item.label) * Algebra::sigmoid_prime(last_z)
        };

        let previous_activations = &activations[activations.len() - 2];
        let delta_trans = delta.to_owned().into_shape((delta.len(), 1)).unwrap();
        let output_activations_trans = previous_activations.to_owned().into_shape((1, previous_activations.len())).unwrap();
        *nabla_weights.last_mut().unwrap() = delta_trans.dot(&output_activations_trans);
        *nabla_biases.last_mut().unwrap() = delta.to_owned();

        for l in 2..self.layers_number {
            let z = &zs[zs.len() - l];

            delta = self.weights[self.weights.len() - l + 1]
                .to_owned().reversed_axes()
                .dot(&delta)
                .to_owned() * Algebra::sigmoid_prime(z);


            let selected_activations = &activations[activations.len() - l - 1];
            let selected_activations_trans = selected_activations.to_owned().into_shape((1, selected_activations.len())).unwrap();
            let delta_trans = delta.to_owned().into_shape((delta.len(), 1)).unwrap();

            let nw_len = nabla_weights.len();
            *nabla_weights.get_mut(nw_len - l).unwrap() = delta_trans.dot(&selected_activations_trans);

            let nb_len = nabla_biases.len();
            *nabla_biases.get_mut(nb_len - l).unwrap() = delta.to_owned();
        }

        (nabla_weights, nabla_biases)
    }

    /// Evaluate the network and print out partial progress.
    ///
    /// Warning: slows down learning substantially!
    fn evaluate(&mut self, test_dataset: &[MnistImage]) -> usize {
        test_dataset.iter().map(|data| {
            let output = self.feed_forward(&data.image)
                .argmax().unwrap();
            if output == data.raw_label { 1 } else { 0 }
        }).sum()
    }

    fn cost_derivative(&mut self, output_activation: &Array1<f32>, label: &Array1<f32>) -> Array1<f32> {
        output_activation.sub(label)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
            assert_eq!(bias_array.len(), *expected_len);
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
            assert_eq!(weights_array.len_of(Axis(0)), *expected_len);
            assert_eq!(weights_array.len_of(Axis(1)), layers_sizes[idx]);
        }
    }

    #[test]
    fn outputs_value_for_each_neuron_in_last_layer() {
        let layers_sizes = vec![5, 3, 3];
        let mut network = NeuralNetwork::from(layers_sizes.clone());

        let input = array![-1., 0., 1., 2., 3.];
        let actual = network.feed_forward(&input);

        assert_eq!(actual.len(), *layers_sizes.last().unwrap());
    }

    #[test]
    fn outputs_values_within_sigmoid_range() {
        let layers_sizes = vec![8, 5, 100];
        let mut network = NeuralNetwork::from(layers_sizes.clone());

        let input = array![-2., -1., 0., 1., 2., 3., 4., 5.];
        let actual = network.feed_forward(&input);

        for actual_value in actual.into_raw_vec().iter() {
            assert!(*actual_value < 1.0);
            assert!(*actual_value > -1.0);
        }
    }
}

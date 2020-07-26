use std::ops::Div;

use crate::algebra::Algebra;
use crate::infrastructure::mnist_loader::dataset::MnistImage;
use log::*;
use ndarray::prelude::*;
use ndarray_rand::rand::rngs::ThreadRng;
use ndarray_rand::rand::seq::SliceRandom;
use ndarray_rand::rand_distr::StandardNormal;
use ndarray_rand::RandomExt;
use ndarray_stats::QuantileExt;

pub struct NeuralNetwork {
    num_layers: usize,
    weights: Vec<Array2<f64>>,
    biases: Vec<Array2<f64>>,
    rng: ThreadRng,
}

impl NeuralNetwork {
    pub fn new(layers_sizes: &[usize]) -> NeuralNetwork {
        info!("Creating network with layers: {:?}", layers_sizes);

        let layers_len = layers_sizes.len();

        let tail = layers_sizes[1..].iter();
        let drop_last = layers_sizes[..layers_len - 1].iter();

        let weights = tail
            .zip(drop_last)
            .map(|(t, dl)| Array2::random((*dl, *t), StandardNormal))
            .collect::<Vec<_>>();

        let biases = layers_sizes[1..]
            .iter()
            .map(|x| Array2::random((1, *x), StandardNormal))
            .collect::<Vec<_>>();

        NeuralNetwork {
            num_layers: layers_len,
            weights,
            biases,
            rng: ThreadRng::default(),
        }
    }

    /// Evaluate the network and print out partial progress.
    ///
    /// Warning: slows down learning substantially!
    pub fn evaluate(&mut self, test_dataset: &[MnistImage]) -> usize {
        test_dataset
            .iter()
            .map(|data| {
                let output = self.feed_forward(&data.image).argmax().unwrap();
                if output.1 == data.classification {
                    1
                } else {
                    0
                }
            })
            .sum()
    }

    /// Input is an (1, n) array.
    /// Weights are (n, m) array.
    /// Biases are (1, n) array.
    ///
    /// Output is an (1, n) array where n is number of neurons is passed through,
    fn feed_forward(&self, input: &Array2<f64>) -> Array2<f64> {
        self.weights.iter().zip(self.biases.iter()).fold(
            input.to_owned(),
            |acc, (weights, biases)| {
                debug!("Acc: {:?}", acc);
                debug!("Weights:\n{:?}", weights);
                debug!("Biases: {:?}", biases);

                Algebra::sigmoid(&(acc.dot(weights) + biases))
            },
        )
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
        eta: f64,
        training_data: &mut [MnistImage],
        test_data: &[MnistImage],
    ) -> &mut NeuralNetwork {
        let training_data_size = training_data.len();
        info!("Training data size: {}", training_data_size);
        info!("Training epochs: {}", epochs);

        for epoch in 1..=epochs {
            training_data.shuffle(&mut self.rng);

            for mini_batch in training_data.chunks(mini_batch_size) {
                self.update_mini_batch(mini_batch, eta);
            }

            match test_data.is_empty() {
                true => {
                    info!("Finished learning epoch: {}/{}", epoch, epochs);
                }
                false => {
                    let evaluation = self.evaluate(test_data);
                    info!(
                        "Finished learning epoch: {}/{}, evaluation: {}/{}",
                        epoch,
                        epochs,
                        evaluation,
                        test_data.len()
                    );
                }
            }
        }

        self
    }

    fn update_mini_batch(&mut self, mini_batch: &[MnistImage], eta: f64) {
        let mini_batch_len = mini_batch.len() as f64;
        let bpr = self.back_propagate(mini_batch);

        for (w, nw) in self.weights.iter_mut().zip(bpr.nabla_weights.iter()) {
            *w -= &(nw * eta).div(mini_batch_len);
        }

        for (b, nb) in self.biases.iter_mut().zip(bpr.nabla_biases.iter()) {
            *b -= &(nb * eta).div(mini_batch_len);
        }
    }

    fn back_propagate(&mut self, mini_batch: &[MnistImage]) -> BackPropagationResult {
        let mini_batch_len = mini_batch.len();
        let weights_len = self.weights.len();
        let biases_len = self.biases.len();

        let mut nabla_weights = self
            .weights
            .iter()
            .map(|w| Array2::ones(w.dim()))
            .collect::<Vec<Array2<f64>>>();

        let mut nabla_biases = self
            .biases
            .iter()
            .map(|b| Array2::ones(b.dim()))
            .collect::<Vec<Array2<f64>>>();

        let mut zs = Vec::<Array2<f64>>::with_capacity(self.weights.len());
        let mut activation: Array2<f64> = Array2::zeros((mini_batch_len, mini_batch[0].image_size));
        let mut classifications: Array2<f64> = Array2::zeros((mini_batch_len, 10));
        let mut activations = vec![activation.to_owned()];

        for (i, d) in mini_batch.iter().enumerate() {
            activation
                .slice_mut(s![i, ..])
                .assign(&d.image.slice(s![0, ..]));
            classifications
                .slice_mut(s![i, ..])
                .assign(&d.label.slice(s![0, ..]));
        }

        for (weights, biases) in self.weights.iter().zip(self.biases.iter()) {
            let z = activation.dot(weights) + biases;
            activation = Algebra::sigmoid(&z);
            zs.push(z);
            activations.push(activation.to_owned());
        }

        let mut delta = self.cost_derivative(activations.last().unwrap(), &classifications)
            * Algebra::sigmoid_prime(zs.last().unwrap());

        let previous_activations = activations[activations.len() - 2].view().reversed_axes();
        nabla_weights[weights_len - 1] = previous_activations.dot(&delta);
        nabla_biases[biases_len - 1]
            .slice_mut(s![0, ..])
            .assign(&delta.sum_axis(Axis(0)));

        for l in 2..self.num_layers {
            let z = &zs[zs.len() - l];
            let sp = Algebra::sigmoid_prime(z);

            delta = delta.dot(&self.weights[weights_len - l + 1].t()) * sp;

            nabla_weights[weights_len - l] = activations[activations.len() - l - 1].t().dot(&delta);
            nabla_biases[biases_len - l]
                .slice_mut(s![0, ..])
                .assign(&delta.sum_axis(Axis(0)));
        }

        BackPropagationResult {
            nabla_weights,
            nabla_biases,
        }
    }

    fn cost_derivative(
        &mut self,
        output_activation: &Array2<f64>,
        label: &Array2<f64>,
    ) -> Array2<f64> {
        output_activation - label
    }
}

struct BackPropagationResult {
    nabla_weights: Vec<Array2<f64>>,
    nabla_biases: Vec<Array2<f64>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initializes_neural_network() {
        let layers_sizes = [1, 2, 3, 4, 5];

        let actual = NeuralNetwork::new(&layers_sizes);

        let expected_layers_len = layers_sizes.len();
        assert_eq!(actual.num_layers, expected_layers_len);
        assert_eq!(actual.weights.len(), expected_layers_len - 1);
        assert_eq!(actual.biases.len(), expected_layers_len - 1)
    }

    #[test]
    fn does_not_initialise_biases_for_input_layer() {
        let layers_sizes = [1, 2, 3, 4, 5];

        let actual_biases = NeuralNetwork::new(&layers_sizes).biases;

        for (bias_array, expected_len) in actual_biases.iter().zip(layers_sizes[1..].iter()) {
            assert_eq!(bias_array.len(), *expected_len);
            assert_eq!(bias_array.len(), *expected_len)
        }
    }

    #[test]
    fn does_not_initialise_weights_for_input_layer() {
        let layers_sizes = [1, 2, 3, 4, 5];

        let actual_weights = NeuralNetwork::new(&layers_sizes).weights;

        for (idx, (weights_array, expected_len)) in actual_weights
            .iter()
            .zip(layers_sizes[1..].iter())
            .enumerate()
        {
            assert_eq!(weights_array.len_of(Axis(0)), layers_sizes[idx]);
            assert_eq!(weights_array.len_of(Axis(1)), *expected_len);
        }
    }

    #[test]
    fn outputs_value_for_each_neuron_in_last_layer() {
        let layers_sizes = [5, 3, 2];
        let network = NeuralNetwork::new(&layers_sizes);

        let input = array![[-1., 0., 1., 2., 3.]];
        let actual = network.feed_forward(&input);

        assert_eq!(actual.len(), *layers_sizes.last().unwrap());
    }

    #[test]
    fn outputs_values_within_sigmoid_range() {
        let layers_sizes = [8, 5, 100];
        let mut network = NeuralNetwork::new(&layers_sizes);

        let input = array![[-2., -1., 0., 1., 2., 3., 4., 5.]];
        let actual = network.feed_forward(&input);

        for actual_value in actual.into_raw_vec().iter() {
            assert!(*actual_value < 1.0);
            assert!(*actual_value > -1.0);
        }
    }
}

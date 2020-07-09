mod arrays_random;

use arrays_random::StandardNormalDistribution;
use ndarray::prelude::*;

use crate::algebra::Algebra;
use crate::infrastructure::logging;

use log::*;

pub struct NeuralNetwork {
    layers_number: usize,
    layers_sizes: Vec<usize>,
    weights: Vec<Array2<f32>>,
    biases: Vec<Array1<f32>>,
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
            // .map(|(x, y)| StandardNormalDistribution::from(&[*x, *y]))
            // weights (weight, neuron)
            .map(|(x, y)| StandardNormalDistribution::from(&[*y, *x]))
            .map(|array| array.into_dimensionality::<Ix2>().unwrap())
            .collect::<Vec<_>>();

        let biases = layers_sizes[1..]
            .iter()
            .map(|x| StandardNormalDistribution::from(&[*x]))
            .map(|array| array.into_dimensionality::<Ix1>().unwrap())
            .collect::<Vec<_>>();

        NeuralNetwork {
            layers_number,
            layers_sizes,
            weights,
            biases,
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
            .fold(input, |interim, (weights, biases)| {
                info!("Interim: {:?}", interim);
                info!("Weights:\n{:?}", weights);
                info!("Biases: {:?}", biases);
                Algebra::sigmoid(interim.dot(weights) + biases)
            })
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
            // assert_eq!(bias_array.len_of(Axis(0)), *expected_len);
            // assert_eq!(bias_array.len_of(Axis(1)), layers_sizes[idx])

            // weights (weight, neuron)
            assert_eq!(weights_array.len_of(Axis(0)), layers_sizes[idx]);
            assert_eq!(weights_array.len_of(Axis(1)), *expected_len);
        }
    }

    #[test]
    fn outputs_value_for_each_neuron_in_last_layer() {
        let layers_sizes = vec![8, 5, 3];
        let network = NeuralNetwork::from(layers_sizes.clone());

        let input = array![-2., -1., 0., 1., 2., 3., 4., 5.];
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

mod arrays_random;

use arrays_random::StandardNormalDistribution;
use ndarray::prelude::*;

use crate::algebra::Algebra;
use crate::infrastructure::logging;

use log::*;

pub struct NeuralNetwork {
    layers_number: usize,
    layers_sizes: Vec<usize>,
    weights: Vec<ArrayD<f32>>,
    biases: Vec<ArrayD<f32>>,
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
            .into_iter()
            .collect::<Vec<ArrayD<f32>>>();

        let biases = layers_sizes[1..]
            .iter()
            .map(|x| StandardNormalDistribution::from(&[*x]))
            .collect::<Vec<_>>();

        NeuralNetwork {
            layers_number,
            layers_sizes,
            weights,
            biases,
        }
    }

    // todo: add tests
    // todo: refactor and simplify
    /// Input is an (m, 1) array.
    ///
    /// Output is an(n, 1) array where n is number of neurons is passed through,
    ///
    /// Uses ndarray dot function: https://docs.rs/ndarray/0.13.1/ndarray/struct.ArrayBase.html#method.dot
    fn feed_forward(self, input: ArrayD<f32>) -> ArrayD<f32> {
        self.weights
            .iter()
            .zip(self.biases.iter())
            .fold(input, |interim, (weights, biases)| {
                println!("INTERIM");
                println!("{:?}", interim);
                println!();
                println!("WEIGHTS");
                println!("{:?}", weights);
                println!();
                println!("BIASES");
                println!("{:?}", biases);
                println!();

                let a = interim.clone().into_dimensionality::<Ix1>().unwrap();
                let b = weights.clone().into_dimensionality::<Ix2>().unwrap();

                println!("RESULT");
                println!("{:?}", a.dot(&b).into_dyn());
                println!();
                println!("RESULT + BIASES");
                println!("{:?}", a.dot(&b).into_dyn() + biases);
                println!();
                println!();

                Algebra::sigmoid(a.dot(&b).into_dyn() + biases)
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
    fn ff() {
        let layers_sizes = vec![4, 3, 2];

        let network = NeuralNetwork::from(layers_sizes.clone());

        let y = network.feed_forward(array![1., 1., 1., 1.].into_dyn());

        println!("RESULT");
        println!("{:?}", y);
    }
}

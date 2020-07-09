use ndarray::prelude::*;
use ndarray::{Dim, IntoDimension};

pub struct Algebra;

impl Algebra {
    /// Calculates Sigmoid value for each element of an array.
    pub fn sigmoid<D>(array: Array<f32, D>) -> Array<f32, D>
    where
        D: Dimension,
    {
        array.mapv(|x| 1.0 / (1.0 + f32::exp(-x)))
    }

    /// Calculates exponential (e^x) for each element of an array.
    pub fn exp<D>(array: Array<f32, D>) -> Array<f32, D>
    where
        D: Dimension,
    {
        array.mapv(f32::exp)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calculates_sigmoid_for_1d_array() {
        let xs = array![-5., -4., -3., -2., -1., 0., 1., 2., 3., 4., 5., 6.];

        let actual = Algebra::sigmoid(xs);

        let expected = array![
            0.006692851,
            0.01798621,
            0.047425874,
            0.11920292,
            0.26894143,
            0.5,
            0.7310586,
            0.880797,
            0.95257413,
            0.98201376,
            0.9933072,
            0.9975274
        ];
        assert_eq!(actual, expected);
    }

    #[test]
    fn calculates_sigmoid_for_2d_array() {
        let xs = array![[-5., -4., -3., -2., -1., 0.], [1., 2., 3., 4., 5., 6.]];

        let actual = Algebra::sigmoid(xs);

        let expected = array![
            [
                0.006692851,
                0.01798621,
                0.047425874,
                0.11920292,
                0.26894143,
                0.5
            ],
            [0.7310586, 0.880797, 0.95257413, 0.98201376, 0.9933072, 0.9975274]
        ];
        assert_eq!(actual, expected);
    }

    #[test]
    fn calculates_exp_for_1d_array() {
        let xs = array![-5., -4., -3., -2., -1., 0., 1., 2., 3., 4., 5., 6.];

        let actual = Algebra::exp(xs);

        let expected = array![
            0.006737947,
            0.01831564,
            0.049787067,
            0.13533528,
            0.36787945,
            1.0,
            2.7182817,
            7.389056,
            20.085537,
            54.59815,
            148.41316,
            403.4288
        ];
        assert_eq!(actual, expected);
    }

    #[test]
    fn calculates_exp_for_2d_array() {
        let xs = array![[-5., -4., -3., -2., -1., 0.], [1., 2., 3., 4., 5., 6.]];

        let actual = Algebra::exp(xs);

        let expected = array![
            [
                0.006737947,
                0.01831564,
                0.049787067,
                0.13533528,
                0.36787945,
                1.0,
            ],
            [2.7182817, 7.389056, 20.085537, 54.59815, 148.41316, 403.4288]
        ];
        assert_eq!(actual, expected);
    }
}

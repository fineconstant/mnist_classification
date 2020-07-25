use ndarray::prelude::*;

pub struct Algebra;

impl Algebra {
    /// Calculates Sigmoid value for each element of an array.
    pub fn sigmoid<D>(array: &Array<f64, D>) -> Array<f64, D>
    where
        D: Dimension,
    {
        array.mapv(|x| 1.0 / (1.0 + f64::exp(-x)))
    }

    /// Calculates first derivative Sigmoid value for each element of an array.
    pub fn sigmoid_prime<D>(array: &Array<f64, D>) -> Array<f64, D>
    where
        D: Dimension,
    {
        Algebra::sigmoid(array).mapv(|x| x * (1.0 - x))
    }

    /// Calculates exponential (e^x) for each element of an array.
    pub fn exp<D>(array: &Array<f64, D>) -> Array<f64, D>
    where
        D: Dimension,
    {
        array.mapv(f64::exp)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calculates_sigmoid_for_1d_array() {
        let xs = array![-5., -4., -3., -2., -1., 0., 1., 2., 3., 4., 5., 6.];

        let actual = Algebra::sigmoid(&xs);

        let expected = array![
            0.0066928509242848554,
            0.01798620996209156,
            0.04742587317756678,
            0.11920292202211755,
            0.2689414213699951,
            0.5,
            0.7310585786300049,
            0.8807970779778823,
            0.9525741268224334,
            0.9820137900379085,
            0.9933071490757153,
            0.9975273768433653
        ];
        assert_eq!(actual, expected);
    }

    #[test]
    fn calculates_sigmoid_for_2d_array() {
        let xs = array![[-5., -4., -3., -2., -1., 0.], [1., 2., 3., 4., 5., 6.]];

        let actual = Algebra::sigmoid(&xs);

        let expected = array![
            [
                0.0066928509242848554,
                0.01798620996209156,
                0.04742587317756678,
                0.11920292202211755,
                0.2689414213699951,
                0.5
            ],
            [
                0.7310585786300049,
                0.8807970779778823,
                0.9525741268224334,
                0.9820137900379085,
                0.9933071490757153,
                0.9975273768433653
            ]
        ];
        assert_eq!(actual, expected);
    }

    #[test]
    fn calculates_sigmoid_prime_for_1d_array() {
        let xs = array![-5., -4., -3., -2., -1., 0., 1., 2., 3., 4., 5., 6.];

        let actual = Algebra::sigmoid_prime(&xs);

        let expected = array![
            0.006648056670790155,
            0.017662706213291118,
            0.04517665973091214,
            0.1049935854035065,
            0.19661193324148185,
            0.25,
            0.19661193324148185,
            0.10499358540350662,
            0.045176659730912,
            0.017662706213291107,
            0.006648056670790033,
            0.002466509291359931
        ];
        assert_eq!(actual, expected);
    }

    #[test]
    fn calculates_sigmoid_prime_for_2d_array() {
        let xs = array![[-5., -4., -3., -2., -1., 0.], [1., 2., 3., 4., 5., 6.]];

        let actual = Algebra::sigmoid_prime(&xs);

        let expected = array![
            [
                0.006648056670790155,
                0.017662706213291118,
                0.04517665973091214,
                0.1049935854035065,
                0.19661193324148185,
                0.25
            ],
            [
                0.19661193324148185,
                0.10499358540350662,
                0.045176659730912,
                0.017662706213291107,
                0.006648056670790033,
                0.002466509291359931
            ]
        ];
        assert_eq!(actual, expected);
    }

    #[test]
    fn calculates_exp_for_1d_array() {
        let xs = array![-5., -4., -3., -2., -1., 0., 1., 2., 3., 4., 5., 6.];

        let actual = Algebra::exp(&xs);

        let expected = array![
            0.006737946999085467,
            0.01831563888873418,
            0.049787068367863944,
            0.1353352832366127,
            0.36787944117144233,
            1.0,
            2.718281828459045,
            7.38905609893065,
            20.085536923187668,
            54.598150033144236,
            148.4131591025766,
            403.4287934927351
        ];
        assert_eq!(actual, expected);
    }

    #[test]
    fn calculates_exp_for_2d_array() {
        let xs = array![[-5., -4., -3., -2., -1., 0.], [1., 2., 3., 4., 5., 6.]];

        let actual = Algebra::exp(&xs);

        let expected = array![
            [
                0.006737946999085467,
                0.01831563888873418,
                0.049787068367863944,
                0.1353352832366127,
                0.36787944117144233,
                1.0
            ],
            [
                2.718281828459045,
                7.38905609893065,
                20.085536923187668,
                54.598150033144236,
                148.4131591025766,
                403.4287934927351
            ]
        ];
        assert_eq!(actual, expected);
    }
}

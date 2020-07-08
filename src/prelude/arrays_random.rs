use ndarray::prelude::*;
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::{RandomExt, F32};

pub struct StandardNormalDistribution;

impl StandardNormalDistribution {
    pub fn from(shape: &[usize]) -> ArrayD<f32> {
        ArrayD::random(IxDyn(shape), F32(Normal::new(0.0, 1.0).unwrap()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creates_an_array_of_desired_shape() {
        let dimensions = [2, 3, 4];

        let actual = StandardNormalDistribution::from(&dimensions);

        assert_eq!(actual.ndim(), dimensions.len());
        assert_eq!(actual.shape(), dimensions);
    }
}

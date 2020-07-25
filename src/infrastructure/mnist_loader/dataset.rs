use std::ops::Div;

use error_chain::*;
use ndarray::{Array1, Array2};

use crate::infrastructure::mnist_loader::error::{Error, ErrorKind};
use crate::infrastructure::mnist_loader::raw::images::MnistRawImages;
use crate::infrastructure::mnist_loader::raw::labels::MnistRawLabels;
use std::fmt::{Display, Formatter};

pub struct MnistImage {
    pub classification: usize,
    pub label: Array1<f64>,
    pub image: Array2<f64>,
}

impl MnistImage {
    pub fn new(
        raw_images: MnistRawImages,
        raw_labels: MnistRawLabels,
    ) -> Result<Vec<MnistImage>, Error> {
        if raw_images.number_of_images != raw_labels.number_of_labels {
            bail!(ErrorKind::IncompatibleDatasets(
                raw_labels.number_of_labels,
                raw_images.number_of_images
            ));
        }

        let dataset = raw_images
            .images
            .iter()
            .zip(raw_labels.labels.iter())
            .map(|(raw_image, raw_label)| {
                let label = MnistImage::vectorize(*raw_label);
                let image = Array2::from_shape_vec((1, raw_image.len()), raw_image.to_owned())
                    .unwrap()
                    .mapv(|v| v as f64)
                    .div(u8::MAX as f64);

                MnistImage {
                    classification: *raw_label as usize,
                    label,
                    image,
                }
            })
            .collect::<Vec<_>>();

        Ok(dataset)
    }

    pub fn from(raw_label: u8, image: Array2<f64>) -> Vec<MnistImage> {
        vec![MnistImage {
            classification: raw_label as usize,
            label: MnistImage::vectorize(raw_label),
            image,
        }]
    }

    /// Returns a 10-dimensional unit vector with a 1.0 in the x-th position and zeroes elsewhere.
    /// This is used to convert a digit into a corresponding desired output from the neural network.
    fn vectorize(x: u8) -> Array1<f64> {
        let mut arr = Array1::<f64>::zeros(10);
        arr[x as usize] = 1.0f64;

        arr
    }
}

impl Display for MnistImage {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "MnistImage(label size: {}, image size: {})",
            self.label.len(),
            self.image.len()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vectorizes_from_zero_to_nine() {
        for x in 0u8..10 {
            let actual_vector = MnistImage::vectorize(x);

            let actual = actual_vector[x as usize];
            let expected = 1.0f64;
            assert_eq!(actual_vector.len(), 10);
            assert_ulps_eq!(actual, expected);
        }
    }

    #[test]
    fn zero_on_other_places() {
        for x in 0..10 {
            let actual_vector = MnistImage::vectorize(x);
            assert_eq!(actual_vector.len(), 10);

            for i in 0..10 {
                if x != i {
                    let actual = actual_vector[i as usize];
                    let expected = 0.0f64;
                    assert_ulps_eq!(actual, expected);
                }
            }
        }
    }
}

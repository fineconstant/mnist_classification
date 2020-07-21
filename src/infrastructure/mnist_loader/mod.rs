use std::io::Error;
use std::io::prelude::*;

use log::*;

use crate::infrastructure::mnist_loader::file_reader::FileReader;
use crate::infrastructure::mnist_loader::file_reader::gz_file_reader::GZFileReader;
use crate::infrastructure::mnist_loader::mnist_errors::MnistFileFormatError;
use crate::infrastructure::mnist_loader::mnist_images::MnistImages;
use crate::infrastructure::mnist_loader::mnist_labels::MnistLabels;
use ndarray::Array1;

mod mnist_labels;
mod mnist_images;
mod file_reader;
mod mnist_errors;

const LABELS_MAGIC_NUMBER: u32 = 2049;
const IMAGES_MAGIC_NUMBER: u32 = 2051;

pub trait MnistFileLoader {
    fn load_labels(path: &str) -> Result<MnistLabels, Error>;
    fn load_images(path: &str) -> Result<MnistImages, Error>;
}

pub struct MnistGzFileLoader;

impl MnistGzFileLoader {
    /// Returns a 10-dimensional unit vector with a 1.0 in the x-th position and zeroes elsewhere.
    /// This is used to convert a digit into a corresponding desired output from the neural network.
    pub fn vectorize(x: u8) -> Array1<f32> {
        let mut arr = Array1::from(vec![0.0f32; 10]);
        arr[x as usize] = 1.0f32;

        arr
    }
}

// todo: tests
impl MnistFileLoader for MnistGzFileLoader {
    fn load_labels(path: &str) -> Result<MnistLabels, Error> {
        info!("Loading labels");
        let mnist_file = &mut GZFileReader::read(path)?;

        let mut buffer_32 = [0; 4];
        mnist_file.read_exact(&mut buffer_32)?;
        let magic_number = u32::from_be_bytes(buffer_32);

        match magic_number {
            LABELS_MAGIC_NUMBER => Ok(()),
            IMAGES_MAGIC_NUMBER => Err(MnistFileFormatError::images_instead_of_labels()),
            _ => Err(MnistFileFormatError::magic_number_error())
        }.unwrap();

        mnist_file.read_exact(&mut buffer_32)?;
        let number_of_labels = u32::from_be_bytes(buffer_32);
        info!("Number of labels: {}", number_of_labels);

        let mut labels: Vec<u8> = Vec::with_capacity(number_of_labels as usize);
        mnist_file.read_to_end(&mut labels)?;

        Ok(MnistLabels::new(number_of_labels, labels))
    }

    fn load_images(path: &str) -> Result<MnistImages, std::io::Error> {
        info!("Loading images");
        let mnist_file = &mut GZFileReader::read(path)?;

        let mut buffer_32 = [0; 4];
        mnist_file.read_exact(&mut buffer_32)?;
        let magic_number = u32::from_be_bytes(buffer_32);

        match magic_number {
            LABELS_MAGIC_NUMBER => Err(MnistFileFormatError::labels_instead_of_images()),
            IMAGES_MAGIC_NUMBER => Ok(()),
            _ => Err(MnistFileFormatError::magic_number_error())
        }.unwrap();

        mnist_file.read_exact(&mut buffer_32)?;
        let number_of_images = u32::from_be_bytes(buffer_32);

        mnist_file.read_exact(&mut buffer_32)?;
        let number_of_rows = u32::from_be_bytes(buffer_32);

        mnist_file.read_exact(&mut buffer_32)?;
        let number_of_columns = u32::from_be_bytes(buffer_32);
        let pixels_per_image = number_of_rows * number_of_columns;

        info!("Number of images: {}", number_of_images);
        info!("Number of rows: {}", number_of_rows);
        info!("Number of columns: {}", number_of_columns);
        info!("Pixels per image: {}", pixels_per_image);

        let images = (0..number_of_images)
            .fold(Vec::new(), |mut acc: Vec<Vec<u8>>, _| {
                let mut image_buffer: Vec<u8> = Vec::with_capacity(pixels_per_image as usize);
                mnist_file.take(pixels_per_image as u64)
                    .read_to_end(&mut image_buffer).unwrap();
                acc.push(image_buffer);
                acc
            });

        Ok(MnistImages::new(number_of_images, images))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vectorizes_from_zero_to_nine() {
        for x in 0u8..10 {
            let actual_vector = MnistGzFileLoader::vectorize(x);

            let actual = actual_vector[x as usize];

            let expected = 1.0f32;
            assert_eq!(actual_vector.len(), 10);
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn zero_on_other_places() {
        for x in 0..10 {
            let actual_vector = MnistGzFileLoader::vectorize(x);
            assert_eq!(actual_vector.len(), 10);

            for i in 0..10 {
                if x != i {
                    let actual = actual_vector[i as usize];
                    let expected = 0.0f32;
                    assert_eq!(actual, expected);
                }
            }
        }
    }
}

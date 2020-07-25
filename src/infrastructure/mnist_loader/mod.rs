use std::fs::File;
use std::io::prelude::*;

use error_chain::*;
use flate2::read::GzDecoder;
use log::*;

use crate::infrastructure::mnist_loader::error::{Error, ErrorKind};
use crate::infrastructure::mnist_loader::file_reader::gz_file_reader::GZFileReader;
use crate::infrastructure::mnist_loader::file_reader::FileReader;
use crate::infrastructure::mnist_loader::raw::images::MnistRawImages;
use crate::infrastructure::mnist_loader::raw::labels::MnistRawLabels;

pub mod dataset;
mod error;
mod file_reader;
mod raw;

const LABELS_MAGIC_NUMBER: u32 = 2049;
const IMAGES_MAGIC_NUMBER: u32 = 2051;

pub trait MnistFileLoader {
    fn load_labels(path: &str) -> Result<MnistRawLabels, Error>;
    fn load_images(path: &str) -> Result<MnistRawImages, Error>;
}

pub struct MnistGzFileLoader;

impl MnistGzFileLoader {
    fn read_magic_number(mnist_file: &mut GzDecoder<File>) -> Result<u32, Error> {
        let mut buffer_32 = [0; 4];
        mnist_file.read_exact(&mut buffer_32)?;
        Ok(u32::from_be_bytes(buffer_32))
    }
}

impl MnistFileLoader for MnistGzFileLoader {
    fn load_labels(path: &str) -> Result<MnistRawLabels, Error> {
        info!("Loading labels");
        let mnist_file = &mut GZFileReader::read(path)?;
        let magic_number = MnistGzFileLoader::read_magic_number(mnist_file)?;

        let _ = match magic_number {
            LABELS_MAGIC_NUMBER => Result::<(), Error>::Ok(()),
            IMAGES_MAGIC_NUMBER => bail!(ErrorKind::ImagesInsteadOfLabelsMagicNumber),
            _ => bail!(ErrorKind::InvalidMagicNumber(magic_number)),
        };

        let mut buffer_32 = [0; 4];
        mnist_file.read_exact(&mut buffer_32)?;
        let number_of_labels = u32::from_be_bytes(buffer_32);
        info!("Number of labels: {}", number_of_labels);

        let mut labels: Vec<u8> = Vec::with_capacity(number_of_labels as usize);
        mnist_file.read_to_end(&mut labels)?;

        Ok(MnistRawLabels::new(number_of_labels, labels))
    }

    fn load_images(path: &str) -> Result<MnistRawImages, Error> {
        info!("Loading images");
        let mnist_file = &mut GZFileReader::read(path)?;
        let magic_number = MnistGzFileLoader::read_magic_number(mnist_file)?;

        let _ = match magic_number {
            IMAGES_MAGIC_NUMBER => Result::<(), Error>::Ok(()),
            LABELS_MAGIC_NUMBER => bail!(ErrorKind::LabelsInsteadOfImagesMagicNumber),
            _ => bail!(ErrorKind::InvalidMagicNumber(magic_number)),
        };

        let mut buffer_32 = [0; 4];
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

        let images = (0..number_of_images).fold(Vec::new(), |mut acc: Vec<Vec<u8>>, _| {
            let mut image_buffer: Vec<u8> = Vec::with_capacity(pixels_per_image as usize);
            mnist_file
                .take(pixels_per_image as u64)
                .read_to_end(&mut image_buffer)
                .unwrap();
            acc.push(image_buffer);
            acc
        });

        Ok(MnistRawImages::new(number_of_images, images))
    }
}

#[cfg(test)]
mod tests {
    use flate2::{write, Compression};
    use tempfile::NamedTempFile;

    use super::*;

    #[test]
    fn loads_labels_from_valid_file() {
        let file = NamedTempFile::new().unwrap();
        let path = file.path().to_str().unwrap();
        let mut encoder = write::GzEncoder::new(&file, Compression::fast());

        let magic_number = LABELS_MAGIC_NUMBER.to_be_bytes();
        let labels = (0u8..20).collect::<Vec<_>>();
        let _ = encoder.write(&magic_number);
        let _ = encoder.write(&(labels.len() as u32).to_be_bytes());
        let _ = encoder.write(labels.as_slice());
        encoder.finish().unwrap();

        let actual = MnistGzFileLoader::load_labels(path).unwrap();

        let expected = MnistRawLabels::new(labels.len() as u32, labels);
        assert_eq!(actual, expected);
    }

    #[test]
    fn error_when_labels_file_not_found() {
        let actual = MnistGzFileLoader::load_labels("foo");
        let expected = "The system cannot find the file specified. (os error 2)";
        assert_eq!(actual.unwrap_err().to_string(), expected);
    }

    #[test]
    fn error_when_images_file_not_found() {
        let actual = MnistGzFileLoader::load_images("foo");
        let expected = "The system cannot find the file specified. (os error 2)";
        assert_eq!(actual.unwrap_err().to_string(), expected);
    }

    #[test]
    fn error_when_empty_labels_file() {
        let file = NamedTempFile::new().unwrap();
        let path = file.path().to_str().unwrap();

        let actual = MnistGzFileLoader::load_labels(&path);

        let expected = "failed to fill whole buffer";
        assert_eq!(actual.unwrap_err().to_string(), expected);
    }

    #[test]
    fn error_when_empty_images_file() {
        let file = NamedTempFile::new().unwrap();
        let path = file.path().to_str().unwrap();

        let actual = MnistGzFileLoader::load_images(&path);

        let expected = "failed to fill whole buffer";
        assert_eq!(actual.unwrap_err().to_string(), expected);
    }

    #[test]
    fn error_when_images_instead_of_labels_header() {
        let file = NamedTempFile::new().unwrap();
        let path = file.path().to_str().unwrap();
        let mut encoder = write::GzEncoder::new(&file, Compression::fast());

        let magic_number = IMAGES_MAGIC_NUMBER.to_be_bytes();
        let _ = encoder.write(&magic_number);
        encoder.finish().unwrap();

        let actual = MnistGzFileLoader::load_labels(&path);

        let expected = "Expected 2049 but got 2051";
        assert_eq!(actual.unwrap_err().to_string(), expected);
    }

    #[test]
    fn error_when_labels_instead_of_images_header() {
        let file = NamedTempFile::new().unwrap();
        let path = file.path().to_str().unwrap();
        let mut encoder = write::GzEncoder::new(&file, Compression::fast());

        let magic_number = LABELS_MAGIC_NUMBER.to_be_bytes();
        let _ = encoder.write(&magic_number);
        encoder.finish().unwrap();

        let actual = MnistGzFileLoader::load_images(&path);

        let expected = "Expected 2051 but got 2049";
        assert_eq!(actual.unwrap_err().to_string(), expected);
    }

    #[test]
    fn error_when_could_not_read_images_header() {
        let file = NamedTempFile::new().unwrap();
        let path = file.path().to_str().unwrap();
        let mut encoder = write::GzEncoder::new(&file, Compression::fast());

        let magic_number = 128u32.to_be_bytes();
        let _ = encoder.write(&magic_number);
        encoder.finish().unwrap();

        let actual = MnistGzFileLoader::load_images(&path);

        let expected = "Expected u32 2049 or 2051 but got: '128'";
        assert_eq!(actual.unwrap_err().to_string(), expected);
    }
}

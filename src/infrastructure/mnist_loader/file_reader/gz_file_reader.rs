use std::fs::File;
use std::io::Error;
use std::path::Path;

use flate2::read::GzDecoder;
use log::*;

use crate::infrastructure::mnist_loader::file_reader::FileReader;

pub struct GZFileReader;

impl FileReader<GzDecoder<File>> for GZFileReader {
    fn read(path: &str) -> Result<GzDecoder<File>, Error> {
        info!("Attempting to read .gz file: {}", path);
        let buf_path = Path::new(path);
        info!("Reading .gz file: {:?}", buf_path.canonicalize()?);
        let mnist_archive = File::open(buf_path)?;
        Ok(GzDecoder::new(mnist_archive))
    }
}

#[cfg(test)]
mod tests {
    use std::io::{ErrorKind, Read, Write};

    use flate2::Compression;
    use flate2::write;
    use tempfile::*;

    use super::*;

    #[test]
    fn opens_file_that_exists() {
        let file = NamedTempFile::new().unwrap();
        let path = file.path().to_str().unwrap();

        let actual = GZFileReader::read(path);

        assert!(actual.is_ok());
    }

    #[test]
    fn error_when_file_does_not_exist() {
        let file_reader = GZFileReader::read("foo");

        assert!(file_reader.is_err());
        let actual = file_reader.map_err(|e| e.kind());

        let expected = ErrorKind::NotFound;
        assert_eq!(actual.unwrap_err(), expected);
    }

    #[test]
    fn reads_file_contents() {
        let file = &NamedTempFile::new().unwrap();
        let path = file.path().to_str().unwrap();

        let test_data = "test";
        let mut encoder = write::GzEncoder::new(file, Compression::fast());
        encoder.write_all(test_data.as_bytes()).unwrap();
        encoder.finish().unwrap();

        let reader = &mut GZFileReader::read(path).unwrap();
        let mut actual = String::new();
        reader.read_to_string(&mut actual).unwrap();

        assert_eq!(actual, test_data)
    }
}

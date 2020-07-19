use std::fs::File;
use std::io::{BufReader, Error};
use std::path::Path;

use log::*;

use crate::infrastructure::mnist_loader::file_reader::FileReader;

struct FlatFileReader;

impl FileReader<BufReader<File>> for FlatFileReader {
    fn read(path: &str) -> Result<BufReader<File>, Error> {
        info!("Attempting to read flat file: {}", path);
        let buf_path = Path::new(path);
        info!("Reading flat file: {:?}", buf_path.canonicalize()?);
        let file = File::open(buf_path)?;
        Ok(BufReader::new(file))
    }
}


#[cfg(test)]
mod tests {
    use std::io::ErrorKind;

    use tempfile::*;

    use super::*;

    #[test]
    fn reads_file_that_exists() {
        let file = NamedTempFile::new().unwrap();
        let path = file.path().to_str().unwrap();

        let actual = FlatFileReader::read(path);

        assert!(actual.is_ok());
    }

    #[test]
    fn error_when_file_does_not_exist() {
        let file_reader = FlatFileReader::read("foo");

        assert!(file_reader.is_err());
        let actual = file_reader.map_err(|e| e.kind());

        let expected = ErrorKind::NotFound;
        assert_eq!(actual.unwrap_err(), expected);
    }
}

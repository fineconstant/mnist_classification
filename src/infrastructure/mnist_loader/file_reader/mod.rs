use std::io::Read;

pub mod gz_file_reader;
pub mod flat_file_reader;

pub trait FileReader<T> where T: Read {
    fn read(path: &str) -> Result<T, std::io::Error>;
}

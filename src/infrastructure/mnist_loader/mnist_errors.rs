#[derive(Debug, Clone)]
pub struct MnistFileFormatError {
    details: String
}

impl MnistFileFormatError {
    pub fn labels_instead_of_images() -> MnistFileFormatError {
        MnistFileFormatError { details: "File contains labels instead of images".to_string() }
    }

    pub fn images_instead_of_labels() -> MnistFileFormatError {
        MnistFileFormatError { details: "File contains images instead of labels".to_string() }
    }

    pub fn magic_number_error() -> MnistFileFormatError {
        MnistFileFormatError { details: "Expected u32 integer containing 2049 or 2051 Magic Number".to_string() }
    }
}

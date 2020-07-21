use std::fmt::{Display, Formatter};

#[derive(Eq, PartialEq, Debug)]
pub struct MnistImages {
    number_of_images: u32,
    images: Vec<Vec<u8>>,
}

impl MnistImages {
    pub fn new(number_of_images: u32, images: Vec<Vec<u8>>) -> MnistImages {
        MnistImages {
            number_of_images,
            images,
        }
    }
}

impl Display for MnistImages {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "MnistImages(number_of_images: {})", self.number_of_images)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creates_new_instance() {
        let number_of_images = 100;
        let images = vec![vec![0u8; 100]; 100];

        let actual = MnistImages::new(number_of_images, images.clone());

        assert_eq!(actual.number_of_images, number_of_images);
        assert_eq!(actual.images, images);
    }
}

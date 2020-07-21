use std::fmt::{Display, Formatter};

#[derive(Eq, PartialEq, Debug)]
pub struct MnistLabels {
    number_of_labels: u32,
    labels: Vec<u8>,
}

impl MnistLabels {
    pub fn new(number_of_labels: u32, labels: Vec<u8>) -> MnistLabels {
        MnistLabels {
            number_of_labels,
            labels,
        }
    }
}

impl Display for MnistLabels {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "MnistLabels(number_of_labels: {})", self.number_of_labels)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creates_new_instance() {
        let number_of_labels = 100;
        let labels = vec![0u8; 100];

        let actual = MnistLabels::new(number_of_labels, labels.clone());

        assert_eq!(actual.number_of_labels, number_of_labels);
        assert_eq!(actual.labels, labels);
    }
}

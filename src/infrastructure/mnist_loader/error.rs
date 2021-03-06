use error_chain::*;

error_chain! {
    foreign_links {
        IoError(::std::io::Error);
    }

    errors {
        ImagesInsteadOfLabelsMagicNumber {
            description("Expected 2049 but got 2051")
            display("Expected 2049 but got 2051")
        }

        LabelsInsteadOfImagesMagicNumber {
            description("Expected 2051 but got 2049")
            display("Expected 2051 but got 2049")
        }

        InvalidMagicNumber(cause: u32) {
            description("Invalid MNIST Magic Number")
            display("Expected u32 2049 or 2051 but got: '{}'", cause)
        }

        IncompatibleDatasets(num_labels:u32, num_images:u32) {
            description("Did not get the same number of labels and images")
            display("Expected the same number of labels and images but got: '{}' labels, {} images", num_labels, num_images)
        }
    }
}

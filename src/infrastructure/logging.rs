pub fn init() {
    env_logger::init()
}

pub fn init_tests() {
    env_logger::builder().is_test(true).try_init();
}
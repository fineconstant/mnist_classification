#![recursion_limit = "1024"]

extern crate error_chain;
extern crate log;

#[cfg(test)]
#[macro_use]
extern crate approx;

pub mod infrastructure;
pub mod prelude;

mod algebra;

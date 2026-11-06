//! Link the frozen runtime alone for process memory measurements.
#![forbid(unsafe_code)]
use baseline as runtime;
include!("../memory_driver.rs");

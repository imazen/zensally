//! Link the current runtime alone for process memory measurements.
#![forbid(unsafe_code)]
use candidate as runtime;
include!("../memory_driver.rs");

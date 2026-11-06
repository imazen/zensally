#![forbid(unsafe_code)]
#[path = "../src/workers.rs"]
mod workers;
use std::hint::black_box;
use zenbench::prelude::*;

fn bench(suite: &mut Suite) {
    let cpus = workers::cpus();
    let mut workers = workers::Workers::new(&cpus);
    suite.group("w24/batch32", |g| {
        g.throughput(Throughput::Elements(workers::BATCH as u64));
        g.bench(format!("image-workers-{}", cpus.len()), move |b| {
            b.iter(|| black_box(workers.batch()));
        });
    });
}
zenbench::main!(bench);

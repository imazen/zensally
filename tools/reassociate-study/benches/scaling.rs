#![forbid(unsafe_code)]
#[path = "../src/workers.rs"]
mod workers;
use std::hint::black_box;
use zenbench::prelude::*;

fn bench(suite: &mut Suite) {
    let cpus = workers::cpus();
    // Build the oracle once and clear its unused pool before either arm exists.
    let fixture = workers::Fixture::new();
    let mut baseline = workers::Workers::with_fixture(&cpus, workers::Arm::Baseline, &fixture);
    let mut candidate = workers::Workers::with_fixture(&cpus, workers::Arm::Candidate, &fixture);
    suite.group("w24/batch32", |g| {
        g.throughput(Throughput::Elements(workers::BATCH as u64));
        g.bench(format!("baseline-workers-{}", cpus.len()), move |b| {
            b.iter(|| black_box(baseline.batch()));
        });
        g.bench(format!("candidate-workers-{}", cpus.len()), move |b| {
            b.iter(|| black_box(candidate.batch()));
        });
    });
}
zenbench::main!(bench);

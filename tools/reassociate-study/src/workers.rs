//! Persistent image workers. Construction, first-touch and bit validation happen
//! before measurement. Each batch processes the same 32 inputs at every width.
use std::{
    hint::black_box,
    sync::{Arc, mpsc},
    thread::{self, JoinHandle},
};

pub(super) const BATCH: usize = 32;

pub(super) fn cpus() -> Vec<usize> {
    let cpus: Vec<usize> = std::env::var("W24_CPU_SET")
        .expect("W24_CPU_SET is required")
        .split(',')
        .map(|s| s.parse().expect("CPU IDs must be comma-separated integers"))
        .collect();
    assert!(!cpus.is_empty() && cpus.len() <= BATCH);
    cpus
}

struct Worker {
    request: mpsc::Sender<bool>,
    done: mpsc::Receiver<usize>,
    handle: Option<JoinHandle<()>>,
}

pub(super) struct Workers {
    workers: Vec<Worker>,
}

#[derive(Clone, Copy, Debug)]
pub(super) enum Arm {
    Baseline,
    Candidate,
}

enum Model {
    Baseline(baseline::MicroSalNetW24),
    Candidate(candidate::MicroSalNetW24),
}

impl Model {
    fn new(arm: Arm) -> Self {
        match arm {
            Arm::Baseline => {
                Self::Baseline(baseline::MicroSalNetW24::new().unwrap().with_fast_sigmoid())
            }
            Arm::Candidate => Self::Candidate(
                candidate::MicroSalNetW24::new()
                    .unwrap()
                    .with_fast_sigmoid(),
            ),
        }
    }
    fn infer(&mut self, input: &[f32]) -> Vec<f32> {
        match self {
            Self::Baseline(model) => {
                model
                    .infer(baseline::Input::contiguous(input).unwrap())
                    .unwrap()
                    .data
            }
            Self::Candidate(model) => {
                model
                    .infer(candidate::Input::contiguous(input).unwrap())
                    .unwrap()
                    .data
            }
        }
    }
}

pub(super) struct Fixture {
    inputs: Arc<Vec<Vec<f32>>>,
    expected: Arc<Vec<Vec<u32>>>,
}

impl Fixture {
    pub(super) fn new() -> Self {
        let inputs: Arc<Vec<Vec<f32>>> = Arc::new(
            (0..BATCH)
                .map(|image| {
                    (0..3 * 65536)
                        .map(|i| ((i * 137 + i / 256 * 59 + image * 101) % 1024) as f32 / 1023.0)
                        .collect()
                })
                .collect(),
        );
        let mut reference = baseline::MicroSalNetW24::new().unwrap().with_fast_sigmoid();
        let expected: Arc<Vec<Vec<u32>>> = Arc::new(
            inputs
                .iter()
                .map(|input| {
                    reference
                        .infer(baseline::Input::contiguous(input).unwrap())
                        .unwrap()
                        .data
                        .into_iter()
                        .map(f32::to_bits)
                        .collect()
                })
                .collect(),
        );
        drop(reference);
        // The frozen oracle uses a different NN source. Its unused TLS pool
        // must not remain parked beside only the candidate worker group.
        baseline_nn::nn::fast::release_buffers();
        Self { inputs, expected }
    }
}

impl Workers {
    pub(super) fn with_fixture(cpus: &[usize], arm: Arm, fixture: &Fixture) -> Self {
        let width = cpus.len();
        let workers = cpus
            .iter()
            .copied()
            .enumerate()
            .map(|(id, cpu)| {
                let inputs = fixture.inputs.clone();
                let expected = fixture.expected.clone();
                let (request, receive) = mpsc::channel();
                let (complete, done) = mpsc::channel();
                let handle = thread::Builder::new()
                    .name(format!("w24-image-{id}"))
                    .spawn(move || {
                        let _affinity = zenbench::platform::pin_current_thread(&[cpu]).unwrap();
                        assert_eq!(
                            zenbench::platform::current_thread_affinity().unwrap(),
                            vec![cpu]
                        );
                        let mut model = Model::new(arm);
                        for image in (id..BATCH).step_by(width) {
                            let output = model.infer(&inputs[image]);
                            assert!(output.iter().all(|x| x.is_finite()));
                            assert!(
                                output
                                    .iter()
                                    .copied()
                                    .map(f32::to_bits)
                                    .eq(expected[image].iter().copied()),
                                "{arm:?} worker {id} image {image} differs from serial reference"
                            );
                        }
                        complete.send(0).unwrap();
                        while receive.recv().unwrap_or(false) {
                            let mut completed = 0;
                            for image in (id..BATCH).step_by(width) {
                                black_box(model.infer(black_box(&inputs[image])));
                                completed += 1;
                            }
                            complete.send(completed).unwrap();
                        }
                    })
                    .unwrap();
                Worker {
                    request,
                    done,
                    handle: Some(handle),
                }
            })
            .collect();
        let result = Self { workers };
        for worker in &result.workers {
            assert_eq!(worker.done.recv().unwrap(), 0);
        }
        eprintln!(
            "validated {arm:?} persistent workers={} cpus={cpus:?} batch={BATCH}",
            width
        );
        result
    }

    pub(super) fn batch(&mut self) -> usize {
        for worker in &self.workers {
            worker.request.send(true).unwrap();
        }
        let completed = self
            .workers
            .iter()
            .map(|worker| worker.done.recv().unwrap())
            .sum::<usize>();
        assert_eq!(completed, BATCH);
        completed
    }
}

impl Drop for Workers {
    fn drop(&mut self) {
        for worker in &self.workers {
            let _ = worker.request.send(false);
        }
        for worker in &mut self.workers {
            worker.handle.take().unwrap().join().unwrap();
        }
    }
}

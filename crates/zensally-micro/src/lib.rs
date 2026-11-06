//! Fixed w24 graph over the committed weights. Experimental: numerical and
//! performance comparisons with tract are required before replacing a backend.
#![no_std]
#![forbid(unsafe_code)]
extern crate alloc;
#[cfg(feature = "std")]
extern crate std;

use alloc::{format, string::String, vec, vec::Vec};
use zenjpegai::{
    nn::{
        Conv2d, ConvTranspose2d,
        fast::{self, BTensor, ConvLayer, ConvTransposeLayer, Engine, Tier},
    },
    tensor::Tensor,
};

mod depthwise;
mod pointwise;
mod workspace;
use workspace::Workspace;
#[cfg(any(
    feature = "experimental-reassociate",
    feature = "experimental-depthwise"
))]
mod reduction;
#[rustfmt::skip]
mod plan;

const MODEL: &[u8] = include_bytes!("../../zensally-tract/models/microsalnet_v3ds_w24.onnx.gz");
const SLOTS: usize = plan::NODES.len() + 1;

/// Model or input error. This experimental runtime does not accept arbitrary graphs.
#[derive(Debug)]
pub struct Error(String);
impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(&self.0)
    }
}
impl core::error::Error for Error {}
impl From<zenjpegai::Error> for Error {
    fn from(e: zenjpegai::Error) -> Self {
        Self(format!("{e}"))
    }
}
type Result<T> = core::result::Result<T, Error>;

struct Weight {
    dims: &'static [usize],
    offset: usize,
    bytes: usize,
}
#[derive(Clone, Copy)]
enum Op {
    Conv {
        w: usize,
        b: usize,
        stride: usize,
        pad: usize,
        dilation: usize,
        groups: usize,
    },
    Transpose {
        w: usize,
        b: usize,
        stride: usize,
        pad: usize,
    },
    Relu,
    Sigmoid,
    Mean,
    Add,
    Mul,
    Concat,
}
struct Node {
    inputs: &'static [usize],
    op: Op,
}
enum Kernel {
    Conv(ConvLayer),
    Pointwise(pointwise::Pointwise),
    Transpose(ConvTransposeLayer),
    Depthwise(depthwise::Depthwise),
    Plain(Op),
}

/// Fixed-shape NCHW input. Row stride is in f32 samples; channel stride is
/// explicit so cropped planes need no repacking by the caller.
#[derive(Clone, Copy)]
pub struct Input<'a> {
    data: &'a [f32],
    row_stride: usize,
    channel_stride: usize,
}
impl<'a> Input<'a> {
    pub fn new(data: &'a [f32], row_stride: usize, channel_stride: usize) -> Result<Self> {
        let required = channel_stride
            .checked_mul(2)
            .and_then(|n| row_stride.checked_mul(255).and_then(|r| n.checked_add(r)))
            .and_then(|n| n.checked_add(256));
        if row_stride < 256
            || row_stride
                .checked_mul(255)
                .and_then(|n| n.checked_add(256))
                .is_none_or(|n| channel_stride < n)
            || required.is_none_or(|n| data.len() < n)
        {
            return Err(Error(
                "input must hold three nonoverlapping 256x256 planes".into(),
            ));
        }
        Ok(Self {
            data,
            row_stride,
            channel_stride,
        })
    }
    pub fn contiguous(data: &'a [f32]) -> Result<Self> {
        Self::new(data, 256, 256 * 256)
    }
}

/// Executes every node of the committed MicroSalNet v3ds w24 model, using
/// packed owned kernels and a generated graph. Loading does not optimize ONNX.
/// Activation scratch is retained per runtime between calls, including without
/// std. Shared NN outputs use its bounded recycle pool when std is enabled.
pub struct MicroSalNetW24 {
    engine: Engine,
    kernels: Vec<Kernel>,
    uses: [usize; SLOTS],
    fused_relu: [bool; SLOTS],
    workspace: Workspace,
    fast_sigmoid: bool,
}
impl MicroSalNetW24 {
    /// Opt in to zenjpegai's vectorized Cephes sigmoid. Weights and graph stay
    /// fixed; results differ from the default libm sigmoid. Validate resulting
    /// source-image rectangles for the application's movement budget.
    pub fn with_fast_sigmoid(mut self) -> Self {
        self.fast_sigmoid = true;
        self
    }
    pub fn new() -> Result<Self> {
        Self::with_engine(Engine::with(Tier::detect(), false))
    }
    /// Scalar reference execution, for parity checks. Public types stay independent
    /// of the internal SIMD proof tokens and packed storage.
    pub fn scalar() -> Result<Self> {
        Self::with_engine(Engine::with(Tier::Scalar, false))
    }
    fn with_engine(engine: Engine) -> Result<Self> {
        let mut bytes = vec![0; plan::MODEL_BYTES];
        let outcome = zenflate::Decompressor::new()
            .gzip_decompress(MODEL, &mut bytes, enough::Unstoppable)
            .map_err(|e| Error(format!("model gzip: {e:?}")))?;
        if outcome.output_written != bytes.len() {
            return Err(Error("truncated embedded model".into()));
        }
        let read = |i: usize| -> Vec<f32> {
            let w = &plan::WEIGHTS[i];
            bytes[w.offset..w.offset + w.bytes]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect()
        };
        let mut kernels = Vec::with_capacity(plan::NODES.len());
        let mut uses = [0; SLOTS];
        for node in plan::NODES {
            for &i in node.inputs {
                uses[i] += 1;
            }
            let kernel = match node.op {
                Op::Conv {
                    w,
                    b,
                    stride,
                    pad,
                    dilation,
                    groups,
                } => {
                    let d = plan::WEIGHTS[w].dims;
                    if groups > 1 {
                        if d[1] != 1 || d[0] != groups || d[2] != 3 || d[3] != 3 {
                            return Err(Error("unsupported grouped convolution".into()));
                        }
                        Kernel::Depthwise(depthwise::Depthwise::new(
                            groups,
                            d[2],
                            d[3],
                            stride,
                            pad,
                            dilation,
                            engine.tier.block(),
                            read(w),
                            read(b),
                        ))
                    } else {
                        if dilation != 1 {
                            return Err(Error("dense dilation not implemented".into()));
                        }
                        if d[2] == 1 && d[3] == 1 && stride == 1 && pad == 0 {
                            Kernel::Pointwise(pointwise::Pointwise::new(
                                d[1],
                                d[0],
                                engine.tier.block(),
                                read(w),
                                read(b),
                            ))
                        } else {
                            Kernel::Conv(ConvLayer::new(
                                Conv2d::new(
                                    d[1],
                                    d[0],
                                    (d[2], d[3]),
                                    stride,
                                    (pad, pad),
                                    1,
                                    read(w),
                                    Some(read(b)),
                                )?,
                                &engine,
                            )?)
                        }
                    }
                }
                Op::Transpose { w, b, stride, pad } => {
                    let d = plan::WEIGHTS[w].dims;
                    if d[2] != d[3] {
                        return Err(Error("nonsquare transpose kernel".into()));
                    }
                    Kernel::Transpose(ConvTransposeLayer::new(
                        ConvTranspose2d::new(
                            d[0],
                            d[1],
                            d[2],
                            stride,
                            pad,
                            0,
                            read(w),
                            Some(read(b)),
                        )?,
                        &engine,
                    )?)
                }
                other => Kernel::Plain(other),
            };
            kernels.push(kernel);
        }
        let mut fused_relu = [false; SLOTS];
        for (i, kernel) in kernels.iter_mut().enumerate() {
            if uses[i + 1] == 1
                && plan::NODES
                    .get(i + 1)
                    .is_some_and(|next| matches!(next.op, Op::Relu) && next.inputs == [i + 1])
                && let Kernel::Depthwise(k) = kernel
            {
                k.relu = true;
                fused_relu[i + 1] = true;
            }
        }
        Ok(Self {
            engine,
            kernels,
            uses,
            fused_relu,
            workspace: Workspace::new()?,
            fast_sigmoid: false,
        })
    }

    pub fn infer(&mut self, input: Input<'_>) -> Result<zensally::SaliencyMap> {
        let out = self.execute(input, None)?;
        if (out.c, out.h, out.w) != (1, 128, 128) {
            return Err(Error("unexpected model output shape".into()));
        }
        Ok(zensally::SaliencyMap {
            width: 128,
            height: 128,
            data: out.data,
        })
    }

    fn execute(
        &mut self,
        input: Input<'_>,
        mut trace: Option<&mut Vec<Tensor<f32>>>,
    ) -> Result<Tensor<f32>> {
        let v = self.engine.tier.block();
        let workspace = &mut self.workspace;
        let mut first = workspace.take(3, 256, 256, v)?;
        for y in 0..256 {
            for x in 0..256 {
                let pixel = &mut first.data[(y * 256 + x) * v..][..v];
                for (c, value) in pixel[..3].iter_mut().enumerate() {
                    *value = input.data[c * input.channel_stride + y * input.row_stride + x];
                }
                pixel[3..].fill(0.0);
            }
        }
        let mut values: [Option<BTensor>; SLOTS] = core::array::from_fn(|_| None);
        // Shared NN kernels own their output allocation. Return those outputs
        // to the shared pool rather than retaining them in our private cache.
        let mut external = [false; SLOTS];
        values[0] = Some(first);
        let mut uses = self.uses;
        for (idx, (node, kernel)) in plan::NODES.iter().zip(&self.kernels).enumerate() {
            let mut reused = if uses[node.inputs[0]] == 1
                && matches!(
                    kernel,
                    Kernel::Plain(Op::Relu | Op::Sigmoid | Op::Add | Op::Mul)
                ) {
                values[node.inputs[0]].take()
            } else {
                None
            };
            external[idx + 1] = if reused.is_some() {
                external[node.inputs[0]]
            } else {
                matches!(kernel, Kernel::Conv(_) | Kernel::Transpose(_))
            };
            let get = |i: usize| {
                values[node.inputs[i]]
                    .as_ref()
                    .expect("generated topological slot")
            };
            let out = match kernel {
                Kernel::Conv(k) => k.forward(&self.engine, get(0))?,
                Kernel::Pointwise(k) => {
                    #[cfg(feature = "experimental-channel-tiles")]
                    if self.fast_sigmoid {
                        k.forward_tiled(&self.engine, get(0), workspace)?
                    } else {
                        k.forward(&self.engine, get(0), workspace)?
                    }
                    #[cfg(all(
                        feature = "experimental-reassociate",
                        not(feature = "experimental-channel-tiles")
                    ))]
                    if self.fast_sigmoid {
                        k.forward_reassociated(&self.engine, get(0), workspace)?
                    } else {
                        k.forward(&self.engine, get(0), workspace)?
                    }
                    #[cfg(not(any(
                        feature = "experimental-reassociate",
                        feature = "experimental-channel-tiles"
                    )))]
                    k.forward(&self.engine, get(0), workspace)?
                }
                Kernel::Transpose(k) => k.forward(&self.engine, get(0))?,
                Kernel::Depthwise(k) => {
                    #[cfg(feature = "experimental-depthwise")]
                    if self.fast_sigmoid {
                        k.forward_reassociated(&self.engine, get(0), workspace)?
                    } else {
                        k.forward(&self.engine, get(0), workspace)?
                    }
                    #[cfg(not(feature = "experimental-depthwise"))]
                    k.forward(&self.engine, get(0), workspace)?
                }
                Kernel::Plain(Op::Concat) => {
                    let [_, _] = node.inputs else {
                        return Err(Error("unexpected fixed concatenation arity".into()));
                    };
                    concatenate(&[get(0), get(1)], workspace)?
                }
                Kernel::Plain(Op::Mean) => spatial_mean(get(0), workspace)?,
                Kernel::Plain(Op::Add) => {
                    let mut out = match reused.take() {
                        Some(out) => out,
                        None => workspace.copy(get(0))?,
                    };
                    fast::add_assign(&mut out, get(1))?;
                    out
                }
                Kernel::Plain(Op::Mul) => {
                    let mut out = match reused.take() {
                        Some(out) => out,
                        None => workspace.copy(get(0))?,
                    };
                    let gate = get(1);
                    if gate.c != out.c || gate.h != 1 || gate.w != 1 {
                        return Err(Error("invalid channel gate".into()));
                    }
                    for cb in 0..out.blocks() {
                        for p in 0..out.h * out.w {
                            for lane in 0..v {
                                out.data[(cb * out.h * out.w + p) * v + lane] *=
                                    gate.data[cb * v + lane];
                            }
                        }
                    }
                    out
                }
                Kernel::Plain(Op::Relu | Op::Sigmoid) => {
                    let mut out = match reused.take() {
                        Some(out) => out,
                        None => workspace.copy(get(0))?,
                    };
                    if matches!(node.op, Op::Relu) {
                        if !self.fused_relu[node.inputs[0]] {
                            fast::relu(&mut out);
                        }
                    } else if self.fast_sigmoid {
                        fast::math::sigmoid(&self.engine, &mut out.data);
                    } else {
                        for value in &mut out.data {
                            *value = 1.0 / (1.0 + libm::expf(-*value));
                        }
                    }
                    out
                }
                _ => return Err(Error("invalid generated kernel".into())),
            };
            if let Some(trace) = trace.as_deref_mut() {
                trace.push(out.to_planar()?);
            }
            for &s in node.inputs {
                uses[s] -= 1;
                if uses[s] == 0
                    && let Some(tensor) = values[s].take()
                    && !external[s]
                {
                    workspace.give(tensor);
                }
            }
            values[idx + 1] = Some(out);
        }
        let tensor = values.last_mut().unwrap().take().unwrap();
        let result = tensor.to_planar();
        if !external[SLOTS - 1] {
            workspace.give(tensor);
        }
        Ok(result?)
    }
}

// The shared fast concatenation requires complete inner blocks. w24 has
// 24-channel skip connections, so 16-lane packing must merge partial blocks.
fn concatenate(parts: &[&BTensor], workspace: &mut Workspace) -> Result<BTensor> {
    let first = parts
        .first()
        .ok_or_else(|| Error("empty concatenation".into()))?;
    if parts
        .iter()
        .any(|p| (p.h, p.w, p.v) != (first.h, first.w, first.v))
    {
        return Err(Error("concatenation shapes disagree".into()));
    }
    let channels = parts
        .iter()
        .try_fold(0usize, |n, p| n.checked_add(p.c))
        .ok_or_else(|| Error("concatenation channel overflow".into()))?;
    let mut out = workspace.take(channels, first.h, first.w, first.v)?;
    if parts[..parts.len() - 1]
        .iter()
        .all(|part| part.c.is_multiple_of(first.v))
    {
        let mut offset = 0;
        for part in parts {
            out.data[offset..][..part.data.len()].copy_from_slice(&part.data);
            offset += part.data.len();
        }
        return Ok(out);
    }
    // Partial inner blocks need logical channel concatenation. Initialise the
    // final unused lanes as well when reusing arbitrary activation storage.
    let tail = channels % first.v;
    if tail != 0 {
        let start = channels / first.v * first.h * first.w * first.v;
        for p in out.data[start..].chunks_exact_mut(first.v) {
            p[tail..].fill(0.0);
        }
    }
    let area = first.h * first.w;
    let v = first.v;
    let mut offset = 0;
    for part in parts {
        for c in 0..part.c {
            let target = offset + c;
            for p in 0..area {
                out.data[(target / v * area + p) * v + target % v] =
                    part.data[(c / v * area + p) * v + c % v];
            }
        }
        offset += part.c;
    }
    Ok(out)
}

fn spatial_mean(x: &BTensor, workspace: &mut Workspace) -> Result<BTensor> {
    let mut out = workspace.take(x.c, 1, 1, x.v)?;
    out.data.fill(0.0);
    let area = x.h * x.w;
    for cb in 0..x.blocks() {
        for p in 0..area {
            for lane in 0..x.v {
                out.data[cb * x.v + lane] += x.data[(cb * area + p) * x.v + lane];
            }
        }
    }
    for value in &mut out.data {
        *value /= area as f32;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repeated_inference_overwrites_poisoned_workspace() {
        for fast in [false, true] {
            let mut reused = MicroSalNetW24::new().unwrap();
            if fast {
                reused = reused.with_fast_sigmoid();
            }
            for case in 0..4 {
                let input: Vec<f32> = (0..3 * 65536)
                    .map(|i| match case {
                        0 => 0.0,
                        1 => 1.0,
                        2 => ((i * 137 % 1024) as f32 - 511.0) / 511.0,
                        _ => {
                            if i % 2 == 0 {
                                -0.0
                            } else {
                                0.0
                            }
                        }
                    })
                    .collect();
                let mut fresh = MicroSalNetW24::scalar().unwrap();
                if fast {
                    fresh = fresh.with_fast_sigmoid();
                }
                reused.workspace.poison();
                let actual = reused.infer(Input::contiguous(&input).unwrap()).unwrap();
                let expected = fresh.infer(Input::contiguous(&input).unwrap()).unwrap();
                assert!(actual.data.iter().all(|v| v.is_finite()));
                assert!(
                    actual
                        .data
                        .iter()
                        .zip(&expected.data)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "case={case} fast={fast}"
                );
            }
        }
    }
    #[test]
    fn partial_channel_blocks_concatenate_without_padding_lanes() {
        for v in [8, 16] {
            for channels in [[24, 24, 1], [3, 9, 7], [16, 32, 9]] {
                let mut expected = Vec::new();
                let parts: Vec<BTensor> = channels
                    .into_iter()
                    .enumerate()
                    .map(|(i, c)| {
                        let values: Vec<f32> = (0..c * 3 * 5)
                            .map(|n| {
                                if n % 11 == 0 {
                                    -0.0
                                } else {
                                    (i * 1000 + n) as f32
                                }
                            })
                            .collect();
                        expected.extend_from_slice(&values);
                        BTensor::from_planar(&Tensor::from_vec(c, 3, 5, values).unwrap(), v)
                            .unwrap()
                    })
                    .collect();
                let refs: Vec<&BTensor> = parts.iter().collect();
                let actual = concatenate(&refs, &mut Workspace::new().unwrap())
                    .unwrap()
                    .to_planar()
                    .unwrap();
                assert_eq!(
                    actual.data.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                );
            }
        }
    }
    #[test]
    fn complete_graph_fusion_preserves_unfused_bits() {
        let mut fused = MicroSalNetW24::new().unwrap();
        let mut plain = MicroSalNetW24::new().unwrap();
        plain.fused_relu.fill(false);
        for k in &mut plain.kernels {
            if let Kernel::Depthwise(k) = k {
                k.relu = false;
            }
        }
        let input: Vec<f32> = (0..3 * 65536)
            .map(|i| (i * 83 % 127) as f32 / 126.0)
            .collect();
        let input = Input::contiguous(&input).unwrap();
        let a = fused.execute(input, None).unwrap();
        let b = plain.execute(input, None).unwrap();
        for (i, (&a, &b)) in a.data.iter().zip(&b.data).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "fused/unfused at {i}");
        }
    }
}

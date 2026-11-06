//! Execution-owned storage for the fixed graph. Buffers remain charged as live
//! NN storage while retained here; resizing never changes their capacity.
use super::{Error, Result};
use alloc::vec::Vec;
use zenjpegai::nn::fast::BTensor;

pub(super) struct Workspace {
    spare: Vec<BTensor>,
}

impl Workspace {
    pub(super) fn new() -> Result<Self> {
        let mut spare = Vec::new();
        // The graph bounds the number of live and spare tensors. Reserve the
        // descriptor storage once; give() cannot grow it during inference.
        spare
            .try_reserve_exact(super::SLOTS)
            .map_err(|_| Error("workspace descriptor allocation failed".into()))?;
        Ok(Self { spare })
    }

    pub(super) fn take(&mut self, c: usize, h: usize, w: usize, v: usize) -> Result<BTensor> {
        let n = c
            .div_ceil(v)
            .checked_mul(h)
            .and_then(|n| n.checked_mul(w))
            .and_then(|n| n.checked_mul(v))
            .ok_or_else(|| Error("workspace shape overflow".into()))?;
        if let Some(i) = self
            .spare
            .iter()
            .enumerate()
            .filter(|(_, b)| b.data.capacity() >= n)
            .min_by_key(|(_, b)| b.data.capacity())
            .map(|(i, _)| i)
        {
            let mut b = self.spare.swap_remove(i);
            b.data.resize(n, 0.0);
            (b.c, b.h, b.w, b.v) = (c, h, w, v);
            return Ok(b);
        }
        Ok(BTensor::scratch(c, h, w, v)?)
    }

    pub(super) fn give(&mut self, tensor: BTensor) {
        #[cfg(not(feature = "std"))]
        {
            // The alloc-only kernels cannot consume this cache for their own
            // dense/transpose outputs. Bound idle retention so those outputs
            // do not overlap an unnecessarily large set of spare activations.
            const MAX_SPARE_FLOATS: usize = 1 << 19;
            if tensor.data.capacity() > MAX_SPARE_FLOATS {
                return;
            }
            while self.spare.iter().map(|b| b.data.capacity()).sum::<usize>()
                + tensor.data.capacity()
                > MAX_SPARE_FLOATS
            {
                let Some((i, _)) = self
                    .spare
                    .iter()
                    .enumerate()
                    .max_by_key(|(_, b)| b.data.capacity())
                else {
                    break;
                };
                drop(self.spare.swap_remove(i));
            }
        }
        assert!(self.spare.len() < self.spare.capacity());
        self.spare.push(tensor);
    }

    pub(super) fn copy(&mut self, x: &BTensor) -> Result<BTensor> {
        let mut out = self.take(x.c, x.h, x.w, x.v)?;
        out.data.copy_from_slice(&x.data);
        Ok(out)
    }

    #[cfg(test)]
    pub(super) fn poison(&mut self) {
        for b in &mut self.spare {
            b.data.fill(f32::NAN);
        }
    }
}

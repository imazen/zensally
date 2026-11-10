//! Private experimental arithmetic. Array round trips permit Rust's algebraic
//! operators without exposing vector representations or using unsafe code.
use magetypes::simd::{backends::F32x8Backend, generic::f32x8};

#[inline(always)]
#[cfg_attr(feature = "experimental-algebraic", clippy::msrv = "1.98")]
pub(super) fn madd<T: F32x8Backend>(token: T, a: f32x8<T>, b: f32x8<T>, c: f32x8<T>) -> f32x8<T> {
    #[cfg(feature = "experimental-algebraic")]
    {
        let a: [f32; 8] = a.into();
        let b: [f32; 8] = b.into();
        let c: [f32; 8] = c.into();
        f32x8::from_array_t(
            token,
            core::array::from_fn(|i| a[i].algebraic_mul(b[i]).algebraic_add(c[i])),
        )
    }
    #[cfg(not(feature = "experimental-algebraic"))]
    {
        let _ = token;
        #[cfg(not(target_arch = "wasm32"))]
        {
            a.mul_add_fused(b, c)
        }
        #[cfg(target_arch = "wasm32")]
        {
            a * b + c
        }
    }
}

//! # kuji
//!
//! Stochastic sampling primitives: Gumbel-max, Reservoir, MCMC.
//!
//! (kuji: lottery/draw in Japanese)
//!
//! ## Modules
//!
//! - `reservoir`: Streaming sampling (reservoir sampling)
//! - `gumbel`: Differentiable sampling (Gumbel-max trick)
//! - `neighbor`: Graph neighbor sampling (extracted from lattix)
//!
//! ## Quick Start
//!
//! ```rust
//! use kuji::reservoir::ReservoirSampler;
//!
//! let mut sampler = ReservoirSampler::new(5);
//! for i in 0..100 {
//!     sampler.add(i);
//! }
//! let samples = sampler.samples();
//! assert_eq!(samples.len(), 5);
//! ```
//!
//! ## Research Context
//!
//! ### Reservoir Sampling
//!
//! - **Algorithm L** (Li, 1994): Faster $O(k(1 + \log(N/k)))$ approach that skips items.
//!   This is what `kuji::reservoir` currently implements.
//! - **Algorithm R** (Vitter, 1985): The standard $O(N)$ baseline, implemented as
//!   [`ReservoirSamplerR`] for reference and A/B comparison.
//! - **Weighted sampling (A-Res)** (Efraimidis–Spirakis, 2006): Streaming top-k with weights,
//!   implemented as [`WeightedReservoirSampler`].
//!
//! ### Gumbel-Max
//!
//! The Gumbel-Max trick $ \arg\max (\log p_i + g_i) $ generalizes to sampling
//! $k$ items without replacement by simply taking the top-$k$ perturbed values
//! (Vieira, 2014).
//!
#![allow(dead_code)]

pub mod gumbel;
pub mod neighbor;
pub mod reservoir;

pub use gumbel::{gumbel_max_sample, gumbel_topk_sample, gumbel_topk_sample_with_rng};
pub use reservoir::{
    ReservoirSampler, ReservoirSamplerR, WeightedReservoirError, WeightedReservoirSampler,
};

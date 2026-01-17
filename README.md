# kuji

Stochastic sampling primitives for unbiased data selection and stream processing.
Implements reservoir sampling (Algorithm L/R), weighted sampling, and Gumbel-max for top-k.

Dual-licensed under MIT or Apache-2.0.

[crates.io](https://crates.io/crates/kuji) | [docs.rs](https://docs.rs/kuji)

```rust
use kuji::reservoir::ReservoirSampler;

let mut sampler = ReservoirSampler::new(5);
for i in 0..100 {
    sampler.add(i);
}
let samples = sampler.samples();
assert_eq!(samples.len(), 5);
```


# kuji

 Stochastic sampling primitives: reservoir sampling (Algorithm L/R, weighted) and Gumbel-max.

Dual-licensed under MIT or Apache-2.0.

```rust
use kuji::reservoir::ReservoirSampler;

let mut sampler = ReservoirSampler::new(5);
for i in 0..100 {
    sampler.add(i);
}
let samples = sampler.samples();
assert_eq!(samples.len(), 5);
```


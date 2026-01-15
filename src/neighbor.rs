//! Graph neighbor sampling.
//!
//! Provides utilities for sampling neighbors from a graph, useful for
//! GNN training (GraphSAGE) and random walks (Node2Vec).

use rand::prelude::*;
#[cfg(test)]
use std::collections::HashSet;

/// Sampler for graph neighborhoods.
pub struct NeighborSampler {
    seed: Option<u64>,
}

impl Default for NeighborSampler {
    fn default() -> Self {
        Self::new()
    }
}

impl NeighborSampler {
    /// Create a new neighbor sampler.
    pub fn new() -> Self {
        Self { seed: None }
    }

    /// Set random seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Sample `k` neighbors uniformly with replacement.
    ///
    /// # Arguments
    ///
    /// * `neighbors`: Slice of neighbor IDs
    /// * `k`: Number of samples to draw
    pub fn sample_uniform_with_replacement<T: Clone>(&self, neighbors: &[T], k: usize) -> Vec<T> {
        if neighbors.is_empty() {
            return Vec::new();
        }

        let mut rng: Box<dyn RngCore> = match self.seed {
            Some(s) => Box::new(StdRng::seed_from_u64(s)),
            None => Box::new(rand::rng()),
        };

        (0..k)
            .map(|_| {
                let idx = rng.random_range(0..neighbors.len());
                neighbors[idx].clone()
            })
            .collect()
    }

    /// Sample `k` neighbors uniformly without replacement.
    ///
    /// If `k >= neighbors.len()`, returns all neighbors (shuffled).
    pub fn sample_uniform_without_replacement<T: Clone + Eq + std::hash::Hash>(
        &self,
        neighbors: &[T],
        k: usize,
    ) -> Vec<T> {
        if neighbors.is_empty() {
            return Vec::new();
        }

        if k >= neighbors.len() {
            let mut result = neighbors.to_vec();
            let mut rng: Box<dyn RngCore> = match self.seed {
                Some(s) => Box::new(StdRng::seed_from_u64(s)),
                None => Box::new(rand::rng()),
            };
            result.shuffle(&mut rng);
            return result;
        }

        let mut rng: Box<dyn RngCore> = match self.seed {
            Some(s) => Box::new(StdRng::seed_from_u64(s)),
            None => Box::new(rand::rng()),
        };

        // Reservoir sampling for indices
        let mut indices: Vec<usize> = (0..neighbors.len()).collect();
        indices.shuffle(&mut rng);

        indices
            .into_iter()
            .take(k)
            .map(|i| neighbors[i].clone())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_with_replacement() {
        let sampler = NeighborSampler::new().with_seed(42);
        let neighbors = vec![1, 2, 3];
        let samples = sampler.sample_uniform_with_replacement(&neighbors, 10);

        assert_eq!(samples.len(), 10);
        for s in samples {
            assert!(neighbors.contains(&s));
        }
    }

    #[test]
    fn test_sample_without_replacement() {
        let sampler = NeighborSampler::new().with_seed(42);
        let neighbors = vec![1, 2, 3, 4, 5];
        let samples = sampler.sample_uniform_without_replacement(&neighbors, 3);

        assert_eq!(samples.len(), 3);
        // Check uniqueness
        let set: HashSet<_> = samples.iter().collect();
        assert_eq!(set.len(), 3);
        for s in samples {
            assert!(neighbors.contains(&s));
        }
    }
}

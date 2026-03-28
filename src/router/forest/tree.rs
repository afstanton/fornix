//! Decision tree node — CART binary splits using Gini impurity.

use serde::{Deserialize, Serialize};

/// A node in the decision tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Node {
    /// An internal split node.
    Split {
        /// Which feature dimension to split on.
        feature: usize,
        /// Threshold value — samples with `feature <= threshold` go left.
        threshold: f32,
        left: Box<Node>,
        right: Box<Node>,
    },
    /// A leaf node storing class probability estimates.
    Leaf {
        /// Probability of class 0 (model_a) at this leaf.
        prob_a: f32,
        /// Probability of class 1 (model_b) at this leaf.
        prob_b: f32,
        /// Number of training samples that reached this leaf.
        n_samples: usize,
    },
}

impl Node {
    /// Predict the probability of class 0 (model_a) for a single sample.
    pub fn predict_proba(&self, sample: &[f32]) -> f32 {
        match self {
            Node::Leaf { prob_a, .. } => *prob_a,
            Node::Split { feature, threshold, left, right } => {
                if sample[*feature] <= *threshold {
                    left.predict_proba(sample)
                } else {
                    right.predict_proba(sample)
                }
            }
        }
    }
}

/// Gini impurity of a binary class distribution.
///
/// `n_a` = count of class 0, `n_b` = count of class 1.
pub fn gini(n_a: usize, n_b: usize) -> f32 {
    let n = (n_a + n_b) as f32;
    if n < f32::EPSILON {
        return 0.0;
    }
    let pa = n_a as f32 / n;
    let pb = n_b as f32 / n;
    1.0 - (pa * pa + pb * pb)
}

/// Weighted Gini after a binary split.
pub fn split_gini(
    left_a: usize, left_b: usize,
    right_a: usize, right_b: usize,
) -> f32 {
    let n_left = (left_a + left_b) as f32;
    let n_right = (right_a + right_b) as f32;
    let n_total = n_left + n_right;
    if n_total < f32::EPSILON {
        return 0.0;
    }
    (n_left / n_total) * gini(left_a, left_b)
        + (n_right / n_total) * gini(right_a, right_b)
}

/// Hyperparameters for a single decision tree.
#[derive(Debug, Clone)]
pub struct TreeParams {
    /// Maximum tree depth. `None` = unlimited.
    pub max_depth: Option<usize>,
    /// Minimum samples required to attempt a split.
    pub min_samples_split: usize,
    /// Minimum samples in a leaf.
    pub min_samples_leaf: usize,
    /// How many features to consider at each split.
    /// `None` = sqrt(n_features).
    pub max_features: Option<usize>,
    /// Seed for deterministic feature selection (per-tree RNG).
    pub seed: u64,
}

impl Default for TreeParams {
    fn default() -> Self {
        Self {
            max_depth: Some(20),
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_features: None,
            seed: 42,
        }
    }
}

/// Build a decision tree from training data.
///
/// `features` is `[n_samples][n_features]`.
/// `labels` is a binary label (0 = model_a, 1 = model_b) per sample.
/// Returns the root [`Node`].
pub fn build_tree(
    features: &[Vec<f32>],
    labels: &[u8],
    params: &TreeParams,
) -> Node {
    let indices: Vec<usize> = (0..features.len()).collect();
    let n_features = features.first().map(|r| r.len()).unwrap_or(0);
    let max_features = params.max_features.unwrap_or_else(|| {
        let sq = (n_features as f64).sqrt().ceil() as usize;
        sq.max(1)
    });
    build_node(features, labels, &indices, 0, params, max_features, params.seed)
}

fn build_node(
    features: &[Vec<f32>],
    labels: &[u8],
    indices: &[usize],
    depth: usize,
    params: &TreeParams,
    max_features: usize,
    seed: u64,
) -> Node {
    let n = indices.len();
    let n_a = indices.iter().filter(|&&i| labels[i] == 0).count();
    let n_b = n - n_a;

    // Pure leaf or stopping condition
    let at_max_depth = params.max_depth.map_or(false, |d| depth >= d);
    let too_small = n < params.min_samples_split;
    let pure = n_a == 0 || n_b == 0;

    if pure || at_max_depth || too_small {
        return make_leaf(n_a, n_b);
    }

    let n_features = features[0].len();

    // Select a random subset of features using a simple LCG
    let feature_indices = random_feature_subset(n_features, max_features, seed ^ (depth as u64 * 2654435761));

    let best = best_split(features, labels, indices, &feature_indices, params.min_samples_leaf);

    match best {
        None => make_leaf(n_a, n_b),
        Some((feat, thresh)) => {
            let (left_idx, right_idx): (Vec<usize>, Vec<usize>) = indices
                .iter()
                .partition(|&&i| features[i][feat] <= thresh);

            if left_idx.is_empty() || right_idx.is_empty() {
                return make_leaf(n_a, n_b);
            }

            let left = build_node(features, labels, &left_idx, depth + 1, params, max_features, seed.wrapping_add(1));
            let right = build_node(features, labels, &right_idx, depth + 1, params, max_features, seed.wrapping_add(2));

            Node::Split {
                feature: feat,
                threshold: thresh,
                left: Box::new(left),
                right: Box::new(right),
            }
        }
    }
}

fn make_leaf(n_a: usize, n_b: usize) -> Node {
    let n = (n_a + n_b) as f32;
    let prob_a = if n < f32::EPSILON { 0.5 } else { n_a as f32 / n };
    Node::Leaf { prob_a, prob_b: 1.0 - prob_a, n_samples: n_a + n_b }
}

fn best_split(
    features: &[Vec<f32>],
    labels: &[u8],
    indices: &[usize],
    feature_indices: &[usize],
    min_samples_leaf: usize,
) -> Option<(usize, f32)> {
    let parent_n_a = indices.iter().filter(|&&i| labels[i] == 0).count();
    let parent_n_b = indices.len() - parent_n_a;
    let parent_gini = gini(parent_n_a, parent_n_b);

    let mut best_gain = 0.0_f32;
    let mut best: Option<(usize, f32)> = None;

    for &feat in feature_indices {
        // Collect and sort (value, label) pairs for this feature
        let mut vals: Vec<(f32, u8)> = indices.iter()
            .map(|&i| (features[i][feat], labels[i]))
            .collect();
        vals.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let n = vals.len();
        let mut left_a = 0usize;
        let mut left_b = 0usize;
        let mut right_a = parent_n_a;
        let mut right_b = parent_n_b;

        for split_pos in 0..n - 1 {
            if vals[split_pos].1 == 0 { left_a += 1; right_a -= 1; }
            else { left_b += 1; right_b -= 1; }

            // Skip if adjacent values are equal (no meaningful split point)
            if (vals[split_pos].0 - vals[split_pos + 1].0).abs() < f32::EPSILON {
                continue;
            }
            // Enforce min_samples_leaf
            let left_n = left_a + left_b;
            let right_n = right_a + right_b;
            if left_n < min_samples_leaf || right_n < min_samples_leaf {
                continue;
            }

            let sg = split_gini(left_a, left_b, right_a, right_b);
            let gain = parent_gini - sg;
            if gain > best_gain {
                best_gain = gain;
                let threshold = (vals[split_pos].0 + vals[split_pos + 1].0) / 2.0;
                best = Some((feat, threshold));
            }
        }
    }

    best
}

/// Simple LCG-based feature subset selection (no `rand` dependency needed).
fn random_feature_subset(n_features: usize, k: usize, seed: u64) -> Vec<usize> {
    if k >= n_features {
        return (0..n_features).collect();
    }
    let mut rng = seed.wrapping_add(1);
    let mut pool: Vec<usize> = (0..n_features).collect();
    // Fisher-Yates shuffle on the pool, take first k
    for i in 0..k {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let j = i + (rng as usize % (n_features - i));
        pool.swap(i, j);
    }
    pool[..k].to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn features(rows: &[[f32; 2]]) -> Vec<Vec<f32>> {
        rows.iter().map(|r| r.to_vec()).collect()
    }

    #[test]
    fn gini_pure_class_a_is_zero() {
        assert!((gini(10, 0)).abs() < 1e-6);
    }

    #[test]
    fn gini_uniform_is_half() {
        assert!((gini(5, 5) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn gini_zero_samples_is_zero() {
        assert_eq!(gini(0, 0), 0.0);
    }

    #[test]
    fn build_tree_pure_class_a_is_leaf() {
        let f = features(&[[0.1, 0.2], [0.3, 0.4]]);
        let l = vec![0u8, 0u8];
        let node = build_tree(&f, &l, &TreeParams::default());
        assert!(matches!(node, Node::Leaf { prob_a, .. } if (prob_a - 1.0).abs() < 1e-6));
    }

    #[test]
    fn build_tree_pure_class_b_is_leaf() {
        let f = features(&[[0.1, 0.2], [0.3, 0.4]]);
        let l = vec![1u8, 1u8];
        let node = build_tree(&f, &l, &TreeParams::default());
        assert!(matches!(node, Node::Leaf { prob_a, .. } if prob_a.abs() < 1e-6));
    }

    #[test]
    fn build_tree_linearly_separable() {
        // Feature 0 splits cleanly at 0.5: < 0.5 → class 0, ≥ 0.5 → class 1
        let f = features(&[
            [0.1, 0.0], [0.2, 0.0], [0.3, 0.0],
            [0.7, 0.0], [0.8, 0.0], [0.9, 0.0],
        ]);
        let l = vec![0, 0, 0, 1, 1, 1];
        let tree = build_tree(&f, &l, &TreeParams::default());

        // Predict class 0 for small values
        let p_a = tree.predict_proba(&[0.1, 0.0]);
        assert!(p_a > 0.5, "expected model_a prob > 0.5 for left side, got {}", p_a);

        // Predict class 1 for large values
        let p_a2 = tree.predict_proba(&[0.9, 0.0]);
        assert!(p_a2 < 0.5, "expected model_a prob < 0.5 for right side, got {}", p_a2);
    }

    #[test]
    fn predict_proba_returns_value_in_zero_one() {
        let f = features(&[[0.1, 0.5], [0.9, 0.2], [0.4, 0.8], [0.6, 0.3]]);
        let l = vec![0, 1, 0, 1];
        let tree = build_tree(&f, &l, &TreeParams::default());
        for row in &f {
            let p = tree.predict_proba(row);
            assert!((0.0..=1.0).contains(&p));
        }
    }

    #[test]
    fn random_feature_subset_length() {
        let subset = random_feature_subset(10, 3, 12345);
        assert_eq!(subset.len(), 3);
    }

    #[test]
    fn random_feature_subset_no_duplicates() {
        let subset = random_feature_subset(10, 5, 99);
        let unique: std::collections::HashSet<_> = subset.iter().collect();
        assert_eq!(unique.len(), 5);
    }

    #[test]
    fn random_feature_subset_k_ge_n_returns_all() {
        let subset = random_feature_subset(4, 10, 1);
        assert_eq!(subset.len(), 4);
    }
}

//! Random Forest ensemble: training and inference.

use serde::{Deserialize, Serialize};

use crate::router::{
    error::{Error, Result},
    forest::tree::{build_tree, Node, TreeParams},
};

/// A trained Random Forest for binary routing classification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RandomForest {
    /// Individual decision trees.
    trees: Vec<Node>,
    /// Number of input features expected at inference.
    pub n_features: usize,
    /// Number of trees in the forest.
    pub n_estimators: usize,
}

impl RandomForest {
    /// Predict `P(model_a)` for a single embedding vector.
    pub fn predict_proba(&self, sample: &[f32]) -> Result<f32> {
        if sample.len() != self.n_features {
            return Err(Error::forest(format!(
                "expected {} features, got {}",
                self.n_features,
                sample.len()
            )));
        }
        if self.trees.is_empty() {
            return Err(Error::forest("forest has no trees"));
        }
        let avg: f32 = self.trees.iter().map(|t| t.predict_proba(sample)).sum::<f32>()
            / self.trees.len() as f32;
        Ok(avg)
    }

    /// Predict labels for a batch of samples.
    pub fn predict_proba_batch(&self, samples: &[Vec<f32>]) -> Result<Vec<f32>> {
        samples.iter().map(|s| self.predict_proba(s)).collect()
    }

    /// Number of trees.
    pub fn n_trees(&self) -> usize {
        self.trees.len()
    }
}

/// Hyperparameters for training a [`RandomForest`].
#[derive(Debug, Clone)]
pub struct ForestParams {
    pub n_estimators: usize,
    pub tree: TreeParams,
    pub seed: u64,
}

impl Default for ForestParams {
    fn default() -> Self {
        Self {
            n_estimators: 100,
            tree: TreeParams::default(),
            seed: 420,
        }
    }
}

/// Train a Random Forest on a feature matrix and binary labels.
pub fn train(
    features: &[Vec<f32>],
    labels: &[u8],
    params: &ForestParams,
) -> Result<RandomForest> {
    if features.is_empty() || labels.is_empty() {
        return Err(Error::forest("training data must not be empty"));
    }
    if features.len() != labels.len() {
        return Err(Error::forest(format!(
            "features ({}) and labels ({}) length mismatch",
            features.len(),
            labels.len()
        )));
    }
    for (i, row) in features.iter().enumerate() {
        if row.is_empty() {
            return Err(Error::forest(format!("sample {} has zero features", i)));
        }
    }

    let n_samples = features.len();
    let n_features = features[0].len();

    let trees: Vec<Node> = (0..params.n_estimators)
        .map(|t| {
            let boot_indices = bootstrap_indices(n_samples, params.seed.wrapping_add(t as u64));
            let boot_features: Vec<Vec<f32>> =
                boot_indices.iter().map(|&i| features[i].clone()).collect();
            let boot_labels: Vec<u8> =
                boot_indices.iter().map(|&i| labels[i]).collect();

            let mut tree_params = params.tree.clone();
            tree_params.seed = params.seed.wrapping_mul(2654435761).wrapping_add(t as u64);
            build_tree(&boot_features, &boot_labels, &tree_params)
        })
        .collect();

    Ok(RandomForest { trees, n_features, n_estimators: params.n_estimators })
}

/// Bootstrap sample indices (sampling with replacement).
fn bootstrap_indices(n: usize, seed: u64) -> Vec<usize> {
    let mut rng = seed.wrapping_add(1);
    (0..n)
        .map(|_| {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (rng >> 33) as usize % n
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn linearly_separable() -> (Vec<Vec<f32>>, Vec<u8>) {
        let features: Vec<Vec<f32>> = (0..20).map(|i| vec![i as f32 / 20.0, 0.5]).collect();
        let labels: Vec<u8> = (0..20).map(|i| if i < 10 { 0 } else { 1 }).collect();
        (features, labels)
    }

    #[test]
    fn train_and_predict_left_prefers_a() {
        let (f, l) = linearly_separable();
        let forest = train(&f, &l, &ForestParams { n_estimators: 10, ..Default::default() }).unwrap();
        let p = forest.predict_proba(&[0.1, 0.5]).unwrap();
        assert!(p > 0.5, "p={}", p);
    }

    #[test]
    fn train_and_predict_right_prefers_b() {
        let (f, l) = linearly_separable();
        let forest = train(&f, &l, &ForestParams { n_estimators: 10, ..Default::default() }).unwrap();
        let p = forest.predict_proba(&[0.9, 0.5]).unwrap();
        assert!(p < 0.5, "p={}", p);
    }

    #[test]
    fn proba_in_zero_one() {
        let (f, l) = linearly_separable();
        let forest = train(&f, &l, &ForestParams { n_estimators: 5, ..Default::default() }).unwrap();
        for row in &f {
            let p = forest.predict_proba(row).unwrap();
            assert!((0.0..=1.0).contains(&p));
        }
    }

    #[test]
    fn wrong_dimension_is_error() {
        let (f, l) = linearly_separable();
        let forest = train(&f, &l, &ForestParams { n_estimators: 3, ..Default::default() }).unwrap();
        assert!(forest.predict_proba(&[0.5]).is_err());
    }

    #[test]
    fn n_trees_matches_n_estimators() {
        let (f, l) = linearly_separable();
        let forest = train(&f, &l, &ForestParams { n_estimators: 7, ..Default::default() }).unwrap();
        assert_eq!(forest.n_trees(), 7);
    }

    #[test]
    fn empty_features_is_error() {
        assert!(train(&[], &[], &ForestParams::default()).is_err());
    }

    #[test]
    fn label_mismatch_is_error() {
        let f = vec![vec![0.1_f32]];
        let l: Vec<u8> = vec![0, 1];
        assert!(train(&f, &l, &ForestParams::default()).is_err());
    }

    #[test]
    fn batch_predict_matches_individual() {
        let (f, l) = linearly_separable();
        let forest = train(&f, &l, &ForestParams { n_estimators: 5, ..Default::default() }).unwrap();
        let batch = forest.predict_proba_batch(&f).unwrap();
        for (i, row) in f.iter().enumerate() {
            let ind = forest.predict_proba(row).unwrap();
            assert!((batch[i] - ind).abs() < 1e-10);
        }
    }

    #[test]
    fn forest_roundtrips_through_json() {
        let (f, l) = linearly_separable();
        let forest = train(&f, &l, &ForestParams { n_estimators: 3, ..Default::default() }).unwrap();
        let json = serde_json::to_string(&forest).unwrap();
        let restored: RandomForest = serde_json::from_str(&json).unwrap();
        let p1 = forest.predict_proba(&[0.3, 0.5]).unwrap();
        let p2 = restored.predict_proba(&[0.3, 0.5]).unwrap();
        assert!((p1 - p2).abs() < 1e-10);
    }
}

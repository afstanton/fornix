//! Re-exports the Random Forest training API and types.

mod tree;
pub use tree::{build_tree, gini, Node, TreeParams};

mod forest_impl;
pub use forest_impl::{train, ForestParams, RandomForest};

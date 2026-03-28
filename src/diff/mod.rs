//! Text diffing and focused snippet extraction.
//!
//! Produces human-readable diff snippets from document version pairs,
//! highlighting exactly what changed with `[[...]]` markers and trimming
//! surrounding context to a configurable window.
//!
//! Uses the `similar` crate (LCS-based word diffing, same algorithm as
//! Ruby's `diff-lcs`) for the underlying edit-distance computation.
//!
//! # Quick start
//!
//! ```rust
//! use fornix::diff::snippet::focused_pair;
//!
//! let (prev, curr) = focused_pair(
//!     "The quick brown fox",
//!     "The quick red fox",
//!     1000,
//!     3,
//! );
//! assert!(prev.contains("[[brown]]"));
//! assert!(curr.contains("[[red]]"));
//! ```

pub mod error;
pub mod snippet;

pub use error::{Error, Result};
pub use snippet::{
    boundary_aware_stitched_pair, focused_pair, StitchedPair, Unit,
};

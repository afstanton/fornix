//! Autonomous agent runtime.
//!
//! The agent solves open-ended objectives by iterating through a
//! model-call → tool-dispatch loop, with support for recursive sub-agents,
//! token/step/time budgets, write-before-read policy enforcement, and
//! optional memory compaction for long sessions.
//!
//! # Provider agnosticism
//!
//! The LLM and tool registry are injected via the [`ModelClient`] and
//! [`ToolRegistry`] traits. For pure-Rust consumers, `rig` is a natural
//! wiring point. The Ruby layer bridges through Magnus to RubyLLM.
//!
//! # Quick start
//!
//! ```rust,no_run
//! use fornix::agent::{Engine, Policy, CallConfig};
//! // Wire up your ModelClient and ToolRegistry implementations, then:
//! // let engine = Engine::new(model, tools, Policy::new("/workspace"), "sys");
//! // let answer = engine.solve("Summarise the codebase", &CallConfig::default(), None)?;
//! ```

pub mod engine;
pub mod error;
pub mod policy;
pub mod token_budget;
pub mod traits;
pub mod types;

pub use engine::Engine;
pub use error::{Error, Result};
pub use policy::Policy;
pub use token_budget::TokenBudget;
pub use traits::{MemoryCompactor, ModelClient, ToolRegistry};
pub use types::{CallConfig, Message, ModelTurn, ToolCall, ToolDef, ToolResult};

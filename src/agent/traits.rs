//! Pluggable traits for the agent engine.

use crate::agent::{error::Result, types::{CallConfig, Message, ModelTurn, ToolCall, ToolDef, ToolResult}};

/// A model client that can complete conversations with optional tool use.
pub trait ModelClient: Send + Sync {
    fn complete(
        &self,
        messages: &[Message],
        tools: &[ToolDef],
        config: &CallConfig,
        system_prompt: &str,
    ) -> Result<ModelTurn>;
}

/// A registry of tools the agent can call.
pub trait ToolRegistry: Send + Sync {
    /// Return tool definitions to pass to the model.
    fn tool_definitions(&self) -> Vec<ToolDef>;

    /// Execute a tool call and return its result.
    fn execute(&self, call: &ToolCall) -> ToolResult;
}

/// An optional memory compactor that summarises long conversations.
pub trait MemoryCompactor: Send + Sync {
    fn should_compact(&self, current_tokens: usize, context_window: usize) -> bool;
    fn compact(&self, messages: &[Message], query: &str) -> Result<CompactionResult>;
}

#[derive(Debug, Clone)]
pub struct CompactionResult {
    pub messages: Vec<Message>,
    pub removed_messages: usize,
    pub summary: String,
    pub summary_length: usize,
}

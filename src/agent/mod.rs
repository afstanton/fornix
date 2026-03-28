//! Agent session runtime, tool registry, and MCP bridge.

use std::collections::HashMap;

/// A single turn in an agent session.
pub struct ModelTurn {
    pub role: String,
    pub content: String,
    pub tool_calls: Vec<ToolCall>,
}

/// A tool invocation requested by the model.
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: HashMap<String, serde_json::Value>,
}

/// The result of executing a tool.
pub struct ToolResult {
    pub call_id: String,
    pub content: String,
    pub is_error: bool,
}

/// Interface for session storage backends (jsonl, log, in-memory, etc.).
pub trait SessionBackend: Send + Sync {
    type Error: std::error::Error + Send + Sync + 'static;

    fn append(&self, session_id: &str, turn: &ModelTurn) -> Result<(), Self::Error>;
    fn load(&self, session_id: &str) -> Result<Vec<ModelTurn>, Self::Error>;
    fn clear(&self, session_id: &str) -> Result<(), Self::Error>;
}

/// Interface for individual agent tools.
pub trait AgentTool: Send + Sync {
    type Error: std::error::Error + Send + Sync + 'static;

    fn name(&self) -> &'static str;
    fn description(&self) -> &'static str;
    fn call(&self, arguments: HashMap<String, serde_json::Value>) -> Result<ToolResult, Self::Error>;
}

/// Interface for MCP (Model Context Protocol) server connections.
pub trait McpBridge: Send + Sync {
    type Error: std::error::Error + Send + Sync + 'static;

    fn list_tools(&self) -> Result<Vec<String>, Self::Error>;
    fn call_tool(
        &self,
        name: &str,
        arguments: HashMap<String, serde_json::Value>,
    ) -> Result<ToolResult, Self::Error>;
}

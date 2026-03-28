//! Agent domain types: messages, tool calls, tool results, model turns.

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

/// A message in the conversation history.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub tool_calls: Vec<SerializedToolCall>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl Message {
    pub fn user(content: impl Into<String>) -> Self {
        Self { role: "user".to_string(), content: content.into(), tool_calls: vec![], tool_call_id: None, name: None }
    }
    pub fn assistant(content: impl Into<String>) -> Self {
        Self { role: "assistant".to_string(), content: content.into(), tool_calls: vec![], tool_call_id: None, name: None }
    }
    pub fn system(content: impl Into<String>) -> Self {
        Self { role: "system".to_string(), content: content.into(), tool_calls: vec![], tool_call_id: None, name: None }
    }
    pub fn tool_result(tool_call_id: impl Into<String>, name: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: "tool".to_string(),
            content: content.into(),
            tool_calls: vec![],
            tool_call_id: Some(tool_call_id.into()),
            name: Some(name.into()),
        }
    }
}

/// A tool call as it appears serialised in the conversation history.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SerializedToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: FunctionCall,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String, // JSON string
}

/// A live tool call produced by the model during a turn.
#[derive(Debug, Clone)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: HashMap<String, serde_json::Value>,
}

impl ToolCall {
    /// Retrieve a string argument by name.
    pub fn arg_str(&self, key: &str) -> Option<&str> {
        self.arguments.get(key)?.as_str()
    }

    /// Whether this is a recursive sub-agent tool call.
    pub fn is_recursive(&self) -> bool {
        matches!(self.name.as_str(), "subtask" | "execute")
    }

    /// Extract the objective from a recursive tool call.
    pub fn recursive_objective(&self) -> String {
        ["objective", "task", "prompt"]
            .iter()
            .filter_map(|&k| self.arg_str(k))
            .find(|s| !s.trim().is_empty())
            .unwrap_or("")
            .trim()
            .to_string()
    }
}

/// The result of executing a tool.
#[derive(Debug, Clone)]
pub struct ToolResult {
    pub tool_call_id: String,
    pub name: String,
    pub content: String,
    pub error: bool,
    pub structured_content: Option<serde_json::Value>,
}

impl ToolResult {
    pub fn ok(tool_call_id: impl Into<String>, name: impl Into<String>, content: impl Into<String>) -> Self {
        Self { tool_call_id: tool_call_id.into(), name: name.into(), content: content.into(), error: false, structured_content: None }
    }
    pub fn err(tool_call_id: impl Into<String>, name: impl Into<String>, msg: impl Into<String>) -> Self {
        Self { tool_call_id: tool_call_id.into(), name: name.into(), content: msg.into(), error: true, structured_content: None }
    }
    pub fn into_message(self) -> Message {
        Message::tool_result(self.tool_call_id, self.name, self.content)
    }
}

/// The output of one model completion turn.
#[derive(Debug, Clone)]
pub struct ModelTurn {
    pub text: String,
    pub tool_calls: Vec<ToolCall>,
    pub input_tokens: usize,
    pub output_tokens: usize,
}

impl ModelTurn {
    pub fn has_tool_calls(&self) -> bool { !self.tool_calls.is_empty() }
    pub fn total_tokens(&self) -> usize { self.input_tokens + self.output_tokens }
}

/// A definition of a tool that can be passed to the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDef {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

/// Configuration for a single model call.
#[derive(Debug, Clone)]
pub struct CallConfig {
    pub model: String,
    pub provider: String,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub max_steps_per_call: usize,
    pub max_depth: usize,
    pub max_total_tokens: Option<usize>,
    pub max_solve_seconds: f64,
}

impl Default for CallConfig {
    fn default() -> Self {
        Self {
            model: "claude-sonnet-4-20250514".to_string(),
            provider: "anthropic".to_string(),
            max_tokens: Some(4096),
            temperature: Some(0.0),
            max_steps_per_call: 20,
            max_depth: 3,
            max_total_tokens: None,
            max_solve_seconds: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn message_roles() {
        assert_eq!(Message::user("hi").role, "user");
        assert_eq!(Message::assistant("ok").role, "assistant");
        assert_eq!(Message::system("sys").role, "system");
    }

    #[test]
    fn tool_call_is_recursive() {
        let tc = ToolCall { id: "1".into(), name: "subtask".into(), arguments: HashMap::new() };
        assert!(tc.is_recursive());
    }

    #[test]
    fn tool_call_recursive_objective() {
        let mut args = HashMap::new();
        args.insert("objective".to_string(), serde_json::json!("write tests"));
        let tc = ToolCall { id: "1".into(), name: "subtask".into(), arguments: args };
        assert_eq!(tc.recursive_objective(), "write tests");
    }

    #[test]
    fn tool_result_ok_not_error() {
        let r = ToolResult::ok("id", "tool", "content");
        assert!(!r.error);
    }

    #[test]
    fn tool_result_err_is_error() {
        let r = ToolResult::err("id", "tool", "failed");
        assert!(r.error);
    }

    #[test]
    fn model_turn_total_tokens() {
        let t = ModelTurn { text: "ok".into(), tool_calls: vec![], input_tokens: 100, output_tokens: 50 };
        assert_eq!(t.total_tokens(), 150);
    }
}

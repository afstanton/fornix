//! Agent execution policy: write-before-read, repeated-shell, parallel write-path rules.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use crate::agent::types::ToolCall;

/// The outcome of a policy check before executing a tool call.
#[derive(Debug)]
pub struct PolicyDecision {
    /// If `Some`, the call is blocked and execution must not proceed.
    pub blocked_reason: Option<String>,
    /// Write-path lease granted for this call (released on completion).
    pub lease: Vec<PathBuf>,
}

impl PolicyDecision {
    pub fn allow(lease: Vec<PathBuf>) -> Self {
        Self { blocked_reason: None, lease }
    }
    pub fn block(reason: impl Into<String>) -> Self {
        Self { blocked_reason: Some(reason.into()), lease: Vec::new() }
    }
    pub fn is_blocked(&self) -> bool { self.blocked_reason.is_some() }
}

/// Per-solve-loop state that rules use to detect violations.
#[derive(Debug, Default)]
pub struct PolicyState {
    pub files_read: HashSet<String>,
    pub shell_commands: HashSet<String>,
}

/// Tool-call execution policy enforcing safety rules.
pub struct Policy {
    root: PathBuf,
    active_write_paths: Mutex<HashSet<String>>,
}

impl Policy {
    pub fn new(root: impl AsRef<Path>) -> Self {
        Self {
            root: root.as_ref().to_path_buf(),
            active_write_paths: Mutex::new(HashSet::new()),
        }
    }

    pub fn new_state(&self) -> PolicyState {
        PolicyState::default()
    }

    /// Evaluate the policy before a tool call.
    pub fn begin_tool_call(&self, call: &ToolCall, state: &PolicyState) -> PolicyDecision {
        // Rule 1: write-before-read
        if let Some(reason) = self.write_before_read_check(call, state) {
            return PolicyDecision::block(reason);
        }
        // Rule 2: repeated shell command
        if let Some(reason) = self.repeated_shell_check(call, state) {
            return PolicyDecision::block(reason);
        }
        // Acquire write-path lease
        let write_paths = self.target_write_paths(call);
        if write_paths.is_empty() {
            return PolicyDecision::allow(vec![]);
        }
        if self.reserve_write_paths(&write_paths) {
            PolicyDecision::allow(write_paths)
        } else {
            PolicyDecision::block(format!(
                "parallel write conflict blocked for {}",
                write_paths.first().map(|p| p.display().to_string()).unwrap_or_default()
            ))
        }
    }

    /// Release write-path leases and update per-session state after a call.
    pub fn end_tool_call(
        &self,
        call: &ToolCall,
        success: bool,
        state: &mut PolicyState,
        lease: &[PathBuf],
    ) {
        self.release_write_paths(lease);
        if !success { return; }
        if self.is_read_tool(call)
            && let Some(path) = self.path_arg(call) {
                state.files_read.insert(path.display().to_string());
            }
        if self.is_shell_tool(call)
            && let Some(cmd) = call.arg_str("command") {
                let cmd = cmd.trim().to_string();
                if !cmd.is_empty() {
                    state.shell_commands.insert(cmd);
                }
            }
    }

    // ─── private rules ───────────────────────────────────────────────────

    fn write_before_read_check(&self, call: &ToolCall, state: &PolicyState) -> Option<String> {
        if !self.is_write_tool(call) { return None; }
        let target = self.path_arg(call)?;
        if !target.exists() { return None; } // new file — no read required
        let key = target.display().to_string();
        if state.files_read.contains(&key) { return None; }
        Some(format!("write-before-read blocked for {}", key))
    }

    fn repeated_shell_check(&self, call: &ToolCall, state: &PolicyState) -> Option<String> {
        if !self.is_shell_tool(call) { return None; }
        let cmd = call.arg_str("command")?.trim().to_string();
        if cmd.is_empty() { return None; }
        if state.shell_commands.contains(&cmd) {
            Some(format!("repeated run_shell blocked for {}", cmd))
        } else {
            None
        }
    }

    fn target_write_paths(&self, call: &ToolCall) -> Vec<PathBuf> {
        match call.name.as_str() {
            "write_file" | "edit_file" | "hashline_edit" => {
                self.path_arg(call).into_iter().collect()
            }
            _ => Vec::new(),
        }
    }

    fn reserve_write_paths(&self, paths: &[PathBuf]) -> bool {
        let Ok(mut active) = self.active_write_paths.lock() else { return false };
        let path_strs: Vec<String> = paths.iter().map(|p| p.display().to_string()).collect();
        if path_strs.iter().any(|p| active.contains(p)) {
            return false;
        }
        for p in path_strs { active.insert(p); }
        true
    }

    fn release_write_paths(&self, paths: &[PathBuf]) {
        if let Ok(mut active) = self.active_write_paths.lock() {
            for p in paths { active.remove(&p.display().to_string()); }
        }
    }

    fn path_arg(&self, call: &ToolCall) -> Option<PathBuf> {
        let raw = call.arg_str("path")?;
        let resolved = self.root.join(raw).canonicalize()
            .unwrap_or_else(|_| self.root.join(raw));
        if resolved.starts_with(&self.root) { Some(resolved) } else { None }
    }

    fn is_read_tool(&self, call: &ToolCall) -> bool {
        matches!(call.name.as_str(), "read_file" | "read_image")
    }
    fn is_write_tool(&self, call: &ToolCall) -> bool {
        matches!(call.name.as_str(), "write_file" | "edit_file" | "hashline_edit" | "apply_patch")
    }
    fn is_shell_tool(&self, call: &ToolCall) -> bool {
        call.name == "run_shell"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call(name: &str, args: &[(&str, &str)]) -> ToolCall {
        ToolCall {
            id: "1".into(),
            name: name.to_string(),
            arguments: args.iter().map(|(k, v)| (k.to_string(), serde_json::json!(v))).collect(),
        }
    }

    fn policy() -> Policy {
        Policy::new(std::env::temp_dir())
    }

    #[test]
    fn unknown_tool_is_allowed() {
        let p = policy();
        let state = p.new_state();
        let d = p.begin_tool_call(&call("unknown_tool", &[]), &state);
        assert!(!d.is_blocked());
    }

    #[test]
    fn repeated_shell_command_is_blocked() {
        let p = policy();
        let mut state = p.new_state();
        let c = call("run_shell", &[("command", "ls -la")]);

        // First call — allowed
        let d1 = p.begin_tool_call(&c, &state);
        assert!(!d1.is_blocked());
        p.end_tool_call(&c, true, &mut state, &d1.lease);

        // Second identical call — blocked
        let d2 = p.begin_tool_call(&c, &state);
        assert!(d2.is_blocked());
        assert!(d2.blocked_reason.as_deref().unwrap().contains("repeated"));
    }

    #[test]
    fn different_shell_commands_not_blocked() {
        let p = policy();
        let mut state = p.new_state();
        let c1 = call("run_shell", &[("command", "ls")]);
        let c2 = call("run_shell", &[("command", "pwd")]);

        let d1 = p.begin_tool_call(&c1, &state);
        p.end_tool_call(&c1, true, &mut state, &d1.lease);

        let d2 = p.begin_tool_call(&c2, &state);
        assert!(!d2.is_blocked());
    }

    #[test]
    fn shell_tool_with_empty_command_not_blocked() {
        let p = policy();
        let state = p.new_state();
        let c = call("run_shell", &[("command", "")]);
        let d = p.begin_tool_call(&c, &state);
        assert!(!d.is_blocked());
    }

    #[test]
    fn policy_decision_allow_and_block() {
        let allow = PolicyDecision::allow(vec![]);
        assert!(!allow.is_blocked());

        let block = PolicyDecision::block("reason");
        assert!(block.is_blocked());
        assert_eq!(block.blocked_reason.as_deref(), Some("reason"));
    }

    #[test]
    fn new_state_has_empty_sets() {
        let p = policy();
        let state = p.new_state();
        assert!(state.files_read.is_empty());
        assert!(state.shell_commands.is_empty());
    }
}

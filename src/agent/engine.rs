//! Agent engine: the recursive solve loop.
//!
//! `Engine` orchestrates model calls, tool dispatch, sub-agent recursion,
//! token/step budgets, context condensation, and optional memory compaction.
//!
//! The LLM and tool registry are injected via the [`ModelClient`] and
//! [`ToolRegistry`] traits, keeping the engine fully provider-agnostic.
//! Wire up `rig`, `async-openai`, or a mock for testing.

use std::time::Instant;

use crate::agent::{
    error::{Error, Result},
    policy::{Policy, PolicyState},
    token_budget::TokenBudget,
    traits::{MemoryCompactor, ModelClient, ToolRegistry},
    types::{CallConfig, Message, ModelTurn, SerializedToolCall, ToolCall, ToolResult},
};

/// Condensed placeholder replacing large tool outputs once the context window
/// is approaching capacity.
const CONDENSE_PLACEHOLDER: &str =
    "[condensed tool output omitted to preserve context window]";

/// How many recent tool results to preserve un-condensed.
const RECENT_TOOL_RESULTS_TO_KEEP: usize = 4;

/// The agent engine.
pub struct Engine<M: ModelClient, T: ToolRegistry> {
    model: M,
    tools: T,
    policy: Policy,
    memory_compactor: Option<Box<dyn MemoryCompactor>>,
    system_prompt: String,
}

impl<M: ModelClient, T: ToolRegistry> Engine<M, T> {
    pub fn new(
        model: M,
        tools: T,
        policy: Policy,
        system_prompt: impl Into<String>,
    ) -> Self {
        Self {
            model,
            tools,
            policy,
            memory_compactor: None,
            system_prompt: system_prompt.into(),
        }
    }

    pub fn with_memory_compactor(mut self, compactor: Box<dyn MemoryCompactor>) -> Self {
        self.memory_compactor = Some(compactor);
        self
    }

    /// Execute the solve loop for `objective` under `config`.
    pub fn solve(
        &self,
        objective: &str,
        config: &CallConfig,
        on_event: Option<&dyn Fn(&str)>,
    ) -> Result<String> {
        let budget = TokenBudget::new();
        self.solve_recursive(objective, 0, config, &budget, on_event)
    }

    // ─── Core loop ───────────────────────────────────────────────────────

    fn solve_recursive(
        &self,
        objective: &str,
        depth: usize,
        config: &CallConfig,
        budget: &TokenBudget,
        on_event: Option<&dyn Fn(&str)>,
    ) -> Result<String> {
        if objective.trim().is_empty() {
            return Ok("No objective provided.".to_string());
        }
        if depth > config.max_depth {
            return Err(Error::MaxDepthExceeded(depth));
        }

        let mut conversation = vec![Message::user(objective)];
        let mut policy_state = self.policy.new_state();
        let deadline = if config.max_solve_seconds > 0.0 {
            Some(Instant::now() + std::time::Duration::from_secs_f64(config.max_solve_seconds))
        } else {
            None
        };

        let tool_defs = self.tools.tool_definitions();
        let mut cumulative_tokens = 0usize;
        let mut partial_answer = String::new();

        for step in 1..=config.max_steps_per_call {
            // Token budget check
            if budget.exceeded(config.max_total_tokens) {
                let used = budget.used();
                let max = config.max_total_tokens.unwrap_or(0);
                return Ok(format!(
                    "{}\n\n[budget exhausted: used {} tokens, max {}]",
                    partial_answer.trim(), used, max
                ));
            }

            // Time limit check
            if deadline.map_or(false, |d| Instant::now() > d) {
                return Err(Error::TimeLimitExceeded { steps: step - 1 });
            }

            emit(on_event, &format!("[d{}:s{}] calling model", depth, step));
            let turn = self.model.complete(&conversation, &tool_defs, config, &self.system_prompt)?;

            budget.add(turn.total_tokens());
            cumulative_tokens += turn.total_tokens();

            if !turn.text.trim().is_empty() {
                partial_answer = turn.text.clone();
            }

            if !turn.has_tool_calls() {
                emit(on_event, &format!("[d{}:s{}] final answer", depth, step));
                return Ok(turn.text);
            }

            // Append assistant turn with tool calls serialised
            let serialized = serialize_tool_calls(&turn);
            conversation.push(Message {
                role: "assistant".to_string(),
                content: turn.text.clone(),
                tool_calls: serialized,
                tool_call_id: None,
                name: None,
            });

            // Dispatch tool calls
            for tc in &turn.tool_calls {
                emit(on_event, &format!("[d{}:s{}] tool {}", depth, step, tc.name));

                let result = if tc.is_recursive() {
                    self.dispatch_recursive(tc, depth, config, budget, on_event)
                } else {
                    let decision = self.policy.begin_tool_call(tc, &policy_state);
                    if decision.is_blocked() {
                        let reason = decision.blocked_reason.clone().unwrap_or_default();
                        let r = ToolResult::err(&tc.id, &tc.name, format!("Policy blocked: {}", reason));
                        self.policy.end_tool_call(tc, false, &mut policy_state, &decision.lease);
                        r
                    } else {
                        let r = self.tools.execute(tc);
                        self.policy.end_tool_call(tc, !r.error, &mut policy_state, &decision.lease);
                        r
                    }
                };

                conversation.push(result.into_message());
            }

            // Condense old tool outputs if context is filling up
            condense_tool_results(&mut conversation, cumulative_tokens);

            // Optional memory compaction
            if let Some(compactor) = &self.memory_compactor {
                let window = 100_000usize; // caller provides real context window
                if compactor.should_compact(cumulative_tokens, window) {
                    if let Ok(result) = compactor.compact(&conversation, objective) {
                        if result.messages != conversation {
                            emit(on_event, "[memory_compacted] older conversation summarized");
                            conversation = result.messages;
                        }
                    }
                }
            }
        }

        Ok(format!(
            "Step budget exhausted. Try a narrower objective or higher max_steps_per_call.\
             \n\n{}",
            partial_answer.trim()
        ))
    }

    fn dispatch_recursive(
        &self,
        call: &ToolCall,
        depth: usize,
        config: &CallConfig,
        budget: &TokenBudget,
        on_event: Option<&dyn Fn(&str)>,
    ) -> ToolResult {
        let objective = call.recursive_objective();
        if objective.is_empty() {
            return ToolResult::err(&call.id, &call.name, "recursive objective cannot be empty");
        }
        if depth >= config.max_depth {
            return ToolResult::err(&call.id, &call.name, format!("max_depth reached at {}", depth));
        }

        // For `execute` always use the cheapest model; for `subtask` honour requested model
        let mut child_config = config.clone();
        if call.name == "execute" {
            child_config.model = cheap_model(&config.provider);
        } else if let Some(m) = call.arg_str("model") {
            if !m.trim().is_empty() {
                child_config.model = m.to_string();
            }
        }

        emit(on_event, &format!("[d{}] recurse {}", depth, call.name));
        match self.solve_recursive(&objective, depth + 1, &child_config, budget, on_event) {
            Ok(result) => ToolResult::ok(&call.id, &call.name, result),
            Err(e) => ToolResult::err(&call.id, &call.name, format!("Tool error: {}", e)),
        }
    }
}

// ─── helpers ─────────────────────────────────────────────────────────────────

fn emit(on_event: Option<&dyn Fn(&str)>, msg: &str) {
    if let Some(cb) = on_event { cb(msg); }
}

fn serialize_tool_calls(turn: &ModelTurn) -> Vec<SerializedToolCall> {
    turn.tool_calls.iter().map(|tc| SerializedToolCall {
        id: tc.id.clone(),
        call_type: "function".to_string(),
        function: crate::agent::types::FunctionCall {
            name: tc.name.clone(),
            arguments: serde_json::to_string(&tc.arguments).unwrap_or_default(),
        },
    }).collect()
}

fn condense_tool_results(conversation: &mut Vec<Message>, cumulative_tokens: usize) {
    let threshold = 75_000usize; // 75% of a rough 100K window
    if cumulative_tokens < threshold { return; }

    let tool_indices: Vec<usize> = conversation
        .iter()
        .enumerate()
        .filter(|(_, m)| m.role == "tool")
        .map(|(i, _)| i)
        .collect();

    let to_condense = if tool_indices.len() > RECENT_TOOL_RESULTS_TO_KEEP {
        &tool_indices[..tool_indices.len() - RECENT_TOOL_RESULTS_TO_KEEP]
    } else {
        return;
    };

    for &idx in to_condense {
        if conversation[idx].content != CONDENSE_PLACEHOLDER {
            conversation[idx].content = CONDENSE_PLACEHOLDER.to_string();
        }
    }
}

fn cheap_model(provider: &str) -> String {
    match provider {
        "anthropic" => "claude-haiku-4-5-20251001".to_string(),
        _ => "gpt-4.1-mini".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::{
        policy::Policy,
        traits::{ModelClient, ToolRegistry},
        types::{CallConfig, Message, ModelTurn, ToolCall, ToolDef, ToolResult},
    };
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    // ── Stub implementations ──

    struct EchoModel {
        calls: Arc<AtomicUsize>,
    }

    impl EchoModel {
        fn new() -> (Self, Arc<AtomicUsize>) {
            let c = Arc::new(AtomicUsize::new(0));
            (Self { calls: c.clone() }, c)
        }
    }

    impl ModelClient for EchoModel {
        fn complete(&self, messages: &[Message], _tools: &[ToolDef], _config: &CallConfig, _sys: &str) -> Result<ModelTurn> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            let content = messages.last().map(|m| m.content.as_str()).unwrap_or("done");
            Ok(ModelTurn {
                text: format!("Answer: {}", content),
                tool_calls: vec![],
                input_tokens: 10,
                output_tokens: 5,
            })
        }
    }

    struct NoopTools;
    impl ToolRegistry for NoopTools {
        fn tool_definitions(&self) -> Vec<ToolDef> { vec![] }
        fn execute(&self, call: &ToolCall) -> ToolResult { ToolResult::ok(&call.id, &call.name, "ok") }
    }

    fn engine() -> Engine<EchoModel, NoopTools> {
        let policy = Policy::new(std::env::temp_dir());
        let (model, _) = EchoModel::new();
        Engine::new(model, NoopTools, policy, "You are a helpful assistant.")
    }

    fn config() -> CallConfig {
        CallConfig { max_steps_per_call: 5, max_depth: 2, ..Default::default() }
    }

    // ── Tests ──

    #[test]
    fn solve_empty_objective_returns_message() {
        let e = engine();
        let r = e.solve("", &config(), None).unwrap();
        assert!(r.contains("No objective provided"));
    }

    #[test]
    fn solve_returns_model_answer() {
        let e = engine();
        let r = e.solve("What is 2+2?", &config(), None).unwrap();
        assert!(!r.is_empty());
    }

    #[test]
    fn solve_emits_events() {
        let e = engine();
        let mut events: Vec<String> = Vec::new();
        e.solve("Test", &config(), Some(&|ev| events.push(ev.to_string()))).unwrap();
        assert!(!events.is_empty());
        assert!(events[0].contains("calling model"));
    }

    #[test]
    fn token_budget_accumulates() {
        let policy = Policy::new(std::env::temp_dir());
        let (model, calls) = EchoModel::new();
        let e = Engine::new(model, NoopTools, policy, "sys");
        let budget = TokenBudget::new();
        e.solve_recursive("question", 0, &config(), &budget, None).unwrap();
        assert!(budget.used() > 0);
        assert_eq!(calls.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn condense_tool_results_preserves_recent() {
        let mut conv: Vec<Message> = (0..8).map(|i| {
            let mut m = Message::tool_result(format!("id{}", i), "t", format!("content{}", i));
            m
        }).collect();
        // Force condensation
        condense_tool_results(&mut conv, 100_000);
        let condensed = conv.iter().filter(|m| m.content == CONDENSE_PLACEHOLDER).count();
        let preserved = conv.len() - condensed;
        assert!(preserved >= RECENT_TOOL_RESULTS_TO_KEEP);
    }

    #[test]
    fn cheap_model_anthropic() {
        assert!(cheap_model("anthropic").contains("haiku") || cheap_model("anthropic").contains("claude"));
    }
}

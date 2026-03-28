//! A dynamically-typed value used in filters and configuration.
//!
//! Wraps `serde_json::Value` with a more ergonomic construction API
//! for use in filter expressions and adapter configuration.

pub use serde_json::Value;

/// Convenience constructors that read more clearly at call sites.
pub mod val {
    use serde_json::Value;

    pub fn str(s: impl Into<String>) -> Value {
        Value::String(s.into())
    }

    pub fn int(n: i64) -> Value {
        Value::Number(n.into())
    }

    pub fn float(n: f64) -> Value {
        serde_json::json!(n)
    }

    pub fn bool(b: bool) -> Value {
        Value::Bool(b)
    }

    pub fn null() -> Value {
        Value::Null
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value as JsonValue;

    #[test]
    fn str_produces_string_variant() {
        let v = val::str("hello");
        assert_eq!(v, JsonValue::String("hello".to_string()));
    }

    #[test]
    fn str_accepts_owned_string() {
        let v = val::str(String::from("world"));
        assert_eq!(v, JsonValue::String("world".to_string()));
    }

    #[test]
    fn int_produces_number_variant() {
        let v = val::int(42);
        assert_eq!(v, serde_json::json!(42_i64));
    }

    #[test]
    fn int_handles_negative() {
        let v = val::int(-7);
        assert_eq!(v, serde_json::json!(-7_i64));
    }

    #[test]
    fn int_handles_zero() {
        let v = val::int(0);
        assert_eq!(v, serde_json::json!(0_i64));
    }

    #[test]
    fn float_produces_number_variant() {
        let v = val::float(3.14);
        // serde_json represents floats as Number; compare via json! macro
        assert_eq!(v, serde_json::json!(3.14_f64));
    }

    #[test]
    fn bool_true() {
        assert_eq!(val::bool(true), JsonValue::Bool(true));
    }

    #[test]
    fn bool_false() {
        assert_eq!(val::bool(false), JsonValue::Bool(false));
    }

    #[test]
    fn null_produces_null_variant() {
        assert_eq!(val::null(), JsonValue::Null);
    }

    #[test]
    fn values_usable_in_serde_json_objects() {
        let obj = serde_json::json!({
            "name": val::str("alice"),
            "age": val::int(30),
            "active": val::bool(true),
            "score": val::float(0.95),
            "tag": val::null(),
        });
        assert_eq!(obj["name"], serde_json::json!("alice"));
        assert_eq!(obj["age"], serde_json::json!(30_i64));
        assert_eq!(obj["active"], serde_json::json!(true));
        assert_eq!(obj["tag"], JsonValue::Null);
    }
}

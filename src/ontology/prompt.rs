//! Ontology-guided LLM extraction prompt construction.
//!
//! [`OntologyPrompt`] builds the per-type guidance sections that are injected
//! into LLM extraction prompts. It mirrors `Cortex::Ontology::Prompt` in Ruby
//! and the `OntologySupport#ontology_prompt_for` /
//! `OntologySupport#ontology_relation_prompt_for` helpers in cortex-graphrag.
//!
//! Prompt text is intentionally terse — it is injected into a larger prompt
//! alongside the text chunk and other extraction instructions.

use crate::ontology::types::{Definition, EntityTypeDefinition, RelationTypeDefinition};

/// Builds ontology-guided prompt fragments for LLM extraction.
pub struct OntologyPrompt;

impl OntologyPrompt {
    /// Build a prompt guidance block for a single entity type.
    ///
    /// Returns `None` if `entity_type` is not in the ontology.
    /// The returned string is a terse multi-line block suitable for
    /// inclusion in an extraction prompt.
    ///
    /// # Example output
    ///
    /// ```text
    /// Entity type: Regulation
    /// Description: A codified rule or requirement in the CFR.
    /// Aliases: Provision, Rule
    /// Extraction patterns: \d+ CFR (CFR citation)
    /// Properties: cfr_citation (string, required), effective_date (date)
    /// ```
    pub fn build_entity_prompt(ontology: &Definition, entity_type: &str) -> Option<String> {
        let def = ontology.resolve_alias(entity_type)?;
        Some(Self::format_entity_block(def))
    }

    /// Build a prompt guidance block for a single relation type.
    ///
    /// Returns `None` if `relation_type` is not in the ontology.
    ///
    /// # Example output
    ///
    /// ```text
    /// Relation type: ISSUED_BY
    /// Description: A regulation issued by an agency.
    /// Allowed source types: Regulation
    /// Allowed target types: Agency
    /// Properties: authority_citation (string, required)
    /// ```
    pub fn build_relation_prompt(ontology: &Definition, relation_type: &str) -> Option<String> {
        let def = ontology.relation_type(relation_type)?;
        Some(Self::format_relation_block(def))
    }

    /// Return the subset of `requested` that are known entity type names
    /// in `ontology` (by canonical name or alias, resolved to canonical).
    ///
    /// If `requested` is empty, returns all entity type names in the ontology.
    pub fn scoped_entity_types<'a>(ontology: &'a Definition, requested: &[&str]) -> Vec<&'a str> {
        if requested.is_empty() {
            return ontology.entity_type_names().collect();
        }
        requested
            .iter()
            .filter_map(|&r| {
                ontology.resolve_alias(r).map(|t| t.name.as_str())
            })
            .collect()
    }

    /// Return the subset of `requested` that are known relation type names
    /// in `ontology`.
    ///
    /// If `requested` is empty, returns all relation type names in the ontology.
    pub fn scoped_relation_types<'a>(ontology: &'a Definition, requested: &[&str]) -> Vec<&'a str> {
        if requested.is_empty() {
            return ontology.relation_type_names().collect();
        }
        requested
            .iter()
            .filter_map(|&r| {
                ontology.relation_type(r).map(|t| t.name.as_str())
            })
            .collect()
    }

    // ── private formatting ────────────────────────────────────────────────────

    fn format_entity_block(def: &EntityTypeDefinition) -> String {
        let mut lines = vec![format!("Entity type: {}", def.name)];

        if let Some(desc) = &def.description {
            let d = desc.trim();
            if !d.is_empty() {
                lines.push(format!("Description: {}", d));
            }
        }

        if !def.aliases.is_empty() {
            lines.push(format!("Aliases: {}", def.aliases.join(", ")));
        }

        if !def.extraction_patterns.is_empty() {
            let patterns: Vec<String> = def.extraction_patterns.iter().map(|p| {
                match &p.description {
                    Some(d) => format!("{} ({})", p.pattern, d),
                    None => p.pattern.clone(),
                }
            }).collect();
            lines.push(format!("Extraction patterns: {}", patterns.join("; ")));
        }

        let prop_descs = Self::format_properties(&def.properties);
        if !prop_descs.is_empty() {
            lines.push(format!("Properties: {}", prop_descs));
        }

        lines.join("\n")
    }

    fn format_relation_block(def: &RelationTypeDefinition) -> String {
        let mut lines = vec![format!("Relation type: {}", def.name)];

        if let Some(desc) = &def.description {
            let d = desc.trim();
            if !d.is_empty() {
                lines.push(format!("Description: {}", d));
            }
        }

        if !def.source_types.is_empty() {
            lines.push(format!("Allowed source types: {}", def.source_types.join(", ")));
        }

        if !def.target_types.is_empty() {
            lines.push(format!("Allowed target types: {}", def.target_types.join(", ")));
        }

        let prop_descs = Self::format_properties(&def.properties);
        if !prop_descs.is_empty() {
            lines.push(format!("Properties: {}", prop_descs));
        }

        lines.join("\n")
    }

    fn format_properties(properties: &[crate::ontology::types::PropertyDefinition]) -> String {
        properties
            .iter()
            .map(|p| {
                if p.required {
                    format!("{} ({}, required)", p.name, p.data_type)
                } else {
                    format!("{} ({})", p.name, p.data_type)
                }
            })
            .collect::<Vec<_>>()
            .join(", ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ontology::types::{
        Definition, EntityTypeDefinition, ExtractionPattern, PropertyDefinition,
        RelationTypeDefinition,
    };

    fn regulatory_definition() -> Definition {
        let mut def = Definition::new("regulatory");
        def.entity_types = vec![
            EntityTypeDefinition {
                name: "Regulation".to_string(),
                description: Some("A codified rule in the CFR.".to_string()),
                extraction_strategy: Some("llm".to_string()),
                extraction_patterns: vec![
                    ExtractionPattern {
                        pattern: r"\d+ CFR".to_string(),
                        description: Some("CFR citation".to_string()),
                    },
                    ExtractionPattern {
                        pattern: r"§\s*\d+".to_string(),
                        description: None,
                    },
                ],
                aliases: vec!["Provision".to_string(), "Rule".to_string()],
                properties: vec![
                    PropertyDefinition::required("cfr_citation", "string"),
                    PropertyDefinition::optional("effective_date", "date"),
                ],
            },
            EntityTypeDefinition {
                name: "Agency".to_string(),
                description: None,
                extraction_strategy: None,
                extraction_patterns: Vec::new(),
                aliases: Vec::new(),
                properties: Vec::new(),
            },
        ];
        def.relation_types = vec![RelationTypeDefinition {
            name: "ISSUED_BY".to_string(),
            description: Some("Regulation issued by Agency.".to_string()),
            source_types: vec!["Regulation".to_string()],
            target_types: vec!["Agency".to_string()],
            properties: vec![PropertyDefinition::required("authority_citation", "string")],
        }];
        def
    }

    // ── build_entity_prompt ──

    #[test]
    fn entity_prompt_contains_type_name() {
        let def = regulatory_definition();
        let prompt = OntologyPrompt::build_entity_prompt(&def, "Regulation").unwrap();
        assert!(prompt.contains("Entity type: Regulation"));
    }

    #[test]
    fn entity_prompt_contains_description() {
        let def = regulatory_definition();
        let prompt = OntologyPrompt::build_entity_prompt(&def, "Regulation").unwrap();
        assert!(prompt.contains("A codified rule in the CFR."));
    }

    #[test]
    fn entity_prompt_contains_aliases() {
        let def = regulatory_definition();
        let prompt = OntologyPrompt::build_entity_prompt(&def, "Regulation").unwrap();
        assert!(prompt.contains("Provision"));
        assert!(prompt.contains("Rule"));
    }

    #[test]
    fn entity_prompt_contains_patterns() {
        let def = regulatory_definition();
        let prompt = OntologyPrompt::build_entity_prompt(&def, "Regulation").unwrap();
        assert!(prompt.contains("CFR citation"));
    }

    #[test]
    fn entity_prompt_contains_properties() {
        let def = regulatory_definition();
        let prompt = OntologyPrompt::build_entity_prompt(&def, "Regulation").unwrap();
        assert!(prompt.contains("cfr_citation"));
        assert!(prompt.contains("required"));
        assert!(prompt.contains("effective_date"));
    }

    #[test]
    fn entity_prompt_via_alias() {
        let def = regulatory_definition();
        let prompt = OntologyPrompt::build_entity_prompt(&def, "Provision").unwrap();
        assert!(prompt.contains("Entity type: Regulation"));
    }

    #[test]
    fn entity_prompt_returns_none_for_unknown() {
        let def = regulatory_definition();
        assert!(OntologyPrompt::build_entity_prompt(&def, "Unicorn").is_none());
    }

    #[test]
    fn entity_prompt_no_description_when_absent() {
        let def = regulatory_definition();
        let prompt = OntologyPrompt::build_entity_prompt(&def, "Agency").unwrap();
        assert!(!prompt.contains("Description:"));
    }

    #[test]
    fn entity_prompt_no_aliases_line_when_none() {
        let def = regulatory_definition();
        let prompt = OntologyPrompt::build_entity_prompt(&def, "Agency").unwrap();
        assert!(!prompt.contains("Aliases:"));
    }

    // ── build_relation_prompt ──

    #[test]
    fn relation_prompt_contains_type_name() {
        let def = regulatory_definition();
        let prompt = OntologyPrompt::build_relation_prompt(&def, "ISSUED_BY").unwrap();
        assert!(prompt.contains("Relation type: ISSUED_BY"));
    }

    #[test]
    fn relation_prompt_contains_description() {
        let def = regulatory_definition();
        let prompt = OntologyPrompt::build_relation_prompt(&def, "ISSUED_BY").unwrap();
        assert!(prompt.contains("Regulation issued by Agency."));
    }

    #[test]
    fn relation_prompt_contains_source_types() {
        let def = regulatory_definition();
        let prompt = OntologyPrompt::build_relation_prompt(&def, "ISSUED_BY").unwrap();
        assert!(prompt.contains("Allowed source types: Regulation"));
    }

    #[test]
    fn relation_prompt_contains_target_types() {
        let def = regulatory_definition();
        let prompt = OntologyPrompt::build_relation_prompt(&def, "ISSUED_BY").unwrap();
        assert!(prompt.contains("Allowed target types: Agency"));
    }

    #[test]
    fn relation_prompt_contains_properties() {
        let def = regulatory_definition();
        let prompt = OntologyPrompt::build_relation_prompt(&def, "ISSUED_BY").unwrap();
        assert!(prompt.contains("authority_citation"));
        assert!(prompt.contains("required"));
    }

    #[test]
    fn relation_prompt_returns_none_for_unknown() {
        let def = regulatory_definition();
        assert!(OntologyPrompt::build_relation_prompt(&def, "NONEXISTENT").is_none());
    }

    // ── scoped_entity_types ──

    #[test]
    fn scoped_entity_types_empty_returns_all() {
        let def = regulatory_definition();
        let types = OntologyPrompt::scoped_entity_types(&def, &[]);
        assert!(types.contains(&"Regulation"));
        assert!(types.contains(&"Agency"));
        assert_eq!(types.len(), 2);
    }

    #[test]
    fn scoped_entity_types_filters_to_known() {
        let def = regulatory_definition();
        let types = OntologyPrompt::scoped_entity_types(&def, &["Regulation", "Unicorn"]);
        assert_eq!(types, vec!["Regulation"]);
    }

    #[test]
    fn scoped_entity_types_resolves_alias() {
        let def = regulatory_definition();
        let types = OntologyPrompt::scoped_entity_types(&def, &["Provision"]);
        // Alias "Provision" resolves to canonical "Regulation"
        assert_eq!(types, vec!["Regulation"]);
    }

    #[test]
    fn scoped_entity_types_deduplicates_via_alias() {
        let def = regulatory_definition();
        // Both "Regulation" and its alias "Provision" resolve to "Regulation"
        let types = OntologyPrompt::scoped_entity_types(&def, &["Regulation", "Provision"]);
        // Both resolve to the same canonical name — result has two entries pointing to same name
        // (deduplication is the caller's responsibility)
        assert!(types.iter().all(|&t| t == "Regulation"));
    }

    // ── scoped_relation_types ──

    #[test]
    fn scoped_relation_types_empty_returns_all() {
        let def = regulatory_definition();
        let types = OntologyPrompt::scoped_relation_types(&def, &[]);
        assert!(types.contains(&"ISSUED_BY"));
        assert_eq!(types.len(), 1);
    }

    #[test]
    fn scoped_relation_types_filters_to_known() {
        let def = regulatory_definition();
        let types = OntologyPrompt::scoped_relation_types(&def, &["ISSUED_BY", "INVENTED_BY"]);
        assert_eq!(types, vec!["ISSUED_BY"]);
    }

    #[test]
    fn scoped_relation_types_unknown_only_returns_empty() {
        let def = regulatory_definition();
        let types = OntologyPrompt::scoped_relation_types(&def, &["MADE_UP"]);
        assert!(types.is_empty());
    }
}

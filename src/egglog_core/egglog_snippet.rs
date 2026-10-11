//! Egglog program fragments as data, and the registry-driven assembler.
//!
//! The op surface of the egglog program — every `Logical*` constructor,
//! every `LayoutTensorOp*` implementation constructor, and every rule
//! associated with an op — lives ALONGSIDE the op that owns it: one `.egg`
//! file per rule, `include_str!`-ed by the op's `snippets()` (the
//! `LogicalOp` registry for logical ops, the `OpMatcher` registry for
//! implementations). The CORE preamble (`egglog_preamble.egg`) keeps
//! everything shared — the IntExpr machinery, layouts, buffers,
//! certificates, the coordinate machinery, the terminal stratum — plus ten
//! `;; @SPLICE <category>` anchor lines marking exactly where each
//! category of op contribution is inserted.
//!
//! Anchor positions are DECLARATION-ORDER-SAFE by construction: each
//! anchor sits after every core declaration its category's rules may
//! reference (the scatter-seed lesson: egglog resolves names top-down).
//! Within a category, contributions splice in registry-registration
//! order; rule order within a ruleset is fixpoint-irrelevant, so the
//! assembled program is semantically identical to the old monolith — the
//! byte-stable goldens are the proof.

/// Where in the core preamble a snippet is spliced. One variant per
/// `;; @SPLICE` anchor, in the anchors' file order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpliceCategory {
    /// `(constructor Logical* ...)` declarations, after the datatype block.
    LogicalConstructors,
    /// `(constructor LayoutTensorOp* ...)` implementation declarations.
    LayoutOpConstructors,
    /// Dtype propagation rules and dtype tripwires.
    Dtype,
    /// Coordinate demand seeds and structural guards (rank-0 rejections,
    /// reduce axis-removal triggers) — after the coordinate machinery.
    Seed,
    /// Shape propagation rules — after `coords-common-shape`.
    Shape,
    /// Logical ring rewrites (commutativity, associativity, distributivity).
    Rewrites,
    /// Forward layout-tensor propagation and canonical mints.
    Forward,
    /// Implementation match rules (the `LayoutTensorOpLit` unions).
    Match,
    /// The gather⇄view unification machinery.
    Coordinate,
    /// Terminal-stratum rules (closures, locks, extent-lock seeds).
    Fixpoint,
}

impl SpliceCategory {
    fn marker(self) -> &'static str {
        match self {
            SpliceCategory::LogicalConstructors => ";; @SPLICE logical-constructors",
            SpliceCategory::LayoutOpConstructors => ";; @SPLICE layout-op-constructors",
            SpliceCategory::Dtype => ";; @SPLICE dtype",
            SpliceCategory::Seed => ";; @SPLICE seed",
            SpliceCategory::Shape => ";; @SPLICE shape",
            SpliceCategory::Rewrites => ";; @SPLICE rewrites",
            SpliceCategory::Forward => ";; @SPLICE forward",
            SpliceCategory::Match => ";; @SPLICE match",
            SpliceCategory::Coordinate => ";; @SPLICE coordinate",
            SpliceCategory::Fixpoint => ";; @SPLICE fixpoint",
        }
    }

    const ALL: [SpliceCategory; 10] = [
        SpliceCategory::LogicalConstructors,
        SpliceCategory::LayoutOpConstructors,
        SpliceCategory::Dtype,
        SpliceCategory::Seed,
        SpliceCategory::Shape,
        SpliceCategory::Rewrites,
        SpliceCategory::Forward,
        SpliceCategory::Match,
        SpliceCategory::Coordinate,
        SpliceCategory::Fixpoint,
    ];
}

/// One op contribution: verbatim egglog text (one rule or one constructor
/// declaration — one `.egg` file) bound for one splice anchor.
#[derive(Debug, Clone, Copy)]
pub struct EgglogSnippet {
    pub category: SpliceCategory,
    pub text: &'static str,
}

/// Splice contributions into the core at their anchors. Every anchor line
/// is kept (as a section header) and followed by its category's snippets
/// in the given order. Panics on a missing anchor — that is schema drift
/// between this module and the core preamble.
pub fn assemble(core: &str, snippets: &[EgglogSnippet]) -> String {
    let mut out = core.to_string();
    for category in SpliceCategory::ALL {
        let marker = category.marker();
        assert!(
            out.contains(marker),
            "core preamble is missing the {marker} anchor"
        );
        let mut section = String::from(marker);
        for snippet in snippets.iter().filter(|s| s.category == category) {
            section.push_str("\n\n");
            section.push_str(snippet.text.trim_end());
        }
        out = out.replacen(marker, &section, 1);
    }
    out
}

/// Where a frontend splices its own egglog text into a runtime's bound
/// program.
pub trait ProgramSeams {
    /// Text rendered after the model and boundary, before the schedule.
    fn before_schedule(&mut self, text: &str);
    /// A post-schedule unit: `text` runs after the schedule; `label` names
    /// it when a runtime isolates a failing unit.
    fn after_schedule(&mut self, label: &str, text: &str);
}

/// THE sanctioned `EGraph` constructor: registers the harness-side
/// primitives the core preamble contracts for. `(bigint-to-i64 z)` is the
/// bounds lattice's only way back out of BigInt — PARTIAL, unmatched
/// outside i64 range — and its one rule-side use is the [n,n] pin-collapse
/// rule in the bounds section. A bare `EGraph::default()` cannot even
/// parse the assembled program, so every runner goes through here.
pub fn new_egraph() -> egglog::EGraph {
    use egglog::sort::Z;

    let mut egraph = egglog::EGraph::default();
    // Tree decomposition is off for now: egglog's decomposition planner was
    // a large performance regression on our saturations (identical fixpoints
    // either way). Temporary until the planner is revisited.
    egraph.no_decomp = true;
    egglog::add_primitive!(&mut egraph, "bigint-to-i64" = |a: Z| -?> i64 {
        i64::try_from(&*a).ok()
    });
    // The vendored substitution primitive (egglog-experimental PR #60):
    // the preamble's subst road calls it from a :naive rule. The skip
    // set names the MEMO constructors the walk must treat as opaque
    // metadata rather than term structure (int-subst-of is the only
    // one — the transform-style census).
    egraph.add_full_primitive(
        crate::subst_primitive::Subst {
            skip: ["int-subst-of".to_string()].into_iter().collect(),
        },
        None,
    );
    // int-subst-of's root is statically IntExpr. Keep the generic API for
    // callers with other roots, but avoid unrelated constructor probes here.
    egraph.add_full_primitive(
        crate::subst_primitive::TypedSubst {
            name: "unstable-subst-int-expr",
            root_sort: "IntExpr",
            skip: ["int-subst-of".to_string()].into_iter().collect(),
        },
        None,
    );
    egraph
}

/// [`assembled_program`] for an arbitrary RUNTIME's matcher set — the
/// TestRuntime seam (ruling 2026-08-13: test-only op variants live in a
/// small tests-side runtime, and extraction takes the runtime's
/// matchers instead of hardcoding the reference registry). Logical-op
/// snippets are always included; the matcher contributions come from
/// the caller.
pub fn assembled_program_for(matchers: &[Box<dyn crate::layout_ir::OpMatcher>]) -> String {
    let core = include_str!("egglog_preamble.egg");
    let mut snippets: Vec<EgglogSnippet> = Vec::new();
    for op in crate::logical_op::built_in_logical_ops() {
        snippets.extend(op.snippets());
    }
    for helper in crate::logical_helper::built_in_logical_helpers() {
        snippets.extend(helper.snippets());
    }
    for matcher in matchers {
        snippets.extend(matcher.snippets());
    }
    assemble(core, &snippets)
}

/// WHAT CORE'S OWN PREAMBLE DECLARES AND DECODES: the five `Layout`
/// constructors. The declaration and its decoder travel together — this
/// is the `core` half of that pairing, the matcher's
/// [`crate::layout_ir::OpMatcher::decoders`] is the other.
pub fn core_decoders() -> Vec<crate::egglog_utils::eclass::ConstructorDecoder> {
    crate::layouts::layout_decoders()
}

/// The decoder registry for a runtime's matcher set — the same shape as
/// [`assembled_program_for`]: core's built-ins first, then each
/// matcher's contribution in registry order. A duplicate
/// `(sort, constructor)` is REFUSED (registering two decoders for one
/// constructor is a bug, never a preference).
pub fn decoder_registry_for(
    matchers: &[Box<dyn crate::layout_ir::OpMatcher>],
) -> anyhow::Result<crate::egglog_utils::eclass::ConstructorRegistry> {
    crate::egglog_utils::eclass::ConstructorRegistry::new(
        core_decoders()
            .into_iter()
            .chain(matchers.iter().flat_map(|matcher| matcher.decoders())),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Splicing inserts at the anchor, keeps the anchor as a header, and
    /// preserves contribution order within a category.
    #[test]
    fn splices_in_order_at_the_anchor() {
        let core = "top\n;; @SPLICE logical-constructors\n\n;; @SPLICE layout-op-constructors\n\n;; @SPLICE dtype\n\n;; @SPLICE seed\n\n;; @SPLICE shape\n\n;; @SPLICE rewrites\n\n;; @SPLICE forward\n\n;; @SPLICE match\n\n;; @SPLICE coordinate\n\n;; @SPLICE fixpoint\nbottom\n";
        let snippets = [
            EgglogSnippet {
                category: SpliceCategory::Dtype,
                text: "(rule-a)\n",
            },
            EgglogSnippet {
                category: SpliceCategory::Dtype,
                text: "(rule-b)\n",
            },
            EgglogSnippet {
                category: SpliceCategory::Fixpoint,
                text: "(rule-c)\n",
            },
        ];
        let out = assemble(core, &snippets);
        assert!(out.contains(";; @SPLICE dtype\n\n(rule-a)\n\n(rule-b)"));
        assert!(out.contains(";; @SPLICE fixpoint\n\n(rule-c)\nbottom"));
    }

    /// The assembled program parses cleanly end to end — every
    /// declaration precedes its uses, every included `.egg` file is
    /// balanced. Core owns no runtime registry (Step B), so this proves
    /// the core preamble + logical-op snippets alone; the reference
    /// registry's assembly is proven by luminal_reference's suites,
    /// which run its program everywhere.
    #[test]
    fn assembled_program_is_loadable() {
        new_egraph()
            .parse_and_run_program(None, &assembled_program_for(&[]))
            .expect("assembled program loads");
    }

    /// The `prop` ruleset holds exactly the shape-of / dtype-of / rank-of
    /// propagation rules: every default-ruleset rule does something else,
    /// and every `prop` rule does nothing else. The schedule runs `prop`
    /// to saturation before the main loop and alongside `(run main_ruleset)` inside
    /// it, so a propagation rule left in the default ruleset would lose
    /// the pre-pass and a minting rule tagged `prop` would run in it.
    #[test]
    fn prop_ruleset_is_exactly_the_propagation_rules() {
        use egglog::ast::{Command, GenericAction};
        const PROP: [&str; 3] = ["shape-of", "dtype-of", "rank-of"];
        let only_propagation = |rule: &egglog::ast::Rule| {
            !rule.head.0.is_empty()
                && rule.head.0.iter().all(|action| match action {
                    GenericAction::Set(_, head, _, _) => PROP.contains(&head.as_str()),
                    _ => false,
                })
        };
        let commands = new_egraph()
            .parse_program(None, &assembled_program_for(&[]))
            .expect("assembled program parses");
        let mut prop_rules = 0;
        for command in &commands {
            let Command::Rule { rule } = command else {
                continue;
            };
            match rule.ruleset.as_str() {
                "prop" => {
                    prop_rules += 1;
                    assert!(
                        only_propagation(rule),
                        "prop rule does more than propagate: {rule}"
                    );
                }
                "" => assert!(
                    !only_propagation(rule),
                    "propagation rule missing `:ruleset prop`: {rule}"
                ),
                _ => {}
            }
        }
        assert!(prop_rules > 0, "no prop rules found");
    }

    // Compare the accelerated rules against exactly the retained one-hop
    // program. All syntax, predicates, unions and subsumption are in egglog.
    fn one_hop_program() -> String {
        let mut text = assembled_program_for(&[]);
        for name in [
            "buffer-list-jump",
            "layout-list-jump",
            "logical-list-jump",
            "expr-list-jump",
        ] {
            let start = text.find(&format!("; BEGIN {name} doubling")).unwrap();
            let marker = format!("; END {name} doubling");
            let end = text.find(&marker).unwrap() + marker.len();
            text.replace_range(start..end, "");
        }
        text
    }

    fn run_walk_fixture(accelerated: bool, seed: &str, schedule: &str, checks: &str) -> usize {
        let mut graph = new_egraph();
        let program = if accelerated {
            assembled_program_for(&[])
        } else {
            one_hop_program()
        };
        graph.parse_and_run_program(None, &program).unwrap();
        graph.parse_and_run_program(None, seed).unwrap();
        graph.parse_and_run_program(None, schedule).unwrap();
        let rounds = graph.get_overall_run_report().iterations.len();
        graph.parse_and_run_program(None, checks).unwrap();
        if schedule.contains("run cleanup") {
            let mut rows = 0;
            graph
                .constructor_enodes("LayoutTensorOpLit", |node| {
                    assert!(node.subsumed, "input producer spelling must be retired");
                    rows += 1;
                })
                .unwrap();
            assert_eq!(rows, 1);
        }
        rounds
    }

    #[test]
    fn doubled_cleanup_preserves_long_list_facts_and_retires_input_producers() {
        let n = 512;
        let mut seed = String::from(
            r#"
            (let sh (ShapeLit (IntExprNil)))
            (let layout (RightMajorContiguousElementLayoutLit sh (BitWidthLit 32)))
            (let leaf (LayoutTensorLit (LogicalTensorInputLit (LogicalIdLit "bound") sh (F32)) layout))
            (let other (LayoutTensorLit (LogicalTensorInputLit (LogicalIdLit "unbound") sh (F32)) layout))
            (let bt (BufferTensorLit leaf (BufferLit 0)))
            (let b0 (BufferTensorNil))
            (let l0 (LayoutTensorNil))
            (let l1 (LayoutTensorCons leaf l0))
        "#,
        );
        for i in 1..=n {
            seed.push_str(&format!("(let b{i} (BufferTensorCons bt b{}))\n", i - 1));
            if i > 1 {
                seed.push_str(&format!("(let l{i} (LayoutTensorCons other l{}))\n", i - 1));
            }
        }
        seed.push_str(&format!(
            "(let binding (BufferInputLit b{n})) (let op (LayoutTensorOpLit l0 l{n}))"
        ));
        let mut checks =
            String::from("(check (input-producer op)) (check (input-leaf-layout-tensor leaf))");
        checks.push_str("(fail (check (input-leaf-layout-tensor other)))");
        for i in 0..=n {
            checks.push_str(&format!("(check (input-buffer-tensor-list b{i}))\n"));
        }
        for i in 1..=n {
            checks.push_str(&format!("(check (list-holds-input-layout-tensor l{i}))\n"));
        }

        let slow = run_walk_fixture(
            false,
            &seed,
            "(run-schedule (saturate (run cleanup)))",
            &checks,
        );
        let fast = run_walk_fixture(
            true,
            &seed,
            "(run-schedule (saturate (run cleanup)))",
            &checks,
        );
        eprintln!("list propagation: one-hop={slow}, doubled={fast} rounds at length={n}");
        assert!(slow > n as usize);
        assert!(
            fast < 40,
            "doubling should not take one round per cons: {fast}"
        );
    }

    #[test]
    fn doubled_terminal_walks_preserve_paired_suffixes_and_shape_locks() {
        let n = 257;
        let mut seed = String::from(
            r#"
            (let sh (ShapeLit (IntExprNil)))
            (let out (LogicalTensorInputLit (LogicalIdLit "out") sh (F32)))
            (let coord (LogicalTensorInputLit (LogicalIdLit "coord") sh (F32)))
            (set (shape-of out) sh) (set (shape-of coord) sh)
            (let c0 (LogicalTensorNil)) (let e0 (IntExprNil)) (let d0 (IntExprNil))
        "#,
        );
        let mut checks = String::new();
        for i in 1..=n {
            seed.push_str(&format!("(let c{i} (LogicalTensorCons coord c{})) (let e{i} (IntExprCons (IntLit 0) e{})) (let d{i} (IntExprCons (IntLit 1) d{}))\n", i-1,i-1,i-1));
        }
        seed.push_str(&format!(
            "(coordinate-shape-lock out c{n}) (map-range-walk e{n} d{n})"
        ));
        for i in 0..=n {
            checks.push_str(&format!(
                "(check (coordinate-shape-lock out c{i})) (check (map-range-walk e{i} d{i}))\n"
            ));
        }
        let schedule = "(run-schedule (saturate (run fixpoint-invariants)))";
        let slow = run_walk_fixture(false, &seed, schedule, &checks);
        let fast = run_walk_fixture(true, &seed, schedule, &checks);
        eprintln!("list propagation: one-hop={slow}, doubled={fast} rounds at length={n}");
        assert!(slow > n as usize);
        assert!(fast < 40, "terminal paths must advance in batches: {fast}");
        // Mismatched lengths stop at the same paired suffix. A negative entry
        // beyond the short dimension list must not acquire a phantom extent.
        let uneven = "(let bad (IntExprCons (IntLit -1) e0)) (let entries (IntExprCons (IntLit 0) bad)) (map-range-walk entries d1)";
        for accelerated in [false, true] {
            run_walk_fixture(
                accelerated,
                &format!("{seed} {uneven}"),
                schedule,
                "(check (map-range-walk bad d0))",
            );
        }
    }

    #[test]
    fn doubled_walks_terminate_on_cyclic_and_aliased_list_spellings() {
        let seed = r#"
            (let sh (ShapeLit (IntExprNil)))
            (let lt (LayoutTensorLit (LogicalTensorInputLit (LogicalIdLit "x") sh (F32)) (RightMajorContiguousElementLayoutLit sh (BitWidthLit 32))))
            (let bt (BufferTensorLit lt (BufferLit 0)))
            (let b (BufferTensorCons bt (BufferTensorNil)))
            (let b2 (BufferTensorCons bt b)) (union b b2)
            (let l (LayoutTensorCons lt (LayoutTensorNil)))
            (let l2 (LayoutTensorCons lt l)) (union l l2)
            (let binding (BufferInputLit b))
            (let op (LayoutTensorOpLit (LayoutTensorNil) l))
        "#;
        for accelerated in [false, true] {
            let rounds = run_walk_fixture(
                accelerated,
                seed,
                "(run-schedule (saturate (run cleanup)))",
                "(check (input-producer op)) (check (input-buffer-tensor-list b2))",
            );
            assert!(
                rounds < 40,
                "bounded auxiliary links must terminate on cycles"
            );
        }
    }

    #[test]
    fn doubled_range_walk_still_rejects_a_violation_at_the_last_entry() {
        for accelerated in [false, true] {
            let mut graph = new_egraph();
            graph
                .parse_and_run_program(
                    None,
                    &if accelerated {
                        assembled_program_for(&[])
                    } else {
                        one_hop_program()
                    },
                )
                .unwrap();
            let mut seed = String::from(
                "(let e0 (IntExprCons (IntLit -1) (IntExprNil))) (let d0 (IntExprCons (IntLit 1) (IntExprNil)))",
            );
            for i in 1..=65 {
                seed.push_str(&format!("(let e{i} (IntExprCons (IntLit 0) e{})) (let d{i} (IntExprCons (IntLit 1) d{}))",i-1,i-1));
            }
            seed.push_str("(map-range-walk e65 d65)");
            graph.parse_and_run_program(None, &seed).unwrap();
            let err = graph
                .parse_and_run_program(None, "(run-schedule (saturate (run fixpoint-invariants)))")
                .unwrap_err();
            assert!(err.to_string().contains("negative literal entry"), "{err}");
        }
    }
}

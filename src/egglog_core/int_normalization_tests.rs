//! Termination and semantic regressions for integer offset normalization.

use luminal::{egglog_snippet, prelude::egglog};

fn graph(seed: &str) -> egglog::EGraph {
    let mut graph = egglog_snippet::new_egraph();
    graph
        .parse_and_run_program(None, &egglog_snippet::assembled_program_for(&[]))
        .unwrap();
    graph.parse_and_run_program(None, seed).unwrap();
    graph
}

// Bound each round and the number of rounds, so the unfixed regression fails
// before the associativity query itself can allocate gigabytes of matches.
fn saturate_main(graph: &mut egglog::EGraph) -> bool {
    for _ in 0..64 {
        let report = graph
            .parse_and_run_program(None, "(run-schedule (run main_ruleset) (run prop))")
            .unwrap();
        assert!(graph.num_tuples() < 8_000, "integer normalization grew past 8,000 rows");
        let updated = report.iter().any(|output| {
            matches!(output, egglog::CommandOutput::RunSchedule(report) if report.updated)
        });
        if !updated {
            return true;
        }
    }
    false
}

#[test]
fn shifted_coordinate_cancellation_reaches_a_fixpoint() {
    let mut graph = graph(
        "(let shape (ShapeLit (IntExprCons (IntLit 8) (IntExprNil))))
         (let x (CoordVar shape 0))
         (let shifted (IntAdd x (IntLit 1)))
         (let restored (IntAdd shifted (IntLit -1)))",
    );
    assert!(saturate_main(&mut graph), "offset cancellation did not converge");
    graph.parse_and_run_program(None, "(check (= restored x))").unwrap();
}

#[test]
fn substitution_of_a_shifted_slice_reaches_a_fixpoint() {
    let mut graph = graph(
        "(let source (ShapeLit (IntExprCons (IntLit 8) (IntExprNil))))
         (let output (ShapeLit (IntExprCons (IntLit 4) (IntExprNil))))
         (let p (CoordVar source 0))
         (let q (CoordVar output 0))
         (let parent (IntAdd p (IntLit -1)))
         (let map (IndexMapLit
             (IntExprCons (IntAdd q (IntLit 1)) (IntExprNil)) source))
         (int-subst-demand parent map)",
    );
    // The input is benign. The walk introduces (q + 1) + (-1).
    assert!(saturate_main(&mut graph));
    graph.parse_and_run_program(None, "(run-schedule (run subst-walk))").unwrap();
    assert!(saturate_main(&mut graph), "post-substitution cancellation did not converge");
    graph
        .parse_and_run_program(None, "(check (= (int-subst-of parent map) q))")
        .unwrap();
}

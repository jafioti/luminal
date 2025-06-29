use colored::Colorize;
use egraph_serialize::{ClassId, EGraph, NodeId};
use itertools::Itertools;
use petgraph::prelude::{NodeIndex, StableGraph};
use petgraph::{Directed, Direction};
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::HashSet;

const WARMUP_TRIALS: usize = 1;
const TRIALS: usize = 3;
const MAX_SEARCHED_GRAPHS: usize = 1_000;

type Cost = u128; // Execution time in microseconds

use std::collections::HashMap;

/// Enumerate every possible extraction and get the runtime cost for each
pub fn search(egraph: &EGraph, inputs: &[Vec<f32>]) {
    // class → nodes
    let mut class_nodes: FxHashMap<ClassId, Vec<NodeId>> = FxHashMap::default();
    for (nid, node) in &egraph.nodes {
        class_nodes
            .entry(node.eclass.clone())
            .or_default()
            .push(nid.clone());
    }

    // reachable classes
    let mut reach = FxHashSet::default();
    let mut stack: Vec<_> = egraph.root_eclasses.iter().collect_vec();
    while let Some(cid) = stack.pop() {
        if reach.insert(cid.clone())
            && let Some(ns) = class_nodes.get(&cid)
        {
            for nid in ns {
                stack.extend(
                    egraph.nodes[nid]
                        .children
                        .iter()
                        .map(|c| egraph.nid_to_cid(c)),
                );
            }
        }
    }

    let mut classes: Vec<_> = reach.into_iter().collect();
    classes.sort();

    // depth-first cartesian product
    fn dfs(
        idx: usize,
        classes: &[ClassId],
        class_nodes: &FxHashMap<ClassId, Vec<NodeId>>,
        cur: &mut FxHashMap<ClassId, NodeId>,
        accepted: &mut usize,
        eg: &EGraph,
        inputs: &[Vec<f32>],
    ) {
        if *accepted >= MAX_SEARCHED_GRAPHS {
            return;
        }
        if idx == classes.len() {
            let er = ExtractionResult {
                choices: cur.clone(),
            };
            if !extraction_contains_cycle(eg, &er) {
                if let Some(c) = cost(&er, eg, inputs) {
                    println!(
                        "{}{}",
                        format!("Graph {accepted} ").bold(),
                        format!("{c}µs").bright_green().bold()
                    );
                    *accepted += 1;
                }
            }
            return;
        }
        let cid = &classes[idx];
        for nid in &class_nodes[cid] {
            cur.insert(cid.clone(), nid.clone());
            dfs(idx + 1, classes, class_nodes, cur, accepted, eg, inputs);
        }
        cur.remove(cid);
    }

    dfs(
        0,
        &classes,
        &class_nodes,
        &mut FxHashMap::default(),
        &mut 0,
        egraph,
        inputs,
    );
}

#[derive(Default, Clone)]
pub struct ExtractionResult {
    pub choices: FxHashMap<ClassId, NodeId>,
}

pub fn extraction_contains_cycle(egraph: &EGraph, extraction: &ExtractionResult) -> bool {
    use std::collections::HashMap;

    #[derive(Copy, Clone)]
    enum Mark {
        Temp,
        Perm,
    } // DFS colours

    let mut marks: HashMap<ClassId, Mark> = HashMap::default();

    fn dfs(
        cid: &ClassId,
        egraph: &EGraph,
        ex: &ExtractionResult,
        marks: &mut HashMap<ClassId, Mark>,
    ) -> bool {
        match marks.get(cid) {
            Some(Mark::Perm) => return false, // already finished, no cycle here
            Some(Mark::Temp) => return true,  // back-edge ⇒ cycle
            None => {}
        }
        marks.insert(cid.clone(), Mark::Temp);

        if let Some(nid) = ex.choices.get(cid)
            && let Some(node) = egraph.nodes.get(nid)
        {
            // if node.op == "SwapLoops" || node.op == "Unary" || node.op == "Binary" {
            //     return true;
            // }
            // `children()` should yield the child ClassIds of this node
            for child in &node.children {
                let child_cid = egraph.nid_to_cid(child);
                // Only follow edges that stay inside the chosen sub-graph
                if ex.choices.contains_key(child_cid) && dfs(child_cid, egraph, ex, marks) {
                    return true;
                }
            }
        }

        marks.insert(cid.clone(), Mark::Perm);
        false
    }

    // Cycle exists if any DFS starting at a chosen class finds one
    extraction
        .choices
        .keys()
        .any(|cid| dfs(cid, egraph, extraction, &mut marks))
}

fn cost(extraction: &ExtractionResult, egraph: &EGraph, inputs: &[Vec<f32>]) -> Option<Cost> {
    // Convert to a petgraph and find the root node
    let graph = extraction_to_graph(egraph, extraction, &egraph.root_eclasses)?;
    let root = graph.externals(Direction::Outgoing).next().unwrap();
    // Codegen
    let kernels = crate::codegen::codegen(graph, root, GPUArch::Metal(HashMap::new()));
    // Warm up resources (buffer allocation, kernel compiler, etc.)
    for _ in 0..WARMUP_TRIALS {
        run_graph(inputs, &kernels);
    }
    // Test runtime
    let mut micros = vec![];
    for _ in 0..TRIALS {
        let (_outputs, m) = run_graph(inputs, &kernels);
        micros.push(m);
    }
    Some(micros.into_iter().sum::<u128>() / TRIALS as u128)
}

/// Build a StableGraph from an ExtractionResult.
///
/// * `extraction.choices` : HashMap<ClassId, NodeId>
/// * `roots`              : entry-point e-classes
///
/// Any helper like `math_from_class()` can be filled in later to turn
/// Loop ranges / strides into your own `Expression` type.
pub fn extraction_to_graph(
    egraph: &EGraph,
    extraction: &ExtractionResult,
    roots: &[ClassId],
) -> Option<StableGraph<GraphTerm, u8, Directed>> {
    // display_graph(&extraction_to_petgraph(egraph, extraction), &[]);
    let mut g: StableGraph<GraphTerm, u8, Directed> = StableGraph::new();

    fn emit<'a>(
        nid: &'a NodeId,
        egraph: &EGraph,
        extraction: &'a ExtractionResult,
        g: &mut StableGraph<GraphTerm, u8, Directed>,
        seen: &mut HashSet<&'a NodeId>,
    ) -> Option<NodeIndex> {
        if seen.contains(nid) {
            return None;
        }
        seen.insert(nid);
        let enode = &egraph.nodes[nid];
        match enode.op.as_str() {
            "GMEM" => Some(g.add_node(GraphTerm::GMEM { label: None })),
            "SMEM" => Some(g.add_node(GraphTerm::SMEM)),
            "SMEMLoad" => Some(g.add_node(GraphTerm::SMEMLoad)),
            "SMEMRead" => Some(g.add_node(GraphTerm::SMEMRead)),
            "NewAcc" => Some(g.add_node(GraphTerm::NewAcc {
                starting_value: "acc".into(),
            })),

            // LoopIn  = (LoopIn <expr> <LoopType> <Math>)
            "LoopIn" => {
                let range =
                    convert_math(egraph.nid_to_cid(&enode.children[1]), egraph, extraction)?;
                let stride =
                    convert_math(egraph.nid_to_cid(&enode.children[2]), egraph, extraction)?;
                let n = g.add_node(GraphTerm::LoopIn { range, stride });
                let child = emit(
                    &extraction.choices[egraph.nid_to_cid(&enode.children[0])],
                    egraph,
                    extraction,
                    g,
                    seen,
                )?;
                g.add_edge(child, n, 0);
                Some(n)
            }

            // LoopOut = same child layout as LoopIn
            "LoopOut" => {
                let range =
                    convert_math(egraph.nid_to_cid(&enode.children[1]), egraph, extraction)?;
                let stride =
                    convert_math(egraph.nid_to_cid(&enode.children[2]), egraph, extraction)?;

                let n = g.add_node(GraphTerm::LoopOut { range, stride });
                let child = emit(
                    &extraction.choices[egraph.nid_to_cid(&enode.children[0])],
                    egraph,
                    extraction,
                    g,
                    seen,
                )?;
                g.add_edge(child, n, 0);
                Some(n)
            }

            "Add" | "Mul" | "Max" => {
                let n = g.add_node(match enode.op.as_str() {
                    "Add" => GraphTerm::Add,
                    "Mul" => GraphTerm::Mul,
                    "Max" => GraphTerm::Max,
                    _ => panic!(),
                });
                let a = emit(
                    &extraction.choices[egraph.nid_to_cid(&enode.children[0])],
                    egraph,
                    extraction,
                    g,
                    seen,
                )?;
                let b = emit(
                    &extraction.choices[egraph.nid_to_cid(&enode.children[1])],
                    egraph,
                    extraction,
                    g,
                    seen,
                )?;
                g.add_edge(a, n, 0);
                g.add_edge(b, n, 0);
                Some(n)
            }
            "Exp" | "Sin" | "Recip" | "Neg" => {
                let n = g.add_node(match enode.op.as_str() {
                    "Exp" => GraphTerm::Exp,
                    "Sin" => GraphTerm::Sin,
                    "Recip" => GraphTerm::Recip,
                    "Neg" => GraphTerm::Neg,
                    _ => panic!(),
                });
                let a = emit(
                    &extraction.choices[egraph.nid_to_cid(&enode.children[0])],
                    egraph,
                    extraction,
                    g,
                    seen,
                )?;
                g.add_edge(a, n, 0);
                Some(n)
            }
            "Unary" | "Binary" | "SwapLoops" => return None,
            _ => unreachable!("unsupported op {}", enode.op),
        }
    }

    for root_cid in roots {
        if let Some(root_nid) = extraction.choices.get(root_cid) {
            emit(root_nid, egraph, extraction, &mut g, &mut HashSet::new())?;
        }
    }
    Some(g)
}

use crate::symbolic::{Expression, Term};
use crate::{GPUArch, GraphTerm, run::run_graph};

fn convert_math(
    cid: &ClassId,
    egraph: &EGraph,
    extraction: &ExtractionResult,
) -> Option<Expression> {
    // Memoise by NodeId so we don’t re-convert shared sub-expressions.
    fn build(
        nid: NodeId,
        egraph: &EGraph,
        extraction: &ExtractionResult,
        memo: &mut HashMap<NodeId, Expression>,
    ) -> Option<Expression> {
        if let Some(m) = memo.get(&nid) {
            return Some(*m);
        }

        let enode = &egraph.nodes[&nid];
        let make_child =
            |child_cid: &ClassId, memo: &mut HashMap<NodeId, Expression>| -> Option<Expression> {
                let child_nid = extraction.choices[child_cid].clone();
                build(child_nid, egraph, extraction, memo)
            };
        let term = match enode.op.as_str() {
            // ----------- literals & vars -----------
            op if op.starts_with("MNum:") => {
                let num: i64 = op["MNum:".len()..].parse().expect("invalid MNum literal");
                Expression::from(num as usize)
            }
            op if op.starts_with("MVar:") => {
                let name = op["MVar:".len()..].to_owned();
                Expression::from(name.chars().next().unwrap())
            }
            op if op.starts_with("Boxed(\"") => {
                let name = op.replace("Boxed(\"", "").replace("\")", "");
                // if name.len() == 1 {
                Expression::from(name.chars().next().unwrap())
                // } else {
                //     panic!("Variable name too long: {name}")
                // }
            }

            // ----------- unary ops -----------
            "MNeg" | "MRecip" => {
                let c0 = make_child(&egraph.nid_to_cid(&enode.children[0]), memo)?;
                match enode.op.as_str() {
                    "MNeg" => c0 * -1,
                    "MRecip" => 1 / c0,
                    _ => unreachable!(),
                }
            }

            // ----------- binary ops -----------
            "MAdd" | "MSub" | "MMul" | "MDiv" | "MMod" | "MMin" | "MMax" | "MAnd" | "MOr"
            | "MGte" | "MLt" | "MFloorTo" => {
                let lhs = make_child(&egraph.nid_to_cid(&enode.children[0]), memo)?;
                let rhs = make_child(&egraph.nid_to_cid(&enode.children[1]), memo)?;
                match enode.op.as_str() {
                    "MAdd" => lhs + rhs,
                    "MSub" => lhs - rhs,
                    "MMul" => lhs * rhs,
                    "MDiv" => lhs / rhs,
                    "MMod" => lhs % rhs,
                    "MMin" => lhs.min(rhs),
                    "MMax" => lhs.max(rhs),
                    "MAnd" => lhs & rhs,
                    "MOr" => lhs | rhs,
                    "MFloorTo" => todo!(),
                    _ => unreachable!(),
                }
            }

            // ----------- ternary -----------
            "MReplace" => return None,

            // ----------- accumulator marker -----------
            "MAccum" => {
                let name = if enode.children.is_empty() {
                    "<acc>".to_owned()
                } else {
                    "<acc>".to_owned()
                };
                Expression::from(Term::Acc(name.chars().next().unwrap()))
            }
            "Loop" => make_child(&egraph.nid_to_cid(&enode.children[1]), memo)?,
            "MNum" | "MVar" => make_child(&egraph.nid_to_cid(&enode.children[0]), memo)?,
            _ => {
                if let Ok(n) = enode.op.parse::<usize>() {
                    Expression::from(n)
                } else {
                    panic!("unsupported Math op '{}'", enode.op)
                }
            }
        };

        memo.insert(nid, term);
        Some(term)
    }

    let mut memo: HashMap<NodeId, Expression> = HashMap::new();
    let nid = extraction.choices[cid].clone();
    build(nid, egraph, extraction, &mut memo)
}

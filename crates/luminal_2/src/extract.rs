use colored::Colorize;
use egraph_serialize::{ClassId, EGraph, NodeId};
use indexmap::IndexMap;
use itertools::Itertools;
use petgraph::{Directed, Direction};
use std::hash::{DefaultHasher, Hash, Hasher};

const WARMUP_TRIALS: usize = 5;
const TRIALS: usize = 5;

/// Enumerate every possible extraction (one node per e‑class reachable
/// from `roots`), call `cost_fn` on each *complete* ExtractionResult,
/// and return all (ExtractionResult, cost) pairs.
///
/// This does a full Cartesian‐product recursion with ZERO partial‐cost pruning.
/// Expect it to blow up exponentially for even modest graphs!
pub fn search(egraph: &EGraph, inputs: &[Vec<f32>]) {
    // 1) Build class → Vec<NodeId> by scanning all nodes in the graph
    let mut class_nodes = IndexMap::<ClassId, Vec<NodeId>>::new();
    for (nid, node) in &egraph.nodes {
        class_nodes
            .entry(node.eclass.clone())
            .or_default()
            .push(nid.clone());
    }

    // 2) Find exactly the set of classes reachable from the roots
    let mut reachable: Vec<ClassId> = Vec::new();
    fn dfs_collect(
        egraph: &EGraph,
        cid: &ClassId,
        seen: &mut Vec<ClassId>,
        class_nodes: &IndexMap<ClassId, Vec<NodeId>>,
    ) {
        if seen.contains(cid) {
            return;
        }
        seen.push(cid.clone());
        // For each node in this class, recurse into its children’s classes
        if let Some(nodes) = class_nodes.get(cid) {
            for nid in nodes {
                for child in &egraph[nid].children {
                    let child_cid = egraph.nid_to_cid(child);
                    dfs_collect(egraph, &child_cid, seen, class_nodes);
                }
            }
        }
    }
    for root in &egraph.root_eclasses {
        dfs_collect(egraph, root, &mut reachable, &class_nodes);
    }

    // 3) Filter class_nodes to only reachable classes, and make it a Vec for ordered recursion
    let class_nodes: Vec<(ClassId, Vec<NodeId>)> = reachable
        .into_iter()
        .filter_map(|cid| {
            class_nodes
                .get(&cid)
                .map(|nodes| (cid.clone(), nodes.clone()))
        })
        .collect();

    // 4) Recursively walk the Cartesian product of choices
    let mut out: Vec<(ExtractionResult, u128)> = Vec::new();
    let mut er = ExtractionResult::default();

    fn recurse(
        idx: usize,
        egraph: &EGraph,
        class_nodes: &[(ClassId, Vec<NodeId>)],
        er: &mut ExtractionResult,
        out: &mut Vec<(ExtractionResult, u128)>,
        inputs: &[Vec<f32>],
        n_accepted: &mut usize,
        memo: &mut HashSet<u64>,
    ) {
        if idx == class_nodes.len() {
            // We have chosen one node for every class → compute cost
            let mut hasher = DefaultHasher::new();
            er.choices.iter().collect_vec().hash(&mut hasher);
            let h = hasher.finish();
            if memo.contains(&h) {
                println!("{}", "Seen".bold());
                return;
            }
            memo.insert(h);
            print!("{}", format!("Graph {} ", n_accepted).bold());
            if let Some(c) = cost(er, egraph, inputs) {
                println!("{}", format!("{c}µs").bright_green().bold());
                out.push((er.clone(), c));
            } else {
                println!("{}", "Invalid".bright_red().bold());
            }
            *n_accepted += 1;
            return;
        }

        let (ref cid, ref opts) = class_nodes[idx];
        for nid in opts {
            er.choices.insert(cid.clone(), nid.clone());
            recurse(
                idx + 1,
                egraph,
                class_nodes,
                er,
                out,
                inputs,
                n_accepted,
                memo,
            );
        }
        er.choices.shift_remove(cid);
    }

    recurse(
        0,
        egraph,
        &class_nodes,
        &mut er,
        &mut out,
        inputs,
        &mut 0,
        &mut HashSet::new(),
    );
    // out
}

#[derive(Default, Clone)]
pub struct ExtractionResult {
    pub choices: IndexMap<ClassId, NodeId>,
}

fn cost(extraction: &ExtractionResult, egraph: &EGraph, inputs: &[Vec<f32>]) -> Option<u128> {
    // Convert to a petgraph
    let graph = extraction_to_graph(egraph, extraction, &egraph.root_eclasses)?;
    let root = graph
        .node_indices()
        .find(|n| {
            graph
                .neighbors_directed(*n, Direction::Outgoing)
                .next()
                .is_none()
        })
        .unwrap();
    let kernels = crate::codegen::codegen(graph, root, GPUArch::Metal(HashMap::new()));
    for _ in 0..WARMUP_TRIALS {
        run_graph(inputs, &kernels);
    }
    let mut micros = vec![];
    for _ in 0..TRIALS {
        let (_outputs, m) = run_graph(inputs, &kernels);
        micros.push(m);
    }
    Some(micros.into_iter().sum::<u128>() / TRIALS as u128)
}

use petgraph::prelude::{NodeIndex, StableGraph};
use std::collections::{HashMap, HashSet};

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
                let range = convert_math(egraph.nid_to_cid(&enode.children[1]), egraph, extraction);
                let stride =
                    convert_math(egraph.nid_to_cid(&enode.children[2]), egraph, extraction);
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
                let range = convert_math(egraph.nid_to_cid(&enode.children[1]), egraph, extraction);
                let stride =
                    convert_math(egraph.nid_to_cid(&enode.children[2]), egraph, extraction);

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
            "Unary" | "Binary" => return None,
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

fn convert_math(cid: &ClassId, egraph: &EGraph, extraction: &ExtractionResult) -> Expression {
    // Memoise by NodeId so we don’t re-convert shared sub-expressions.
    fn build(
        nid: NodeId,
        egraph: &EGraph,
        extraction: &ExtractionResult,
        memo: &mut HashMap<NodeId, Expression>,
    ) -> Expression {
        if let Some(m) = memo.get(&nid) {
            return m.clone();
        }

        let enode = &egraph.nodes[&nid];
        let make_child =
            |child_cid: &ClassId, memo: &mut HashMap<NodeId, Expression>| -> Expression {
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
                if name.len() == 1 {
                    Expression::from(name.chars().next().unwrap())
                } else {
                    panic!("Variable name too long: {name}")
                }
            }

            // ----------- unary ops -----------
            "MNeg" | "MRecip" => {
                let c0 = make_child(&egraph.nid_to_cid(&enode.children[0]), memo);
                match enode.op.as_str() {
                    "MNeg" => c0 * -1,
                    "MRecip" => 1 / c0,
                    _ => unreachable!(),
                }
            }

            // ----------- binary ops -----------
            "MAdd" | "MSub" | "MMul" | "MDiv" | "MMod" | "MMin" | "MMax" | "MAnd" | "MOr"
            | "MGte" | "MLt" | "MFloorTo" => {
                let lhs = make_child(&egraph.nid_to_cid(&enode.children[0]), memo);
                let rhs = make_child(&egraph.nid_to_cid(&enode.children[1]), memo);
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
            "MReplace" => panic!("MReplace not codegennable"),

            // ----------- accumulator marker -----------
            "MAccum" => {
                let name = if enode.children.is_empty() {
                    "<acc>".to_owned()
                } else {
                    "<acc>".to_owned()
                };
                Expression::from(Term::Acc(name.chars().next().unwrap()))
            }
            "Loop" => make_child(&egraph.nid_to_cid(&enode.children[1]), memo),
            "MNum" | "MVar" => make_child(&egraph.nid_to_cid(&enode.children[0]), memo),
            _ => {
                if let Ok(n) = enode.op.parse::<usize>() {
                    Expression::from(n)
                } else {
                    panic!("unsupported Math op '{}'", enode.op)
                }
            }
        };

        memo.insert(nid, term.clone());
        term
    }

    let mut memo: HashMap<NodeId, Expression> = HashMap::new();
    let nid = extraction.choices[cid].clone();
    build(nid, egraph, extraction, &mut memo)
}

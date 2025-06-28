use colored::Colorize;
use egraph_serialize::{ClassId, EGraph, NodeId};
use petgraph::{Directed, Direction};
use rustc_hash::{FxHashMap, FxHashSet};

const WARMUP_TRIALS: usize = 5;
const TRIALS: usize = 5;

type Cost = u128; // Execution time in microseconds

use std::collections::HashMap;

/// All complete extractions whose nodes lie in the sub-DAG reachable from
/// `roots`.  **Exponential** in the number of reachable e-classes.
pub fn list_extractions(egraph: &EGraph, roots: &[ClassId]) -> Vec<ExtractionResult> {
    // 1) Build class → Vec<NodeId>  (one scan of the graph)
    let mut class_nodes: FxHashMap<ClassId, Vec<NodeId>> = FxHashMap::default();
    for (nid, node) in &egraph.nodes {
        class_nodes
            .entry(node.eclass.clone())
            .or_default()
            .push(nid.clone());
    }

    // 2) Collect the *set* of e-classes that are reachable from the roots
    fn collect(
        cid: &ClassId,
        egraph: &EGraph,
        class_nodes: &FxHashMap<ClassId, Vec<NodeId>>,
        seen: &mut FxHashSet<ClassId>,
    ) {
        if !seen.insert(cid.clone()) {
            return; // already visited
        }
        if let Some(nodes) = class_nodes.get(cid) {
            for nid in nodes {
                for child_cid in &egraph.nodes[nid].children {
                    let ccid = egraph.nid_to_cid(child_cid);
                    collect(&ccid, egraph, class_nodes, seen);
                }
            }
        }
    }

    let mut reachable: FxHashSet<ClassId> = FxHashSet::default();
    for root in roots {
        collect(root, egraph, &class_nodes, &mut reachable);
    }

    // 3) Put the reachable classes in a deterministic order
    let mut classes: Vec<ClassId> = reachable.into_iter().collect();
    classes.sort(); // optional but nice for reproducibility

    // 4) Depth-first Cartesian product
    let mut out = Vec::<ExtractionResult>::new();
    let mut choc = FxHashMap::<ClassId, NodeId>::default();

    fn recurse(
        idx: usize,
        classes: &[ClassId],
        class_nodes: &FxHashMap<ClassId, Vec<NodeId>>,
        current: &mut FxHashMap<ClassId, NodeId>,
        out: &mut Vec<ExtractionResult>,
    ) {
        if idx == classes.len() {
            out.push(ExtractionResult {
                choices: current.clone(),
                ..Default::default()
            });
            return;
        }
        let cid = &classes[idx];
        for nid in &class_nodes[cid] {
            current.insert(cid.clone(), nid.clone());
            recurse(idx + 1, classes, class_nodes, current, out);
            current.remove(cid);
        }
    }

    recurse(0, &classes, &class_nodes, &mut choc, &mut out);
    out
}

/// Enumerate every possible extraction and get the runtime cost for each
pub fn search(egraph: &EGraph, inputs: &[Vec<f32>]) {
    let extractions = list_extractions(egraph, &egraph.root_eclasses);
    println!("Extractions: {}", extractions.len());

    // Get runtime cost for each extraction
    for (i, extraction) in extractions.into_iter().enumerate() {
        print!("{}", format!("Graph {i} ").bold());
        if let Some(c) = cost(&extraction, egraph, inputs) {
            println!("{}", format!("{c}µs").bright_green().bold());
        } else {
            println!("{}", "Invalid".bright_red().bold());
        }
    }
}

#[derive(Default, Clone)]
pub struct ExtractionResult {
    pub choices: FxHashMap<ClassId, NodeId>,
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

use petgraph::prelude::{NodeIndex, StableGraph};
use std::collections::HashSet;

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

fn main() {
    match run_egglog_program(include_str!("code.lisp")) {
        Ok((s, _serialized, el_repr)) => {
            if s.is_empty() {
                println!("{}", "Success!".bright_green().bold())
            } else {
                println!("{}", format!("Success: {s:?}").bright_green().bold())
            }
            println!("Kernel: {}", el_repr.bold());
            // println!("{serialized}");
        }
        Err(e) => println!("{e}"),
    }
}

use colored::Colorize;
use egglog::{EGraph, Error, Term, TermDag, ast::Literal, var};
use petgraph::{Directed, graph::NodeIndex, prelude::StableGraph};
use regex::Regex;
use std::collections::HashMap;

/// Runs an Egglog program from a string and returns its output messages.
pub fn run_egglog_program(code: &str) -> Result<(Vec<String>, String, String), Error> {
    // Create a fresh EGraph with all the defaults
    let mut egraph = EGraph::default();
    egraph.enable_messages();
    // The first argument is an optional “filename” used for error messages;
    // here we don’t have one, so pass None.
    let commands = egraph.parser.get_program_from_string(None, code)?;
    let msgs = egraph.run_program(commands)?;
    println!("Run Report:  {}", egraph.get_run_report().as_ref().unwrap());
    let (sort, value) = egraph.eval_expr(&var!("one_output"))?;
    let (termdag, root, _) = egraph.extract_value(&sort, value)?;
    // Remove the underscore prefix and uncomment the display_graph call to display the graph in GraphViz.
    let (_petgraph, _root_idx) = dag_to_petgraph(&termdag, root.clone());
    // display_graph(&petgraph, &[root_idx]);
    let s = egraph.serialize(egglog::SerializeConfig {
        root_eclasses: vec![(sort, value)],
        ..Default::default()
    });
    println!(
        "Nodes: {} Roots: {} Class Data: {}",
        s.nodes.len(),
        s.root_eclasses.len(),
        s.class_data.len()
    );
    let json = serde_json::to_string_pretty(&s).unwrap();
    Ok((msgs, json, termdag.to_string(&root)))
}

fn dag_to_petgraph(dag: &TermDag, root: Term) -> (StableGraph<String, u8, Directed>, NodeIndex) {
    let mut graph: StableGraph<String, u8, Directed> = StableGraph::new();
    let mut map: HashMap<Term, NodeIndex> = HashMap::new();

    // recursive DFS that interns each term exactly once
    fn intern(
        dag: &TermDag,
        g: &mut StableGraph<String, u8, Directed>,
        map: &mut HashMap<Term, NodeIndex>,
        t: Term,
    ) -> NodeIndex {
        if let Some(&idx) = map.get(&t) {
            return idx;
        }
        let string = match &t {
            Term::Var(v) => v.as_str().to_string(),
            Term::Lit(v) => match v {
                Literal::Bool(b) => b.to_string(),
                Literal::Float(f) => f.to_string(),
                Literal::Int(i) => format!("int {i}"),
                Literal::String(s) => format!("\"{s}\""),
                Literal::Unit => "()".to_string(),
            },
            Term::App(v, _) => v.as_str().to_string(),
        };
        let idx = g.add_node(string.clone());
        map.insert(t.clone(), idx);
        if let Term::App(_, children) = &t {
            if string == "LoopIn" || string == "LoopOut" {
                let c_idx = intern(dag, g, map, dag.get(children[0]).clone());
                g.add_edge(c_idx, idx, 0);
            } else {
                for (i, child) in children.iter().enumerate() {
                    let c_idx = intern(dag, g, map, dag.get(*child).clone());
                    g.add_edge(c_idx, idx, i as u8);
                }
            }
        }
        idx
    }

    let root_idx = intern(dag, &mut graph, &mut map, root);
    (graph, root_idx)
}

/// View a debug graph in the browser
pub fn display_graph(
    graph: &petgraph::stable_graph::StableGraph<String, u8, petgraph::Directed, u32>,
    mark_nodes: &[NodeIndex],
) {
    let mut graph_string =
        petgraph::dot::Dot::with_config(&graph, &[petgraph::dot::Config::EdgeIndexLabel])
            .to_string();
    let re = Regex::new(r#"label\s*=\s*"\d+""#).unwrap();
    graph_string = re.replace_all(&graph_string, "").to_string();
    for n in mark_nodes {
        graph_string = graph_string.replace(
            &format!("    {} [ label =", n.index()),
            &format!(
                "    {} [ style=\"filled\" fillcolor=\"yellow\" label =",
                n.index()
            ),
        );
    }

    let url = format!(
        "https://dreampuf.github.io/GraphvizOnline/#{}",
        urlencoding::encode(&graph_string)
    );
    if let Err(e) = webbrowser::open(&url) {
        panic!("Error displaying graph: {:?}", e);
    }
}

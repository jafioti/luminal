use itertools::Itertools;

use super::symbolic::*;

// This is a shape tracker allowing for zero-copy movement ops based off of https://github.com/tinygrad/tinygrad/blob/master/tinygrad/shape/shapetracker.py

fn expr_node(idx: Node, shape: &[usize], strides: &[usize]) -> Node {
    let mut acc = 1;
    let mut ret = vec![];
    for (d, s) in shape.iter().zip(strides.iter()).rev() {
        ret.push(((idx.clone() / (acc as i32)) % (*d as i32)) * (*s as i32));
        acc *= d;
    }

    Node::sum(ret)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct View {
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
}

impl View {
    fn is_contiguous(&self) -> bool {
        self.shape
            .iter()
            .zip(self.strides.iter())
            .zip(default_strides(&self.shape))
            .all(|((&sh, &st), def_st)| st == def_st || sh == 1)
    }
}

fn merge_views(v2: &View, v1: &View) -> Option<View> {
    let idxs = v1
        .shape
        .iter()
        .enumerate()
        .map(|(i, s)| Node::variable(format!("idx{i}"), 0, (s - 1) as i32))
        .collect::<Vec<_>>();
    let idx = Node::sum(
        idxs.clone()
            .into_iter()
            .zip(v1.shape.iter())
            .zip(v1.strides.iter())
            .filter(|((_, sh), st)| **sh != 1 && **st != 0)
            .map(|((i, _), st)| i * *st as i32)
            .collect_vec(),
    );

    let idx = expr_node(idx, &v2.shape, &v2.strides);
    let mut ret = vec![0; idxs.len()];
    for node in if let NodeType::RedNode(RedOp::Sum, n) = idx.node_type {
        n
    } else {
        vec![idx]
    } {
        if let NodeType::OpNode(Op::Mul, a) = &node.node_type {
            if matches!(a.node_type, NodeType::Variable(_)) {
                ret[idxs.iter().position(|i| *i == **a).unwrap()] = node.b as usize;
            } else if matches!(node.node_type, NodeType::Variable(_)) {
                ret[idxs.iter().position(|i| *i == node).unwrap()] = 1;
            }
        } else if matches!(node.node_type, NodeType::Variable(_)) {
            ret[idxs.iter().position(|i| *i == node).unwrap()] = 1;
        }
    }
    if ret.iter().any(|i| *i == 0) {
        None
    } else {
        Some(View {
            shape: v1.shape.clone(),
            strides: ret,
        })
    }
}

pub fn default_strides(shape: &[usize]) -> Vec<usize> {
    let mut acc = 1;
    let mut strides = shape.to_vec();
    for i in strides.iter_mut().rev() {
        let tmp = *i;
        *i = if *i == 0 { 0 } else { acc };
        acc *= tmp;
    }

    strides
}

#[derive(Debug, Clone)]
pub struct ShapeTracker {
    pub views: Vec<View>,
}

impl ShapeTracker {
    pub fn new(shape: Vec<usize>) -> Self {
        Self {
            views: vec![View {
                strides: default_strides(&shape),
                shape,
            }],
        }
    }

    pub fn shape(&self) -> &Vec<usize> {
        &self.views.last().unwrap().shape
    }

    pub fn reshape(&mut self, new_shape: Vec<usize>) {
        let new_view = View {
            strides: default_strides(&new_shape),
            shape: new_shape,
        };
        if self.views.last().unwrap().is_contiguous() {
            *self.views.last_mut().unwrap() = new_view;
        } else {
            self.views.push(new_view);
            self.simplify();
        }
    }

    pub fn expand(&mut self, dimension: usize, new_size: usize) {
        let mut new_shape = self.shape().to_vec();
        new_shape.insert(dimension, new_size);
        self.views.last_mut().unwrap().shape = new_shape;
    }

    fn simplify(&mut self) {
        while self.views.len() > 1 {
            if let Some(merged) = merge_views(
                &self.views[self.views.len() - 2],
                &self.views[self.views.len() - 1],
            ) {
                self.views.pop();
                *self.views.last_mut().unwrap() = merged;
            } else {
                break;
            }
        }
    }

    pub fn permute(&mut self, new_dims: &[usize]) {
        let view = self.views.last_mut().unwrap();
        let (old_shape, old_strides) = (view.shape.clone(), view.strides.clone());
        for (i, j) in new_dims.iter().enumerate() {
            view.shape[i] = old_shape[*j];
            view.strides[i] = old_strides[*j];
        }
    }

    pub fn index_fn(&self) -> impl Fn(usize) -> usize {
        // Get expression
        let mut idx = Node::variable(
            "idx".to_string(),
            0,
            self.shape().iter().product::<usize>() as i32,
        );
        for view in self.views.iter().rev() {
            idx = expr_node(idx, &view.shape, &view.strides);
        }

        // Turn expression into function by unwrapping it into a series of function chains
        let node = &idx;
        let mut all_ops_and_nums = vec![];
        if let NodeType::RedNode(RedOp::Sum, nodes) = &node.node_type {
            for node in nodes {
                all_ops_and_nums.push(get_ops_and_nums(node));
            }
        } else {
            all_ops_and_nums.push(get_ops_and_nums(node));
        }

        move |i| {
            let orig = i as i32;
            let mut result = 0;
            for ops_and_nums in &all_ops_and_nums {
                let mut i = orig;
                for (op, num) in ops_and_nums {
                    i = (op)(i, *num);
                }
                result += i as usize
            }
            result
        }
    }
}

#[allow(clippy::type_complexity)]
fn get_ops_and_nums(mut node: &Node) -> Vec<(fn(i32, i32) -> i32, i32)> {
    let mut ops_and_nums = vec![];
    loop {
        match &node.node_type {
            NodeType::OpNode(op, a) => {
                ops_and_nums.push((
                    match op {
                        Op::Div => std::ops::Div::<i32>::div,
                        Op::Mul => std::ops::Mul::<i32>::mul,
                        Op::Mod => std::ops::Rem::<i32>::rem,
                    },
                    node.b,
                ));
                node = a.as_ref();
            }
            NodeType::Variable(_) => break,
            NodeType::Num => panic!("Num node encountered"),
            NodeType::RedNode(_, _) => panic!("Rednode encountered"),
        }
    }
    ops_and_nums.reverse();
    ops_and_nums
}

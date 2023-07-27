use itertools::Itertools;

use super::{symbolic::*, RealDim};

// This is a shape tracker allowing for zero-copy movement ops based off of https://github.com/tinygrad/tinygrad/blob/master/tinygrad/shape/shapetracker.py

fn expr_node(idx: Node, shape_strides: &[(usize, usize)]) -> Node {
    let mut acc = 1;
    let mut ret = vec![];
    for (d, s) in shape_strides.iter().rev() {
        ret.push(((idx.clone() / (acc as i32)) % (*d as i32)) * (*s as i32));
        acc *= d;
    }

    Node::sum(ret)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct View {
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub shape_strides: Vec<(usize, usize)>,
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

    let idx = expr_node(idx, &v2.shape_strides);
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
            shape_strides: to_shapes_strides(&v1.shape, &ret),
            strides: ret,
        })
    }
}

pub fn default_strides(shape: &[usize]) -> Vec<usize> {
    let mut acc = 1;
    let mut strides = shape.to_vec();
    for i in strides.iter_mut().rev() {
        let tmp = *i;
        *i = if *i == 1 { 0 } else { acc };
        acc *= tmp;
    }

    strides
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShapeTracker {
    pub views: Vec<View>,
}

impl ShapeTracker {
    pub fn new(shape: Vec<usize>) -> Self {
        let strides = default_strides(&shape);
        Self {
            views: vec![View {
                shape_strides: to_shapes_strides(&shape, &strides),
                strides,
                shape,
            }],
        }
    }

    pub fn shape(&self) -> &Vec<usize> {
        &self.views.last().unwrap().shape
    }

    pub fn reshape(&mut self, new_shape: Vec<usize>) {
        let strides = default_strides(&new_shape);
        let new_view = View {
            shape_strides: to_shapes_strides(&new_shape, &strides),
            strides,
            shape: new_shape,
        };
        if self.views.last().unwrap().is_contiguous() {
            *self.views.last_mut().unwrap() = new_view;
        } else {
            self.views.push(new_view);
            self.simplify();
        }
    }

    pub fn expand(&mut self, dimension: usize, new_size: RealDim) {
        self.views.last_mut().unwrap().shape.insert(
            dimension,
            match new_size {
                RealDim::Const(i) => i,
                RealDim::Dyn => 0,
            },
        );
        self.views.last_mut().unwrap().strides.insert(dimension, 0);
        self.views.last_mut().unwrap().shape_strides = to_shapes_strides(
            &self.views.last().unwrap().shape,
            &self.views.last().unwrap().strides,
        );
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
        view.shape_strides = to_shapes_strides(&view.shape, &view.strides);
    }

    pub fn index_fn_node(&self) -> Node {
        // Get expression
        let mut idx = Node::variable(
            "idx".to_string(),
            0,
            self.shape().iter().product::<usize>() as i32,
        );
        for view in self.views.iter().rev() {
            idx = expr_node(idx, &view.shape_strides);
        }
        idx
    }

    pub fn index_fn(&self) -> impl Fn(usize) -> usize {
        let idx = self.index_fn_node();
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
                if ops_and_nums.is_empty() {
                    continue;
                }
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
            NodeType::Num => break,
            NodeType::RedNode(_, _) => panic!("Rednode encountered"),
        }
    }
    ops_and_nums.reverse();
    ops_and_nums
}

fn to_shapes_strides(shape: &[usize], strides: &[usize]) -> Vec<(usize, usize)> {
    let mut ret = if !shape.is_empty() {
        vec![(shape[0], strides[0])]
    } else {
        vec![]
    };

    for i in 1..shape.len() {
        if (strides[i] != 0
            && ret
                .last()
                .map(|(_, x)| *x == shape[i] * strides[i])
                .unwrap_or_default())
            || ret.last().map(|(i, _)| *i == 1).unwrap_or_default()
            || (strides[i] == 0 && ret.last().map(|(_, i)| *i == 0).unwrap_or_default())
        {
            *ret.last_mut().unwrap() = (ret.last().unwrap().0 * shape[i], strides[i]);
        } else {
            ret.push((shape[i], strides[i]));
        }
    }
    ret
}

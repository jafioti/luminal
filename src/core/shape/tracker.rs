use itertools::Itertools;

use super::{symbolic::*, RealDim};

pub use super::symbolic::Node;

// This is a shape tracker allowing for zero-copy movement ops based off of https://github.com/tinygrad/tinygrad/blob/master/tinygrad/shape/shapetracker.py
// This is more or less directly translated from python and so it is very ugly. Needs refactor!
// Uses enums as substitute for python inheritance

fn expr_node(idx: Node, offset: i32, shape_strides: &[(usize, usize)]) -> Node {
    let mut acc = 1;
    let mut ret = if offset != 0 {
        vec![Node::num(offset)]
    } else {
        vec![]
    };
    for (d, s) in shape_strides.iter().rev() {
        ret.push(((idx.clone() / (acc as i32)) % (*d as i32)) * (*s as i32));
        acc *= d;
    }

    Node::sum(ret)
}

fn idxs_to_idx(shape: &[usize], indexes: Vec<Node>) -> Node {
    let mut acc = 1;
    let mut ret = Vec::with_capacity(shape.len());
    for (tidx, d) in indexes.into_iter().zip(shape.iter()).rev() {
        ret.push(tidx * acc);
        acc *= *d as i32;
    }
    Node::sum(ret)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct View {
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub offset: i32,
    pub mask: Option<Vec<(usize, usize)>>,
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

    fn expr_idxs(&self, idxs: Vec<Node>) -> Node {
        let mut nodes = idxs
            .into_iter()
            .zip(self.shape.iter())
            .zip(self.strides.iter())
            .filter(|((_, sh), st)| **sh != 1 && **st != 0)
            .map(|((idx, _), st)| idx * *st as i32)
            .collect_vec();
        nodes.insert(0, Node::num(self.offset));
        Node::sum(nodes)
    }

    fn expr_node(&self, idx: Node) -> Node {
        let mut acc = 1;
        let mut ret = if self.offset != 0 {
            vec![Node::num(self.offset)]
        } else {
            vec![]
        };
        for (d, s) in self.shape_strides.iter().rev() {
            ret.push(((idx.clone() / (acc as i32)) % (*d as i32)) * (*s as i32));
            acc *= d;
        }

        Node::sum(ret)
    }

    fn expr_node_mask(&self, idx: Node, valid: Option<Node>) -> Node {
        let mut expr = if let Some(valid) = valid {
            vec![valid]
        } else {
            vec![]
        };
        if let Some(mask) = &self.mask {
            let mut acc = 1;
            for (ns, (x, y)) in self.shape.iter().zip(mask.iter()).rev() {
                let base = (idx.clone() / acc) % *ns as i32;
                expr.push(base.clone().ge(*x as i32));
                expr.push(base.lt(*y as i32));
                acc *= *ns as i32;
            }
        }
        Node::ands(expr)
    }
}

fn merge_views(v2: &View, v1: &View) -> Option<View> {
    if v2.mask.is_some() {
        return None; // This is in tinygrad
    }
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

    let idx = expr_node(idx, v2.offset, &v2.shape_strides);
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
        let shape_strides = to_shapes_strides(&v1.shape, &ret);
        Some(View {
            shape: v1.shape.clone(),
            strides: ret,
            mask: v1.mask.clone(),
            offset: expr_node(
                Node::variable("idx".to_string(), 0, 0),
                v1.offset,
                &shape_strides,
            )
            .b,
            shape_strides,
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

#[allow(clippy::type_complexity)]
pub fn get_pad_args(shape: &[i32], arg: &[(i32, i32)]) -> (Vec<(i32, i32)>, Vec<(usize, usize)>) {
    (
        shape
            .iter()
            .zip(arg.iter())
            .map(|(s, (b, e))| (-b, s + e))
            .collect(),
        shape
            .iter()
            .zip(arg.iter())
            .map(|(s, (b, _))| (*b as usize, (s + b) as usize))
            .collect(),
    )
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
                mask: None,
                offset: 0,
            }],
        }
    }

    pub fn get_real_shape<const N: usize>(&self, other: [&ShapeTracker; N]) -> Vec<usize> {
        let mut our = self.views.last().unwrap().shape.clone();
        if !our.iter().any(|i| *i == 100) {
            return our;
        }

        // Fill in holes
        for other in other {
            let mut has_zero = false;
            for (i, o) in our.iter_mut().enumerate() {
                if *o == 100 {
                    has_zero = true;
                    *o = other.shape()[i];
                }
            }
            if !has_zero {
                break;
            }
        }

        // Turn remaining 100s into 1s
        for i in &mut our {
            if *i == 100 {
                *i = 1;
            }
        }
        our
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
            mask: None,
            offset: self.views.last().unwrap().offset,
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
                RealDim::Dyn => 100, // Very sloppy, this just needs to be a substantial number that the symbolic library can't get rid of. This needs to change!
            },
        );
        self.views.last_mut().unwrap().strides.insert(dimension, 0);

        self.views.last_mut().unwrap().shape_strides = to_shapes_strides(
            &self.views.last().unwrap().shape,
            &self.views.last().unwrap().strides,
        );
    }

    pub fn reset_shape_strides(&mut self) {
        self.views.last_mut().unwrap().shape_strides = to_shapes_strides(
            &self.views.last().unwrap().shape,
            &self.views.last().unwrap().strides,
        );
    }

    fn unsafe_resize(&mut self, arg: &[(i32, i32)], mut mask: Option<Vec<(usize, usize)>>) {
        let new_offset = self
            .views
            .last()
            .unwrap()
            .strides
            .iter()
            .zip(arg.iter())
            .map(|(a, (b, _))| *a as i32 * b)
            .sum::<i32>();
        if self.views.last().unwrap().mask.is_some() {
            let n_mask: Vec<(usize, usize)> = self
                .views
                .last()
                .unwrap()
                .mask
                .as_ref()
                .unwrap()
                .iter()
                .zip(arg.iter())
                .map(|((mx, my), (ax, ay))| {
                    (
                        0_i32.max(*mx as i32 - *ax) as usize,
                        (*my as i32 - *ax).min(*ay - *ax) as usize,
                    )
                })
                .collect();
            if let Some(m) = mask {
                mask = Some(
                    n_mask
                        .into_iter()
                        .zip(m.into_iter())
                        .map(|((mx1, my1), (mx2, my2))| (mx1.max(mx2), my1.min(my2)))
                        .collect(),
                )
            } else {
                mask = Some(n_mask);
            }
        }
        let shape = arg
            .iter()
            .map(|(a, b)| (b - a) as usize)
            .collect::<Vec<_>>();
        *self.views.last_mut().unwrap() = View {
            strides: self.views.last().unwrap().strides.clone(),
            offset: self.views.last().unwrap().offset + new_offset,
            mask,
            shape_strides: to_shapes_strides(&shape, &self.views.last().unwrap().strides),
            shape,
        };
    }

    pub fn slice(&mut self, ranges: &[(usize, usize)]) {
        self.unsafe_resize(
            &ranges
                .iter()
                .enumerate()
                .map(|(dim, (a, b))| (*a as i32, (*b).min(self.shape()[dim]) as i32))
                .collect_vec(),
            None,
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
            idx = expr_node(idx, view.offset, &view.shape_strides);
        }
        idx
    }

    pub fn pad(&mut self, arg: &[(i32, i32)]) {
        if arg.iter().any(|(a, b)| *a != 0 || *b != 0) {
            let (zvarg, mask) =
                get_pad_args(&self.shape().iter().map(|i| *i as i32).collect_vec(), arg);
            self.unsafe_resize(&zvarg, Some(mask))
        }
    }

    fn _expr_idxs(&self, mut idx: Node, mut valid: Node) -> (Node, Node) {
        for v in self.views.iter().rev().skip(1) {
            valid = v.expr_node_mask(idx.clone(), Some(valid));
            idx = v.expr_node(idx);
        }
        (idx, valid)
    }

    pub fn expr_idxs(&self, indexes: Vec<Node>) -> (Node, Node) {
        let idx = self.views.last().unwrap().expr_idxs(indexes.clone());
        let valid = self.views.last().unwrap().expr_node_mask(
            idxs_to_idx(&self.views.last().unwrap().shape, indexes),
            None,
        );
        self._expr_idxs(idx, valid)
    }

    pub fn index_node(&self) -> (Node, Node) {
        let idx = Node::variable(
            "idx".to_string(),
            0,
            self.shape().iter().product::<usize>() as i32,
        );
        let (idx, valid) = self._expr_idxs(
            self.views.last().unwrap().expr_node(idx.clone()),
            self.views.last().unwrap().expr_node_mask(idx, None),
        );

        (idx, valid)
    }
}

pub fn to_shapes_strides(shape: &[usize], strides: &[usize]) -> Vec<(usize, usize)> {
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

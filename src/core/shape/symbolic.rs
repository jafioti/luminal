use std::{
    collections::{HashMap, HashSet},
    ops::*,
};

use itertools::Itertools;

// This is a tiny symbolic algebra library based off of https://github.com/tinygrad/tinygrad/blob/master/tinygrad/shape/symbolic.py

#[derive(Debug, Clone, Copy, PartialEq, Hash, Eq)]
pub enum Op {
    Mul,
    Div,
    Mod,
}

#[derive(Debug, Clone, Copy, PartialEq, Hash, Eq)]
pub enum RedOp {
    Sum,
}

#[derive(Debug, Clone, PartialEq, Hash, Eq)]
pub enum NodeType {
    Variable(String),
    Num,
    OpNode(Op, Box<Node>),
    RedNode(RedOp, Vec<Node>),
}

impl ToString for Node {
    fn to_string(&self) -> String {
        match &self.node_type {
            NodeType::Num => self.b.to_string(),
            NodeType::Variable(n) => format!("{n}[{}-{}]", self.min, self.max),
            NodeType::OpNode(o, a) => format!(
                "({} {} {})",
                a.to_string(),
                match o {
                    Op::Div => "/",
                    Op::Mod => "%",
                    Op::Mul => "*",
                },
                self.b
            ),
            NodeType::RedNode(o, n) => format!(
                "({})",
                n.iter()
                    .map(|i| i.to_string())
                    .sorted()
                    .collect::<Vec<_>>()
                    .join(match o {
                        RedOp::Sum => " + ",
                    })
            ),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Hash, Eq)]
pub struct Node {
    pub b: i32,
    pub min: i32,
    pub max: i32,
    pub node_type: NodeType,
}

impl Node {
    pub fn to_string_no_range(&self) -> String {
        match &self.node_type {
            NodeType::Num => self.b.to_string(),
            NodeType::Variable(n) => n.to_string(),
            NodeType::OpNode(o, a) => format!(
                "({} {} {})",
                a.to_string_no_range(),
                match o {
                    Op::Div => "/",
                    Op::Mod => "%",
                    Op::Mul => "*",
                },
                self.b
            ),
            NodeType::RedNode(o, n) => format!(
                "({})",
                n.iter()
                    .map(|i| i.to_string_no_range())
                    .sorted()
                    .collect::<Vec<_>>()
                    .join(match o {
                        RedOp::Sum => " + ",
                    })
            ),
        }
    }

    pub fn variable(name: String, min: i32, max: i32) -> Self {
        Node {
            b: min,
            min,
            max,
            node_type: if min == max {
                NodeType::Num
            } else {
                NodeType::Variable(name)
            },
        }
    }

    pub fn num(num: i32) -> Self {
        Node {
            b: num,
            min: num,
            max: num,
            node_type: NodeType::Num,
        }
    }

    pub fn sum(mut nodes: Vec<Node>) -> Node {
        if nodes.is_empty() {
            return Node {
                b: 0,
                min: 0,
                max: 0,
                node_type: NodeType::Num,
            };
        }
        if nodes.len() == 1 {
            return nodes.pop().unwrap();
        }

        let mut new_nodes = vec![];
        let mut num_node_sum = 0;
        for node in nodes {
            if !matches!(
                node.node_type,
                NodeType::Num | NodeType::RedNode(RedOp::Sum, _)
            ) {
                new_nodes.push(node);
            } else if matches!(node.node_type, NodeType::Num) {
                num_node_sum += node.b;
            } else if matches!(node.node_type, NodeType::RedNode(RedOp::Sum, _)) {
                for sub_node in flat_rednode_components(&node) {
                    if matches!(sub_node.node_type, NodeType::Num) {
                        num_node_sum += sub_node.b;
                    } else {
                        new_nodes.push(sub_node.clone());
                    }
                }
            }
        }

        let num_new_nodes = new_nodes.len();
        if num_new_nodes > 1
            && new_nodes
                .iter()
                .map(|i| {
                    if let NodeType::OpNode(Op::Mul, a) = &i.node_type {
                        a.as_ref()
                    } else {
                        i
                    }
                })
                .collect::<HashSet<&Node>>()
                .len()
                < num_new_nodes
        {
            new_nodes = Self::factorize(new_nodes);
        }

        if num_node_sum > 0 {
            new_nodes.push(Node::num(num_node_sum));
        }

        if new_nodes.is_empty() {
            Node::num(0)
        } else if new_nodes.len() == 1 {
            new_nodes.pop().unwrap()
        } else {
            Node {
                b: 0,
                min: new_nodes.iter().map(|n| n.min).sum(),
                max: new_nodes.iter().map(|n| n.max).sum(),
                node_type: NodeType::RedNode(RedOp::Sum, new_nodes),
            }
        }
    }

    pub fn factorize(nodes: Vec<Node>) -> Vec<Node> {
        let mut mul_groups = HashMap::new();
        for x in nodes {
            let (a, b) = if let NodeType::OpNode(Op::Mul, a) = x.node_type {
                ((*a).clone(), x.b)
            } else {
                let b = x.b;
                (x, b)
            };

            let val = mul_groups.get(&a).copied().unwrap_or_default() + b;
            mul_groups.insert(a, val);
        }

        mul_groups
            .into_iter()
            .filter(|(_, i)| *i != 0)
            .map(|(a, b_sum)| {
                if b_sum != 1 {
                    let (min, max) = if b_sum >= 0 {
                        (a.min * b_sum, a.max * b_sum)
                    } else {
                        (a.max * b_sum, a.min * b_sum)
                    };
                    Node {
                        b: b_sum,
                        min,
                        max,
                        node_type: NodeType::OpNode(Op::Mul, Box::new(a)),
                    }
                } else {
                    a
                }
            })
            .collect()
    }
}

fn flat_rednode_components(node: &Node) -> Vec<&Node> {
    if let NodeType::RedNode(_, nodes) = &node.node_type {
        nodes
            .iter()
            .flat_map(|n| {
                let mut nodes = flat_rednode_components(n);
                nodes.push(n);
                nodes
            })
            .collect()
    } else {
        vec![]
    }
}

impl Add<i32> for Node {
    type Output = Node;

    fn add(self, rhs: i32) -> Self::Output {
        Node::sum(vec![self, Node::num(rhs)])
    }
}

impl Add<Node> for Node {
    type Output = Node;

    fn add(self, rhs: Node) -> Self::Output {
        Node::sum(vec![self, rhs])
    }
}

impl std::iter::Sum for Node {
    fn sum<I: Iterator<Item = Self>>(mut iter: I) -> Self {
        let mut acc = iter.next().unwrap_or(Node::num(0));
        for i in iter {
            acc = acc + i;
        }
        acc
    }
}

impl Sub<i32> for Node {
    type Output = Node;

    fn sub(self, rhs: i32) -> Self::Output {
        self + -rhs
    }
}

impl Sub<Node> for Node {
    type Output = Node;

    fn sub(self, rhs: Node) -> Self::Output {
        self + -rhs
    }
}

impl Div<i32> for Node {
    type Output = Node;
    fn div(mut self, rhs: i32) -> Self::Output {
        if rhs < 0 {
            return (self / -rhs) * -1;
        }
        if rhs == 1 {
            return self;
        }
        if let NodeType::OpNode(Op::Div, a) = &self.node_type {
            return *a.clone() / (self.b * rhs); // Two divs is one div
        }
        if let NodeType::OpNode(Op::Mul, a) = &self.node_type {
            if self.b % rhs == 0 {
                return *a.clone() * (self.b / rhs);
            }
            if rhs % self.b == 0 && self.b > 0 {
                return *a.clone() / (rhs / self.b);
            }
        }
        if let NodeType::OpNode(Op::Mod, a) = &self.node_type {
            if self.b % rhs == 0 {
                return (*a.clone() / self.b) % (self.b / rhs); // Put the div inside mod
            }
        }
        if matches!(self.node_type, NodeType::RedNode(RedOp::Sum, _)) {
            let mut fully_divided = vec![];
            let mut rest = vec![];
            let mut _gcd = rhs;
            let mut divisor = 1;
            for node in flat_rednode_components(&self) {
                // Distribute the divide
                if matches!(node.node_type, NodeType::OpNode(Op::Mul, _) | NodeType::Num) {
                    if node.b % rhs == 0 {
                        fully_divided.push(node.clone() / rhs);
                    } else {
                        _gcd = gcd(_gcd, node.b);
                        rest.push(node.clone());
                        if matches!(node.node_type, NodeType::OpNode(Op::Mul, _))
                            && divisor == 1
                            && rhs % node.b == 0
                        {
                            divisor = node.b;
                        }
                    }
                } else {
                    rest.push(node.clone());
                    _gcd = 1;
                }
            }

            if _gcd > 1 {
                return Node::sum(fully_divided) + Node::sum(rest) / _gcd / (rhs / _gcd);
            } else if divisor > 1 {
                return Node::sum(fully_divided) + Node::sum(rest) / divisor / (rhs / divisor);
            } else {
                self = Node::sum(rest);
                self = if self.min < 0 {
                    let offset = self.min / rhs;
                    (self + -offset * rhs) / rhs + offset
                } else if self.min / rhs == self.max / rhs {
                    Node::num(self.min / rhs)
                } else {
                    Node {
                        b: rhs,
                        min: self.min / rhs,
                        max: self.max / rhs,
                        node_type: NodeType::OpNode(Op::Div, Box::new(self)),
                    }
                };
                return Node::sum(fully_divided) + self;
            }
        }

        if self.min < 0 {
            let offset = self.min / rhs;
            (self + -offset * rhs) / rhs + offset
        } else if self.min / rhs == self.max / rhs {
            Node::num(self.min / rhs)
        } else {
            Node {
                b: rhs,
                min: self.min / rhs,
                max: self.max / rhs,
                node_type: NodeType::OpNode(Op::Div, Box::new(self)),
            }
        }
    }
}

impl Rem<i32> for Node {
    type Output = Node;
    fn rem(mut self, rhs: i32) -> Self::Output {
        if let NodeType::OpNode(Op::Mul, a) = &self.node_type {
            self = *a.clone() * (self.b % rhs);
        }

        if let NodeType::RedNode(RedOp::Sum, n) = &self.node_type {
            let mut new_nodes = vec![];
            for node in n {
                if let NodeType::Num = node.node_type {
                    new_nodes.push(Node::num(node.b % rhs));
                } else if let NodeType::OpNode(Op::Mul, a) = node.node_type.clone() {
                    new_nodes.push(*a * (node.b % rhs));
                } else {
                    new_nodes.push(node.clone());
                }
            }

            self = Node::sum(new_nodes);
        }

        if self.min < 0 {
            let min = self.min;
            (self - ((min / rhs) * rhs)) % rhs
        } else if self.max < rhs {
            return self;
        } else {
            let (min, max) = if self.max - self.min >= rhs
                || (self.min != self.max && self.min % rhs >= self.max % rhs)
            {
                (0, rhs - 1)
            } else {
                (self.min % rhs, self.max % rhs)
            };
            Node {
                b: rhs,
                max,
                min,
                node_type: NodeType::OpNode(Op::Mod, Box::new(self)),
            }
        }
    }
}

impl Mul<i32> for Node {
    type Output = Node;

    fn mul(self, rhs: i32) -> Self::Output {
        if rhs == 0 || (matches!(self.node_type, NodeType::Num) && self.b == 0) {
            Node::num(0)
        } else if rhs == 1 {
            self
        } else if let NodeType::OpNode(Op::Mul, a) = self.node_type {
            // Two muls in one mul
            *a * (self.b * rhs)
        } else if let NodeType::RedNode(RedOp::Sum, n) = self.node_type {
            Node::sum(n.into_iter().map(|n| n * rhs).collect()) // Distribute mul into sum
        } else {
            let (min, max) = if rhs >= 0 {
                (self.min * rhs, self.max * rhs)
            } else {
                (self.max * rhs, self.min * rhs)
            };
            Node {
                b: rhs,
                min,
                max,
                node_type: NodeType::OpNode(Op::Mul, Box::new(self)),
            }
        }
    }
}

impl Neg for Node {
    type Output = Self;
    fn neg(self) -> Self::Output {
        self * -1
    }
}

pub fn gcd(mut n: i32, mut m: i32) -> i32 {
    assert!(n != 0 && m != 0);
    while m != 0 {
        if m < n {
            std::mem::swap(&mut m, &mut n);
        }
        m %= n;
    }
    n
}

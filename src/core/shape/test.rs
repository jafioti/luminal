use super::ShapeTracker;
use crate::{prelude::View, shape::symbolic::Node};

#[test]
fn test_create() {
    let s = ShapeTracker::new(vec![2, 2, 3]);
    assert_eq!(
        s.index_fn_node(),
        Node {
            b: 12,
            min: 0,
            max: 11,
            node_type: crate::shape::symbolic::NodeType::OpNode(
                crate::shape::symbolic::Op::Mod,
                Box::new(Node {
                    b: 0,
                    min: 0,
                    max: 12,
                    node_type: crate::shape::symbolic::NodeType::Variable("idx".to_string())
                })
            )
        }
    );
}

#[test]
fn test_reshape() {
    let mut s = ShapeTracker::new(vec![2, 5, 3]);

    s.reshape(vec![10, 3]);

    assert_eq!(
        s.views,
        vec![View {
            shape: vec![10, 3],
            strides: vec![3, 1],
            shape_strides: vec![(30, 1)],
            mask: None,
            offset: 0,
        }]
    );

    s.reshape(vec![30]);

    assert_eq!(
        s.views,
        vec![View {
            shape: vec![30],
            strides: vec![1],
            shape_strides: vec![(30, 1)],
            mask: None,
            offset: 0,
        }]
    );
}

#[test]
fn test_permute() {
    let mut s = ShapeTracker::new(vec![2, 5, 3]);

    s.permute(&[1, 0, 2]);

    assert_eq!(
        s.views,
        vec![View {
            shape: vec![5, 2, 3],
            strides: vec![3, 15, 1],
            shape_strides: vec![(5, 3), (2, 15), (3, 1)],
            mask: None,
            offset: 0,
        }]
    );

    s.permute(&[0, 2, 1]);

    assert_eq!(
        s.views,
        vec![View {
            shape: vec![5, 3, 2],
            strides: vec![3, 1, 15],
            shape_strides: vec![(15, 1), (2, 15)],
            mask: None,
            offset: 0,
        }]
    );
}

#[test]
fn test_expand() {
    let mut s = ShapeTracker::new(vec![2, 5]);

    s.expand(2, crate::prelude::RealDim::Const(3));

    assert_eq!(
        s.views,
        vec![View {
            shape: vec![2, 5, 3],
            strides: vec![5, 1, 0],
            shape_strides: vec![(10, 1), (3, 0)],
            mask: None,
            offset: 0,
        }]
    );
}

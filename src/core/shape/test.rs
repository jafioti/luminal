use super::ShapeTracker;
use crate::prelude::View;

#[test]
fn test_reshape() {
    let mut s = ShapeTracker::new(vec![2, 5, 3]);

    s.reshape(vec![10, 3]);

    assert_eq!(
        s.views,
        vec![View {
            shape: vec![10, 3],
            strides: vec![3, 1],
        }]
    );

    s.reshape(vec![30]);

    assert_eq!(
        s.views,
        vec![View {
            shape: vec![30],
            strides: vec![1],
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
        }]
    );

    s.permute(&[0, 2, 1]);

    assert_eq!(
        s.views,
        vec![View {
            shape: vec![5, 3, 2],
            strides: vec![3, 1, 15],
        }]
    );
}

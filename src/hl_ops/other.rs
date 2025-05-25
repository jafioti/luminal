use std::{f32, path::PathBuf};

use colored::Colorize;
use itertools::Itertools;

use crate::{
    op::{self, Constant, ConstantValue},
    prelude::*,
};

impl GraphTensor {
    /// Cumulative sum last dimension
    pub fn cumsum_last_dim(mut self) -> Self {
        let axis = self.shape.len() - 1;
        if !self.shape.is_contiguous() {
            self = self.contiguous();
        }
        // Pad out length
        let orig_length = self.shape.dims[self.shape.indexes[axis]];
        self.shape.padding[self.shape.indexes[axis]].0 = orig_length - 1;
        self = self.contiguous();

        // Pool
        let mut pooled = self.pool_last_dim(orig_length, 1, 1);
        // Sum Reduce along new dimension
        let final_id = self
            .graph()
            .add_op(op::SumReduce(axis))
            .input(pooled.id, 0, pooled.shape)
            .finish();
        pooled.shape.remove_dim(axis + 1);
        GraphTensor::from_id(final_id, pooled.shape, self.graph_ref)
    }

    /// Cumulative max last dimension
    pub fn cummax_last_dim(mut self) -> Self {
        let axis = self.shape.len() - 1;
        if !self.shape.is_contiguous() {
            self = self.contiguous();
        }
        // Pad out length
        let orig_length = self.shape.dims[self.shape.indexes[axis]];
        self.shape.padding[self.shape.indexes[axis]].0 = orig_length - 1;
        self = self.contiguous();

        // Pool
        let mut pooled = self.pool_last_dim(orig_length, 1, 1);
        // Max Reduce along new dimension
        let final_id = self
            .graph()
            .add_op(op::MaxReduce(axis))
            .input(pooled.id, 0, pooled.shape)
            .finish();
        pooled.shape.remove_dim(axis + 1);
        GraphTensor::from_id(final_id, pooled.shape, self.graph_ref)
    }

    /// Cumulative product last dimension
    pub fn cumprod_last_dim(self) -> Self {
        self.ln().cumsum_last_dim().exp()
    }
}

impl Graph {
    /// A scalar constant
    pub fn constant(&mut self, i: impl Into<ConstantValue>) -> GraphTensor {
        GraphTensor::from_id(
            self.add_op(Constant(i.into(), &self.dyn_map)).finish(),
            ShapeTracker::new(()),
            self,
        )
    }

    /// ARange from 0 to N
    pub fn arange(&mut self, to: impl Into<Expression>) -> GraphTensor {
        let to = to.into();
        if to.to_usize().map(|i| i == 1).unwrap_or_default() {
            // Single number ARange is just 0
            self.constant(0.).expand(0, to)
        } else {
            self.constant(1.).expand(0, to).cumsum_last_dim() - 1.
        }
    }

    /// Lower left-hand triangle of 1s. Currently required to be square
    ///
    /// Same API as https://pytorch.org/docs/stable/generated/torch.tril
    pub fn tril(&mut self, size: impl Into<Expression>, diagonal: i32) -> GraphTensor {
        let size = size.into();
        let horizontal = self.arange(size).expand(0, size);
        let vertical = self.arange(size).expand(1, size);

        (horizontal - (diagonal as f32 + 1.)).less_than(vertical)
    }

    /// Upper right-hand triangle of 1s
    ///
    /// Same API as https://pytorch.org/docs/stable/generated/torch.triu
    pub fn triu(&mut self, size: impl Into<Expression>, diagonal: i32) -> GraphTensor {
        let size = size.into();
        let horizontal = self.arange(size).expand(0, size);
        let vertical = self.arange(size).expand(1, size);

        (horizontal - (diagonal as f32 - 1.)).greater_than(vertical)
    }
}

impl GraphTensor {
    /// Gather a batch of vectors from a matrix
    pub fn gather(self, indexes: GraphTensor) -> GraphTensor {
        let (vocab, dim) = self.dims2();
        let batch = indexes.dims1();
        let one_hot = indexes
            .graph()
            .arange(vocab)
            .expand(0, batch)
            .equals(indexes.expand(1, vocab));
        (one_hot.expand(2, dim) * self.expand(0, batch)).sum_reduce(1)
    }

    /// Print the value of this tensor when the graph is ran
    pub fn print<T: ToString>(&self, message: T) -> Self {
        let message = message.to_string();
        let id = self
            .graph()
            .add_op(op::Function(
                "Print".to_string(),
                Box::new(move |inp| {
                    for (i, (tensor, tracker)) in inp.iter().enumerate() {
                        println!("{message} ({})", i + 1);
                        let d = tensor.borrowed().downcast_ref::<Vec<f32>>().unwrap();
                        println!(
                            "Elements: {} Start: {:?} Mid: {:?} End: {:?}",
                            d.len(),
                            &d[..d.len().min(5)],
                            &d[d.len().saturating_div(2)
                                ..(d.len().saturating_div(2) + 5).min(d.len())],
                            &d[d.len().saturating_sub(5)..]
                        );
                        println!("Shape: {:?}", tracker);
                    }
                    vec![]
                }),
            ))
            .input(self.id, 0, self.shape)
            .finish();
        self.graph().no_delete.insert(id);
        *self
    }

    /// Check the tensor value against a binary file
    pub fn diff(&self, file: impl Into<PathBuf>, atol: f32, rtol: f32) -> Self {
        let path = file.into();
        let id = self
            .graph()
            .add_op(op::Function(
                format!("Diff {path:?}"),
                Box::new(move |mut inp| {
                    // Get tensor data and file data
                    let (tensor, shape) = inp.pop().unwrap();
                    let d = tensor.borrowed().downcast_ref::<Vec<f32>>().unwrap();
                    let mut data = vec![0.; d.len()];
                    let (ind, val) = (shape.index_expression(), shape.valid_expression());
                    let mut stack = vec![];
                    #[allow(unused_mut)]
                    for (i, mut r) in data.iter_mut().enumerate() {
                        if val.exec_single_var_stack(i, &mut stack) != 0 {
                            *r = d[ind.exec_single_var_stack(i, &mut stack)];
                        }
                    }
                    let bin_data = std::fs::read(&path)
                        .unwrap()
                        .chunks(4)
                        .map(|i| {
                            f32::from_ne_bytes([i[0], i[1], i[2], i[3]]).clamp(f32::MIN, f32::MAX)
                        })
                        .collect::<Vec<_>>();
                    if data.len() != bin_data.len() {
                        println!(
                            "{}",
                            format!(
                                "{} | Length mismatch! Data: {}, File: {}",
                                path.as_os_str().to_str().unwrap(),
                                data.len(),
                                bin_data.len()
                            )
                            .bold()
                            .red()
                        );
                        println!("Data Shape: {shape:?}");
                        return vec![];
                    }
                    let data_nan = data.iter().any(|i| i.is_nan());
                    let file_nan = bin_data.iter().any(|i| i.is_nan());
                    if data_nan {
                        println!(
                            "{}",
                            format!("{} | Data contains nan!", path.to_str().unwrap())
                                .bold()
                                .red()
                        );
                    }
                    if file_nan {
                        println!(
                            "{}",
                            format!("{} | File contains nan!", path.to_str().unwrap())
                                .bold()
                                .red()
                        );
                    }
                    if data_nan || file_nan {
                        return vec![];
                    }
                    let mut matched = true;
                    for (i, (a, b)) in data.iter().zip(bin_data.iter()).enumerate() {
                        let tolerance = atol + rtol * a.abs().max(b.abs());
                        if (a - b).abs() > tolerance {
                            println!(
                                "{}",
                                format!("{} | Value Mismatch!", path.to_str().unwrap())
                                    .bold()
                                    .red()
                            );
                            if let Some((i, _)) = data.iter().enumerate().find(|(_, i)| i.is_nan())
                            {
                                println!("Index {} is nan!", i.to_string().bold());
                            }
                            println!("{a} is not equal to {b}, index {i}");

                            let mut diffs: Vec<f32> = data
                                .iter()
                                .zip(bin_data.iter())
                                .map(|(a, b)| (a - b).abs())
                                .collect();
                            diffs.sort_by(|x, y| {
                                x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal)
                            });
                            let len = diffs.len();
                            // percentile indices (clamp to len-1)
                            let p50_idx = ((len as f32) * 0.50).round() as usize;
                            let p95_idx = ((len as f32) * 0.95).round() as usize;
                            let p99_idx = ((len as f32) * 0.99).round() as usize;
                            let p50 = diffs[p50_idx.min(len - 1)];
                            let p95 = diffs[p95_idx.min(len - 1)];
                            let p99 = diffs[p99_idx.min(len - 1)];

                            // summary stats
                            let avg_dist = diffs.iter().sum::<f32>() / len as f32;
                            let max_dist = *diffs.last().unwrap();
                            let sum_dist = data
                                .iter()
                                .zip(bin_data.iter())
                                .map(|(a, b)| (a - b).powi(2))
                                .sum::<f32>();

                            println!(
                                "Avg dist: {}, Max dist: {} Sum dist: {}",
                                avg_dist.to_string().bold().red(),
                                max_dist.to_string().bold().red(),
                                sum_dist.to_string().bold().red(),
                            );
                            println!(
                                "p50: {}  p95: {}  p99: {}",
                                p50.to_string().bold().red(),
                                p95.to_string().bold().red(),
                                p99.to_string().bold().red(),
                            );

                            println!("Data Shape: {shape:?}");
                            println!("{}: {:?}", "This".bold(), &data[..10]);
                            println!("{}: {:?}", "File".bold(), &bin_data[..10]);
                            println!(
                                "Largest Mismatches: {:?}",
                                data.iter()
                                    .zip(bin_data.iter())
                                    .filter(|(a, b)| (**a - **b).abs() > 0.01)
                                    .sorted_by(|(a, b), (c, d)| (**c - **d)
                                        .abs()
                                        .partial_cmp(&(**a - **b).abs())
                                        .unwrap_or(std::cmp::Ordering::Equal))
                                    .take(10)
                                    .collect::<Vec<_>>()
                            );
                            println!(
                                "A avg: {} B avg: {}",
                                data.iter().sum::<f32>() / data.len() as f32,
                                bin_data.iter().sum::<f32>() / bin_data.len() as f32
                            );
                            println!(
                                "A max: {} B max: {}",
                                data.iter()
                                    .max_by(|a, b| a
                                        .partial_cmp(b)
                                        .unwrap_or(std::cmp::Ordering::Equal))
                                    .unwrap(),
                                bin_data
                                    .iter()
                                    .max_by(|a, b| a
                                        .partial_cmp(b)
                                        .unwrap_or(std::cmp::Ordering::Equal))
                                    .unwrap()
                            );
                            println!(
                                "A min: {} B min: {}",
                                data.iter()
                                    .min_by(|a, b| a
                                        .partial_cmp(b)
                                        .unwrap_or(std::cmp::Ordering::Equal))
                                    .unwrap(),
                                bin_data
                                    .iter()
                                    .min_by(|a, b| a
                                        .partial_cmp(b)
                                        .unwrap_or(std::cmp::Ordering::Equal))
                                    .unwrap()
                            );
                            matched = false;
                            break;
                        }
                    }
                    if matched {
                        println!(
                            "{}",
                            format!("{} matched", path.to_str().unwrap())
                                .bold()
                                .bright_green()
                        );
                    }
                    vec![]
                }),
            ))
            .input(self.id, 0, self.shape)
            .finish();
        self.graph().no_delete.insert(id);
        *self
    }
}

#[cfg(test)]
mod tests {
    crate::test_imports!();
    #[test]
    fn test_arange() {
        let mut cx = Graph::new();

        let arange = cx.arange(10).retrieve();
        cx.execute();

        assert_exact(&arange.data(), &[0., 1., 2., 3., 4., 5., 6., 7., 8., 9.]);
    }

    #[test]
    fn test_cumprod() {
        let mut cx = Graph::new();

        let a = cx.tensor(3).set(vec![3., 2., 5.]);
        let b = a.cumprod_last_dim().retrieve();
        cx.execute();

        assert_close(&b.data(), &[3., 6., 30.]);
    }

    #[test]
    fn test_dyn_arange() {
        let mut cx = Graph::new();

        let arange = cx.arange('a').retrieve();
        cx.set_dyn_dim('a', 6);

        cx.execute();

        assert_exact(&arange.data(), &[0., 1., 2., 3., 4., 5.]);
    }

    #[test]
    fn test_tril() {
        let mut cx = Graph::new();

        let triangle = cx.tril(5, 1).retrieve();

        cx.execute();

        assert_exact(
            &triangle.data(),
            &[
                [1.00, 1.00, 0.00, 0.00, 0.00],
                [1.00, 1.00, 1.00, 0.00, 0.00],
                [1.00, 1.00, 1.00, 1.00, 0.00],
                [1.00, 1.00, 1.00, 1.00, 1.00],
                [1.00, 1.00, 1.00, 1.00, 1.00],
            ]
            .into_iter()
            .flatten()
            .collect::<Vec<_>>(),
        );
    }

    #[test]
    fn test_triu() {
        let mut cx = Graph::new();

        let a = cx.triu(3, -1).retrieve();
        let b = cx.triu(3, 0).retrieve();
        let c = cx.triu(3, 1).retrieve();

        cx.execute();

        assert_exact(
            &a.data(),
            &[[1.00, 1.00, 1.00], [1.00, 1.00, 1.00], [0.00, 1.00, 1.00]]
                .into_iter()
                .flatten()
                .collect::<Vec<_>>(),
        );
        assert_exact(
            &b.data(),
            &[[1.00, 1.00, 1.00], [0.00, 1.00, 1.00], [0.00, 0.00, 1.00]]
                .into_iter()
                .flatten()
                .collect::<Vec<_>>(),
        );
        assert_exact(
            &c.data(),
            &[[0.00, 1.00, 1.00], [0.00, 0.00, 1.00], [0.00, 0.00, 0.00]]
                .into_iter()
                .flatten()
                .collect::<Vec<_>>(),
        );
    }
}

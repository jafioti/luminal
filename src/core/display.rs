use std::fmt::Debug;

use itertools::Itertools;

use crate::{
    graph_tensor::GraphTensor,
    shape::{symbolic::Expression, Shape},
};

fn pretty_print_tensor_recursive(
    f: &mut std::fmt::Formatter<'_>,
    data: &[f32],
    shape: &[usize],
    level: usize,
) -> std::fmt::Result {
    if shape.is_empty() {
        // Base case: no dimensions left
        return Ok(());
    }

    let indent = "  ".repeat(level);

    if shape.len() == 1 {
        // If this is the innermost dimension, print the raw data in a single line
        write!(f, "{}[", indent)?;
        for (i, value) in data.iter().enumerate() {
            write!(f, "{:.2}", value)?;
            if i < data.len() - 1 {
                write!(f, ", ")?;
            }
        }
        write!(f, "]")?; // No newline after the innermost array
    } else {
        // For higher dimensions, handle the nesting
        writeln!(f, "{}[", indent)?;
        let stride = shape[1..].iter().product();
        for (i, chunk) in data.chunks(stride).enumerate() {
            pretty_print_tensor_recursive(f, chunk, &shape[1..], level + 1)?;
            if i < shape[0] - 1 {
                write!(f, ",\n")?; // Place the comma right after the bracket and then a newline
            }
        }
        writeln!(f)?; // Add a newline before closing the current dimension bracket
        write!(f, "{}]", indent)?; // Close the current dimension bracket
    }

    // Only add a newline after the top-level closing bracket
    if level == 0 {
        writeln!(f)?;
    }

    Ok(())
}

impl<S: Shape> Debug for GraphTensor<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Get the data
        let data = self.data();

        // Get the shape
        let shape: Vec<usize> = self
            .shape
            .shape()
            .iter()
            .map(|expr: &Expression| expr.to_usize().unwrap())
            .collect_vec();

        // Now we just print the shape
        writeln!(f, "Tensor with Shape: {:?}", shape)?;

        // Now we try to print it by going dimension by dimension, recursively
        pretty_print_tensor_recursive(f, &data, &shape, 0)
    }
}

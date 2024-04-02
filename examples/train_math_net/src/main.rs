use luminal::prelude::*;
use luminal_nn::{Linear, Swish};
use luminal_training::{mse_loss, sgd, Autograd};
use rand::{rngs::ThreadRng, thread_rng, Rng};

// This is a simple example of using luminal to train.
// Here we are training an MLP to add 4 bit numbers together into a resultant 5 bit number.

fn main() {
    // Setup gradient graph
    let mut grad_cx = Graph::new();
    let model =
        <(Linear<8, 16>, Swish, Linear<16, 16>, Swish, Linear<16, 5>)>::initialize(&mut grad_cx);
    let mut input = grad_cx.tensor::<R1<8>>();
    let mut target = grad_cx.tensor::<R1<5>>();
    let mut output = model.forward(input).retrieve();
    let mut loss = mse_loss(output, target).retrieve();

    let weights = params(&model);
    grad_cx.keep_tensors(&weights);
    let grads = grad_cx.compile(
        Autograd::new(&weights, loss),
        (&mut input, &mut target, &mut loss, &mut output),
    );
    grad_cx.keep_tensors(&grads);

    // Setup opt graph
    let (old_weights, opt_grads, new_weights, mut opt_cx, lr) = sgd(&grads);
    lr.set(1e-1);

    let mut rng = thread_rng();
    let (mut loss_avg, mut acc_avg) = (ExponentialAverage::new(1.0), ExponentialAverage::new(0.0));
    let mut iter = 0;
    while acc_avg.value < 0.995 {
        // Generate problem
        let (problem, answer) = make_problem(&mut rng);

        // Get gradients
        input.set(problem);
        target.set(answer);
        grad_cx.execute();

        // Update weights
        transfer_data(&weights, &mut grad_cx, &old_weights, &mut opt_cx);
        transfer_data(&grads, &mut grad_cx, &opt_grads, &mut opt_cx);
        opt_cx.execute();
        transfer_data(&new_weights, &mut opt_cx, &weights, &mut grad_cx);

        // Report progress
        loss_avg.update(loss.data()[0]);
        loss.drop();
        acc_avg.update(
            output
                .data()
                .into_iter()
                .zip(answer)
                .filter(|(a, b)| (a - b).abs() < 0.5)
                .count() as f32
                / 5.,
        );
        output.drop();
        println!(
            "Iter {iter} Loss: {:.2} Acc: {:.2}",
            loss_avg.value, acc_avg.value
        );
        iter += 1;
    }
    println!("Finished in {iter} iterations");
}

// Generate data
fn make_problem(rng: &mut ThreadRng) -> ([f32; 8], [f32; 5]) {
    let (n1, n2): (u8, u8) = (rng.gen_range(0..16), rng.gen_range(0..16));
    let ans = n1.wrapping_add(n2);
    let mut p = [0.; 8];
    get_lower_bits(n1, 4, &mut p);
    get_lower_bits(n2, 4, &mut p[4..]);
    let mut a = [0.; 5];
    get_lower_bits(ans, 5, &mut a);
    (p, a)
}

fn get_lower_bits(byte: u8, bits: usize, slice: &mut [f32]) {
    #[allow(clippy::needless_range_loop)]
    for i in 0..bits {
        slice[i] = if byte >> i & 1 == 1 { 1.0 } else { 0.0 };
    }
}

// Smooth metrics
pub struct ExponentialAverage {
    beta: f32,
    moment: f32,
    pub value: f32,
    t: i32,
}

impl ExponentialAverage {
    fn new(initial: f32) -> Self {
        ExponentialAverage {
            beta: 0.999,
            moment: 0.,
            value: initial,
            t: 0,
        }
    }
}

impl ExponentialAverage {
    pub fn update(&mut self, value: f32) {
        self.t += 1;
        self.moment = self.beta * self.moment + (1. - self.beta) * value;
        // bias correction
        self.value = self.moment / (1. - f32::powi(self.beta, self.t));
    }

    pub fn reset(&mut self) {
        self.moment = 0.;
        self.value = 0.0;
        self.t = 0;
    }
}

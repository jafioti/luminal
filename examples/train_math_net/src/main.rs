use luminal::prelude::*;
use luminal_nn::{Linear, Swish};
use luminal_training::{mse_loss, sgd_on_graph, Autograd};
use rand::{rngs::ThreadRng, thread_rng, Rng};

// This is a simple example of using luminal to train.
// Here we are training an MLP to add 4 bit numbers together into a resultant 5 bit number.
// Run with the "metal" feature to compile to Metal backend with luminal_metal

fn main() {
    // Setup gradient graph
    let mut cx = Graph::new();
    let model = <(Linear<8, 16>, Swish, Linear<16, 16>, Swish, Linear<16, 5>)>::initialize(&mut cx);
    let mut input = cx.tensor::<R1<8>>();
    let mut target = cx.tensor::<R1<5>>();
    let mut output = model.forward(input).retrieve();
    let mut loss = mse_loss(output, target).retrieve();

    let mut weights = params(&model);
    let grads = cx.compile(Autograd::new(&weights, loss), ());
    let (mut new_weights, lr) = sgd_on_graph(&mut cx, &weights, &grads);
    cx.keep_tensors(&new_weights);
    cx.keep_tensors(&weights);
    lr.set(1e-1);

    #[cfg(all(not(feature = "metal"), not(feature = "cuda")))]
    cx.compile(
        GenericCompiler::default(),
        (
            &mut input,
            &mut target,
            &mut loss,
            &mut output,
            &mut weights,
            &mut new_weights,
        ),
    );

    #[cfg(feature = "metal")]
    cx.compile(
        luminal_metal::MetalCompiler::<f32>::default(),
        (
            &mut input,
            &mut target,
            &mut loss,
            &mut output,
            &mut weights,
            &mut new_weights,
        ),
    );

    #[cfg(feature = "cuda")]
    cx.compile(
        luminal_cuda::CudaCompiler::<f32>::default(),
        (
            &mut input,
            &mut target,
            &mut loss,
            &mut output,
            &mut weights,
            &mut new_weights,
        ),
    );

    let mut rng = thread_rng();
    let (mut loss_avg, mut acc_avg) = (ExponentialAverage::new(1.0), ExponentialAverage::new(0.0));
    let mut iter = 0;
    let start = std::time::Instant::now();
    while acc_avg.value < 0.995 {
        // Generate problem
        let (problem, answer) = make_problem(&mut rng);
        input.set(problem);
        target.set(answer);

        // Execute graph and update weights
        cx.execute();
        transfer_data_same_graph(&new_weights, &weights, &mut cx);

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
    println!(
        "Took {:.2}s, {:.2}µs / iter",
        start.elapsed().as_secs_f32(),
        start.elapsed().as_micros() / iter
    );
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

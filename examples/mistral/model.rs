use half::{bf16, f16};
use itertools::Itertools;
use luminal::{
    nn::{linear::Linear, norm::RMSNorm},
    prelude::*,
};
use memmap2::{Mmap, MmapOptions};
use rust_tokenizers::{
    error::TokenizerError,
    tokenizer::{SentencePieceBpeTokenizer, Tokenizer, TruncationStrategy},
};
use safetensors::{tensor::TensorView, SafeTensors};
use std::{
    collections::HashMap,
    fs::File,
    ops::{Add, Div},
};
use yoke::{Yoke, Yokeable};

// Mistral 7B Config
pub const VOCAB_SIZE: usize = 32000;
pub const HIDDEN_DIM: usize = 4096;
// pub const NUM_LAYERS: usize = 32;
pub const NUM_LAYERS: usize = 1;
pub const NUM_ATTENTION_HEADS: usize = 32;
pub const NUM_KV_HEADS: usize = 8;
pub const MLP_PROJECTION_DIM: usize = 14336;
pub const ROPE_THETA: f32 = 1000000.0;

pub const NUM_ATTENTION_GROUPS: usize = NUM_ATTENTION_HEADS / NUM_KV_HEADS;
pub const ATTENTION_HEAD_DIM: usize = HIDDEN_DIM / NUM_ATTENTION_HEADS;
pub const ATTENTION_HEAD_DIM_OVER_2: usize = ATTENTION_HEAD_DIM / 2;
pub const ATTENTION_PROJECTION_DIM: usize = ATTENTION_HEAD_DIM * NUM_KV_HEADS;

// Helper to deserialize safetensors stored in bf16
pub fn convert_vector_bf16_f32(tensor_view: &TensorView<'_>) -> Vec<f32> {
    // Get the data
    let data = tensor_view.data();

    // Create a mutable vector to store the final output
    let mut output: Vec<f32> = Vec::with_capacity(data.len() / 2);

    // Iterate over the raw buffer in chunks of 2 bytes
    for chunk in data.chunks_exact(2) {
        let value = bf16::from_le_bytes(chunk.try_into().unwrap()).to_f32();
        output.push(value);
    }

    output
}

// Rotary Embeddings
pub fn compute_rotary_embedding_frequencies<SequenceLength: Dimension>(
    graph: &mut Graph,
) -> (
    GraphTensor<(SequenceLength, Const<ATTENTION_HEAD_DIM>)>,
    GraphTensor<(SequenceLength, Const<ATTENTION_HEAD_DIM>)>,
) {
    let frequencies =
        (graph.arange::<Const<ATTENTION_HEAD_DIM>>() * 2.0) / (ATTENTION_HEAD_DIM as f32);

    let frequencies = frequencies
        .pow2(ROPE_THETA)
        .recip()
        .reshape::<R2<1, ATTENTION_HEAD_DIM>>();
    let t = graph
        .arange::<SequenceLength>()
        .reshape::<(SequenceLength, Const<1>)>();
    let frequencies = t.matmul(frequencies);

    let real = frequencies.cos();
    let imaginary = frequencies.sin();

    (real, imaginary)
}

pub fn apply_rotary_embeddings<
    Batch: Dimension,
    NumAttentionHeads: Dimension,
    SequenceLength: Dimension,
    const ATTENTION_HEAD_DIM: usize,
    const ATTENTION_HEAD_DIM_OVER_2: usize,
>(
    input: GraphTensor<(
        Batch,
        NumAttentionHeads,
        SequenceLength,
        Const<ATTENTION_HEAD_DIM>,
    )>,
    frequencies: (
        GraphTensor<(SequenceLength, Const<ATTENTION_HEAD_DIM_OVER_2>)>,
        GraphTensor<(SequenceLength, Const<ATTENTION_HEAD_DIM_OVER_2>)>,
    ),
) -> GraphTensor<(
    Batch,
    NumAttentionHeads,
    SequenceLength,
    Const<ATTENTION_HEAD_DIM>,
)> {
    let (real, imaginary) = frequencies;

    // Split into real and imaginary
    let input_expanded = input.reshape::<(
        Batch,
        NumAttentionHeads,
        SequenceLength,
        Const<ATTENTION_HEAD_DIM_OVER_2>,
        Const<2>,
    )>();

    let input_real = input_expanded
        .slice((.., .., .., .., ..1))
        .contiguous()
        .reshape::<(
            Batch,
            NumAttentionHeads,
            SequenceLength,
            Const<ATTENTION_HEAD_DIM_OVER_2>,
        )>();

    let input_imaginary = input_expanded
        .slice((.., .., .., .., 1..))
        .contiguous()
        .reshape::<(
            Batch,
            NumAttentionHeads,
            SequenceLength,
            Const<ATTENTION_HEAD_DIM_OVER_2>,
        )>();

    // x = a + bi, y = c + di
    // x * y = (ac - bd) + (ad + bc)i
    let (a, b) = (real, imaginary);
    let (c, d) = (input_real, input_imaginary);

    let output_real = (a.expand() * c) - (b.expand() * d);
    let output_imaginary = (a.expand() * d) + (b.expand() * c);

    // Finally, we put the real and imaginary together
    let output_real = output_real
        .reshape::<(
            Batch,
            NumAttentionHeads,
            SequenceLength,
            Const<ATTENTION_HEAD_DIM_OVER_2>,
            Const<1>,
        )>()
        .contiguous();
    let output_imaginary = output_imaginary
        .reshape::<(
            Batch,
            NumAttentionHeads,
            SequenceLength,
            Const<ATTENTION_HEAD_DIM_OVER_2>,
            Const<1>,
        )>()
        .contiguous();

    let output = output_real.concat_along::<(
        Batch,
        NumAttentionHeads,
        SequenceLength,
        Const<ATTENTION_HEAD_DIM_OVER_2>,
        Const<2>,
    ), Axis<4>, _>(output_imaginary);

    output.reshape()
}

// Create the self-Attention layer
pub struct Attention {
    pub q_proj: GraphTensor<R2<HIDDEN_DIM, HIDDEN_DIM>>,
    // pub k_proj: GraphTensor<R2<HIDDEN_DIM, ATTENTION_PROJECTION_DIM>>,
    pub k_proj: GraphTensor<R2<ATTENTION_PROJECTION_DIM, HIDDEN_DIM>>,
    pub v_proj: GraphTensor<R2<ATTENTION_PROJECTION_DIM, HIDDEN_DIM>>,
    // pub v_proj: GraphTensor<R2<HIDDEN_DIM, ATTENTION_PROJECTION_DIM>>,
    pub o_proj: GraphTensor<R2<HIDDEN_DIM, HIDDEN_DIM>>,
}

impl Attention {
    // Helper to get a graph
    fn graph(&self) -> &mut Graph {
        self.q_proj.graph()
    }

    // Forward method
    fn forward<Batch: Dimension, SequenceLength: Dimension>(
        &self,
        x: GraphTensor<(Batch, SequenceLength, Const<HIDDEN_DIM>)>,
    ) -> GraphTensor<(Batch, SequenceLength, Const<HIDDEN_DIM>)> {
        let xq = x
            .matmul(self.q_proj)
            .reshape::<(
                Batch,
                SequenceLength,
                Const<NUM_ATTENTION_HEADS>,
                Const<ATTENTION_HEAD_DIM>,
            )>()
            .permute::<_, Axes4<0, 2, 1, 3>>();

        // let xk = x
        //     .matmul(self.k_proj)
        //     .reshape::<(
        //         Batch,
        //         SequenceLength,
        //         Const<NUM_KV_HEADS>,
        //         Const<ATTENTION_HEAD_DIM>,
        //     )>()
        //     .permute::<_, Axes4<0, 2, 1, 3>>();
        let xk = x.matmul(self.k_proj.permute());
        let xk = xk
            .reshape::<(
                Batch,
                SequenceLength,
                Const<NUM_KV_HEADS>,
                Const<ATTENTION_HEAD_DIM>,
            )>()
            .permute::<_, Axes4<0, 2, 1, 3>>();

        let xv = x.matmul(self.v_proj.permute());
        let xv = xv
            .reshape::<(
                Batch,
                SequenceLength,
                Const<NUM_KV_HEADS>,
                Const<ATTENTION_HEAD_DIM>,
            )>()
            .permute::<_, Axes4<0, 2, 1, 3>>();

        // We apply rotary embeddings
        let rotary_frequencies =
            compute_rotary_embedding_frequencies::<SequenceLength>(&mut self.graph());
        let xq = apply_rotary_embeddings(xq, rotary_frequencies);
        let xk = apply_rotary_embeddings(xk, rotary_frequencies);

        // We repeat xv and xk to match the size of xq
        let xk = xk
            .expand::<(
                Batch,
                Const<NUM_KV_HEADS>,
                Const<NUM_ATTENTION_GROUPS>,
                SequenceLength,
                Const<ATTENTION_HEAD_DIM>,
            ), Axis<2>>()
            .reshape::<(
                Batch,
                SequenceLength,
                Const<NUM_ATTENTION_HEADS>,
                Const<ATTENTION_HEAD_DIM>,
            )>();

        let xv = xv
            .expand::<(
                Batch,
                Const<NUM_KV_HEADS>,
                Const<NUM_ATTENTION_GROUPS>,
                SequenceLength,
                Const<ATTENTION_HEAD_DIM>,
            ), Axis<2>>()
            .reshape::<(
                Batch,
                SequenceLength,
                Const<NUM_ATTENTION_HEADS>,
                Const<ATTENTION_HEAD_DIM>,
            )>();

        // Attention mask
        let attention_mask =
            self.graph().triu::<SequenceLength, SequenceLength>(1) * f16::MIN.to_f32();

        // Finally we compute the outputs (attention calculation)
        let xo = xq
            .matmul(xk.permute())
            .div((ATTENTION_HEAD_DIM as f64).sqrt() as f32)
            .add(attention_mask.expand())
            .softmax::<3>()
            .matmul(xv.permute())
            .permute::<(
                Batch,
                SequenceLength,
                Const<NUM_ATTENTION_HEADS>,
                Const<ATTENTION_HEAD_DIM>,
            ), _>()
            .dyn_reshape::<(Batch, SequenceLength, Const<HIDDEN_DIM>)>(vec![
                Batch::const_size(),
                SequenceLength::const_size(),
                HIDDEN_DIM.into(),
            ])
            .matmul(self.o_proj.permute());

        xo
    }
}

// Create the FeedForward Layer
pub struct FeedForward {
    pub gate_proj: GraphTensor<R2<MLP_PROJECTION_DIM, HIDDEN_DIM>>,
    pub down_proj: GraphTensor<R2<HIDDEN_DIM, MLP_PROJECTION_DIM>>,
    pub up_proj: GraphTensor<R2<MLP_PROJECTION_DIM, HIDDEN_DIM>>,
}

impl FeedForward {
    fn forward<Batch: Dimension, SequenceLength: Dimension>(
        &self,
        x: GraphTensor<(Batch, SequenceLength, Const<HIDDEN_DIM>)>,
    ) -> GraphTensor<(Batch, SequenceLength, Const<HIDDEN_DIM>)> {
        (x.matmul(self.gate_proj.permute()).swish() * x.matmul(self.up_proj.permute()))
            .matmul(self.down_proj.permute())
    }
}

// Create the Transformer Block
pub struct TransformerBlock {
    pub attention: Attention,
    pub attention_norm: RMSNorm<HIDDEN_DIM>,
    pub feed_forward: FeedForward,
    pub feed_forward_norm: RMSNorm<HIDDEN_DIM>,
}

// impl TransformerBlock {
//     fn forward<Batch: Dimension, SequenceLength: Dimension>(
//         &self,
//         x: GraphTensor<(Batch, SequenceLength, Const<HIDDEN_DIM>)>,
//     ) -> GraphTensor<(Batch, SequenceLength, Const<HIDDEN_DIM>)> {
//         let r = self.attention.forward(self.attention_norm.forward(x));
//         let h = x + r;
//         let r = self.feed_forward.forward(self.feed_forward_norm.forward(h));
//         h + r
//     }
// }

pub struct Mistral {
    // Graph
    pub graph: Box<Graph>,

    // Tokenizer
    pub tokenizer: SentencePieceBpeTokenizer,

    // Embedding
    pub embedding: GraphTensor<R2<VOCAB_SIZE, HIDDEN_DIM>>,

    // Transformer Layers
    pub transformer_layers: Vec<TransformerBlock>,

    // Final Norm layer
    // pub norm: RMSNorm<HIDDEN_DIM>,

    // LM Head Layer
    // pub lm_head: Linear<HIDDEN_DIM, VOCAB_SIZE>,

    // Input Layer
    pub input: GraphTensor<(Const<1>, Dyn<'s'>)>,
}

#[derive(Yokeable)]
struct SafeTensors_<'a>(SafeTensors<'a>);

fn get_tensor<'a>(
    weight_file_mapper: &'a HashMap<String, String>,
    file_tensor_mapper: &'a HashMap<String, Yoke<SafeTensors_<'static>, Mmap>>,
    weight_name: &'a str,
) -> Result<TensorView<'a>, String> {
    let file_path = weight_file_mapper
        .get(weight_name)
        .ok_or("Weight not found".to_string())?;

    let tensors = file_tensor_mapper
        .get(file_path)
        .ok_or("Tensors not found".to_string())?;

    let tensor = tensors
        .get()
        .0
        .tensor(weight_name)
        .map_err(|e| e.to_string())?;

    Ok(tensor)
}

impl Mistral {
    pub fn debug_run(&mut self) {
        // Set a value for the input
        let text = "Hello, how are";
        let token_ids = self.encode(text);
        let n_tokens = token_ids.len();
        self.input.set_dyn(token_ids, vec![1, n_tokens]);

        // Forward pass

        // Embeddings is correct
        let hidden_states = self.embedding.gather(self.input);

        // Transformer Layer
        // let hidden_states = self.transformer_layers[0].forward(hidden_states);

        let hidden_states = self.transformer_layers[0]
            .attention_norm
            .forward(hidden_states);

        // let eps = self.transformer_layers[0].attention_norm.epsilon;
        // println!("eps: {eps}");

        let hidden_states = self.transformer_layers[0].attention.forward(hidden_states);
        // let q_proj = self.transformer_layers[0].attention.q_proj;
        // let query_states = hidden_states.matmul(q_proj.permute());

        // let k_proj = self.transformer_layers[0].attention.k_proj;
        // let key_states = hidden_states.matmul(k_proj.permute());

        // let v_proj = self.transformer_layers[0].attention.v_proj;
        // // let value_states = hidden_states.matmul(v_proj.permute());

        // // Now, let's compute the rotary embeddings
        // let (cos, sin) = compute_rotary_embedding_frequencies::<Dyn<'s'>>(&mut self.graph);

        // let x = sin / 2.0;

        // // Compute rotary embeddings directly here
        // let x = self.graph.constant(ATTENTION_HEAD_DIM as f32);
        // let frequencies = self.graph.arange::<Const<ATTENTION_HEAD_DIM>>();
        // let frequencies = frequencies * 2.0;
        // let frequencies = frequencies * (x.recip().expand());

        // let frequencies = frequencies
        //     .pow2(ROPE_THETA)
        //     .recip()
        //     .reshape::<R2<1, ATTENTION_HEAD_DIM>>();
        // let t = self
        //     .graph
        //     .arange::<Dyn<'s'>>()
        //     .reshape::<(Dyn<'s'>, Const<1>)>();
        // let frequencies = t.matmul(frequencies);
        // let cos = frequencies.cos();
        // let sin = frequencies.sin();

        // let frequencies = frequencies.recip();
        // frequencies.retrieve();
        /*
            let frequencies =
            (graph.arange::<Const<HIDDEN_DIM_OVER_2>>() * 2.0) / (HIDDEN_DIM_OVER_2 as f32 * 2.0);
        let frequencies = frequencies
            .pow2(ROPE_THETA)
            .recip()
            .reshape::<R2<1, HIDDEN_DIM_OVER_2>>();
        let t = graph
            .arange::<SequenceLength>()
            .reshape::<(SequenceLength, Const<1>)>();
        let frequencies = t.matmul(frequencies);

        let real = frequencies.cos();
        let imaginary = frequencies.sin();

        (real, imaginary)

             */

        // q_proj.retrieve();
        // k_proj.retrieve();
        // query_states.retrieve();
        // key_states.retrieve();
        // value_states.retrieve();
        hidden_states.retrieve();
        // sin.retrieve();

        // Compile the graph
        self.graph.compile(<(
            PreGenericCompiler,
            MetalFp16Compiler,
            // CPUCompiler,
            PostGenericCompiler,
        )>::default());

        // self.graph.display();

        // Execute the graph
        self.graph.execute_debug();

        // println!("input: {:?}", self.input);
        // println!("embedding: {:?}", self.embedding);
        println!("hidden_states: {:?}", hidden_states);
        // println!("token_ids_one_hot: {:?}", token_ids_one_hot);
        // println!("input_layer_norm: {:?}", input_layer_norm);
        // println!("query_states: {:?}", query_states);
        // println!("k_proj: {:?}", k_proj);
        // println!("key_states: {:?}", key_states);
        // println!("value_states: {:?}", value_states);
        // println!("rotary_frequencies (sin): {:?}", sin);
        // println!("frequencies {:?}", frequencies);
    }

    // Infer next token
    // pub fn infer_next_token(
    //     &mut self,
    //     output_token_ids: GraphTensor<(Const<1>, Dyn<'s'>)>,
    //     text: &str,
    // ) -> String {
    //     // First, we encode the text
    //     let token_ids = self.encode(text);
    //     let n_tokens = token_ids.len();

    //     // Insert the data in the input node
    //     self.input.set_dyn(token_ids, vec![1, n_tokens]);

    //     // Execute the graph
    //     self.graph.execute_debug();

    //     println!("Token IDs: {:?}", output_token_ids);

    //     // Pull the data from the output node
    //     let output_token_ids = output_token_ids.data();

    //     let output_text = self.decode(output_token_ids);

    //     output_text
    // }

    // pub fn build_forward_graph(&mut self) -> GraphTensor<(Const<1>, Dyn<'s'>)> {
    //     let output_probabilities = self.forward(self.input);

    //     // Do the sampling in the graph computation
    //     let output_token_ids = output_probabilities.argmax();
    //     output_token_ids.retrieve();

    //     output_token_ids
    // }

    pub fn compile_forward_graph(&mut self) {
        self.graph
            .compile(<(PreGenericCompiler, MetalFp16Compiler, PostGenericCompiler)>::default());
    }

    // Forward pass
    // pub fn forward<Batch: Dimension, SequenceLength: Dimension>(
    //     &mut self,
    //     input: GraphTensor<(Batch, SequenceLength)>,
    // ) -> GraphTensor<(Batch, SequenceLength, Const<VOCAB_SIZE>)> {
    //     // First compute the embeddings
    //     let mut hidden_states = self.embedding.gather(input);
    //     hidden_states.retrieve();
    //     hidden_states.print("Embedding");

    //     // Loop over transformer layers
    //     for (layer_index, transformer_layer) in self.transformer_layers.iter().enumerate() {
    //         hidden_states = transformer_layer.forward(hidden_states);
    //         hidden_states.retrieve();
    //         hidden_states.print(format!("Layer {layer_index} Output").as_str());
    //     }

    //     // Normalize
    //     hidden_states = self.norm.forward(hidden_states);
    //     hidden_states.print("Norm Output");

    //     // Compute output
    //     self.lm_head.forward(hidden_states)
    // }

    // Initializer
    pub fn new(tokenizer_path: &str) -> Result<Self, TokenizerError> {
        // Load the tokenizer
        let tokenizer = SentencePieceBpeTokenizer::from_file(tokenizer_path, false)?;

        // Create the graph
        let mut graph = Box::new(Graph::new());

        // Create the embedding
        let embedding = graph.named_tensor("embedding");

        // Create the transformer layers
        let transformer_layers = (0..NUM_LAYERS)
            .map(|i| {
                let mut attention_norm = RMSNorm::initialize(graph.as_mut());
                attention_norm.epsilon = 1e-5;

                let mut feed_forward_norm = RMSNorm::initialize(graph.as_mut());
                feed_forward_norm.epsilon = 1e-5;

                TransformerBlock {
                    attention: Attention {
                        q_proj: graph.named_tensor(format!("layers.{i}.attention.q_proj").as_str()),
                        k_proj: graph.named_tensor(format!("layers.{i}.attention.k_proj").as_str()),
                        v_proj: graph.named_tensor(format!("layers.{i}.attention.v_proj").as_str()),
                        o_proj: graph.named_tensor(format!("layers.{i}.attention.o_proj").as_str()),
                    },
                    attention_norm: attention_norm,
                    feed_forward: FeedForward {
                        gate_proj: graph.named_tensor(format!("layers.{i}.mlp.gate_proj").as_str()),
                        down_proj: graph.named_tensor(format!("layers.{i}.mlp.down_proj").as_str()),
                        up_proj: graph.named_tensor(format!("layers.{i}.mlp.up_proj").as_str()),
                    },
                    feed_forward_norm: feed_forward_norm,
                }
            })
            .collect_vec();

        // // Create the norm
        // let norm = RMSNorm::initialize(graph.as_mut());

        // // Create the lm head
        // let lm_head = Linear::initialize(graph.as_mut());

        // // Create the input node
        let input = graph.named_tensor::<(Const<1>, Dyn<'s'>)>("input_node");

        Ok(Self {
            tokenizer,
            graph,
            embedding,
            transformer_layers,
            // norm,
            // lm_head,
            input,
        })
    }

    // Method to encode text as vector
    pub fn encode(&self, text: &str) -> Vec<f32> {
        let mut vector = self
            .tokenizer
            .encode(text, None, text.len(), &TruncationStrategy::LongestFirst, 0)
            .token_ids
            .iter()
            .map(|&x| x as f32)
            .collect_vec();

        vector.insert(0, 1.0); // Start token

        vector
    }

    // Method to decode tokens as text
    pub fn decode(&self, token_ids: Vec<f32>) -> String {
        let binding = token_ids.iter().map(|i| *i as i64).collect_vec();
        let token_ids = binding.as_slice();

        self.tokenizer.decode(token_ids, true, false)
    }

    pub unsafe fn load_safe_tensors_from_files(
        &mut self,
        file_paths: Vec<String>,
    ) -> Result<(), String> {
        let mut weight_file_mapper = HashMap::new();
        let mut file_tensor_mapper = HashMap::new();

        // First, we read the tensors from all the files
        for file_path in file_paths.iter() {
            let file = File::open(file_path).map_err(|e| e.to_string())?;
            let buffer = MmapOptions::new().map(&file).map_err(|e| e.to_string())?;
            let tensors =
                Yoke::<SafeTensors_<'static>, Mmap>::try_attach_to_cart(buffer, |data| {
                    let tensors = SafeTensors::deserialize(data).map_err(|e| e.to_string())?;
                    Ok::<_, String>(SafeTensors_(tensors))
                })?;

            for name in tensors.get().0.names() {
                weight_file_mapper.insert(name.to_string(), file_path.to_string());
            }

            file_tensor_mapper.insert(file_path.to_string(), tensors);
        }

        // Then we populate the layers

        // Layer: Embeddings
        let embeddings_safe_tensor = &get_tensor(
            &weight_file_mapper,
            &file_tensor_mapper,
            "model.embed_tokens.weight",
        )?;
        let embeddings = convert_vector_bf16_f32(embeddings_safe_tensor);
        self.embedding.set(embeddings);

        // Layers: Transformer Layers
        for layer_index in 0..self.transformer_layers.len() {
            // We populate each layer here
            let weight_prefix = format!("model.layers.{layer_index}");

            // Attention
            let weight_name = format!("{weight_prefix}.self_attn.q_proj.weight");
            let q_proj_safe_tensor = &get_tensor(
                &weight_file_mapper,
                &file_tensor_mapper,
                weight_name.as_str(),
            )?;

            let q_proj = convert_vector_bf16_f32(q_proj_safe_tensor);
            self.transformer_layers[layer_index]
                .attention
                .q_proj
                .set(q_proj);

            let weight_name = format!("{weight_prefix}.self_attn.k_proj.weight");
            let k_proj_safe_tensor = &get_tensor(
                &weight_file_mapper,
                &file_tensor_mapper,
                weight_name.as_str(),
            )?;
            let k_proj = convert_vector_bf16_f32(k_proj_safe_tensor);
            self.transformer_layers[layer_index]
                .attention
                .k_proj
                .set(k_proj);

            let weight_name = format!("{weight_prefix}.self_attn.v_proj.weight");
            let v_proj_safe_tensor = &get_tensor(
                &weight_file_mapper,
                &file_tensor_mapper,
                weight_name.as_str(),
            )?;
            let v_proj = convert_vector_bf16_f32(v_proj_safe_tensor);
            self.transformer_layers[layer_index]
                .attention
                .v_proj
                .set(v_proj);

            let weight_name = format!("{weight_prefix}.self_attn.o_proj.weight");
            let o_proj_safe_tensor = &get_tensor(
                &weight_file_mapper,
                &file_tensor_mapper,
                weight_name.as_str(),
            )?;
            let o_proj = convert_vector_bf16_f32(o_proj_safe_tensor);
            self.transformer_layers[layer_index]
                .attention
                .o_proj
                .set(o_proj);

            // Norms
            let weight_name = format!("{weight_prefix}.input_layernorm.weight");
            let attention_norm_safe_tensor = &get_tensor(
                &weight_file_mapper,
                &file_tensor_mapper,
                weight_name.as_str(),
            )?;
            let attention_norm = convert_vector_bf16_f32(attention_norm_safe_tensor);
            self.transformer_layers[layer_index]
                .attention_norm
                .weight
                .set(attention_norm);

            let weight_name = format!("{weight_prefix}.post_attention_layernorm.weight");
            let feed_forward_norm_safe_tensor = &get_tensor(
                &weight_file_mapper,
                &file_tensor_mapper,
                weight_name.as_str(),
            )?;
            let feed_forward_norm = convert_vector_bf16_f32(feed_forward_norm_safe_tensor);
            self.transformer_layers[layer_index]
                .feed_forward_norm
                .weight
                .set(feed_forward_norm);

            // Feed Forward
            let weight_name = format!("{weight_prefix}.mlp.gate_proj.weight");
            let gate_proj_safe_tensor = &get_tensor(
                &weight_file_mapper,
                &file_tensor_mapper,
                weight_name.as_str(),
            )?;
            let gate_proj = convert_vector_bf16_f32(gate_proj_safe_tensor);
            self.transformer_layers[layer_index]
                .feed_forward
                .gate_proj
                .set(gate_proj);

            let weight_name = format!("{weight_prefix}.mlp.down_proj.weight");
            let down_proj_safe_tensor = &get_tensor(
                &weight_file_mapper,
                &file_tensor_mapper,
                weight_name.as_str(),
            )?;
            let down_proj = convert_vector_bf16_f32(down_proj_safe_tensor);
            self.transformer_layers[layer_index]
                .feed_forward
                .down_proj
                .set(down_proj);

            let weight_name = format!("{weight_prefix}.mlp.up_proj.weight");
            let up_proj_safe_tensor = &get_tensor(
                &weight_file_mapper,
                &file_tensor_mapper,
                weight_name.as_str(),
            )?;
            let up_proj = convert_vector_bf16_f32(up_proj_safe_tensor);
            self.transformer_layers[layer_index]
                .feed_forward
                .up_proj
                .set(up_proj);
        }

        // // Layer: Norm
        // let norm_safe_tensor = &get_tensor(
        //     &weight_file_mapper,
        //     &file_tensor_mapper,
        //     "model.norm.weight",
        // )?;
        // let norm = convert_vector_bf16_f32(norm_safe_tensor);
        // self.norm.weight.set(norm);

        // // Layer: LM Head
        // let lm_head_safe_tensor =
        //     &get_tensor(&weight_file_mapper, &file_tensor_mapper, "lm_head.weight")?;
        // let lm_head = convert_vector_bf16_f32(lm_head_safe_tensor);
        // self.lm_head.weight.set(lm_head);

        Ok(())
    }
}

// impl SerializeModule for Mistral {
//     fn serialize(&self, s: &mut Serializer) {
//         s.module("norm", &self.norm);
//         s.module("lm_head/weight", &self.lm_head);
//     }
// }

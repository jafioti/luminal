use colored::Colorize;
use half::bf16;
use itertools::Itertools;
use luminal::{nn::norm::RMSNorm, op::Function, prelude::*, shape::symbolic::Expression};
use memmap2::MmapOptions;
use rust_tokenizers::{
    error::TokenizerError,
    tokenizer::{SentencePieceBpeTokenizer, Tokenizer, TruncationStrategy},
};
use safetensors::{tensor::TensorView, SafeTensors};
use std::{collections::HashMap, fs::File};

// Mistral 7B Config
pub const VOCAB_SIZE: usize = 32000;
pub const HIDDEN_DIM: usize = 4096;
pub const NUM_LAYERS: usize = 32;
// pub const NUM_LAYERS: usize = 1;
pub const NUM_ATTENTION_HEADS: usize = 32;
pub const NUM_KV_HEADS: usize = 8;
pub const MLP_PROJECTION_DIM: usize = 14336;
pub const ROPE_THETA: f32 = 1000000.0;

// For some reason, setting it to 32k doesn't work
// Setting it to a different value should affect computations
// pub const MAX_POSITION_EMBEDDINGS: usize = 32768;
pub const MAX_POSITION_EMBEDDINGS: usize = 10;

pub const NUM_ATTENTION_GROUPS: usize = NUM_ATTENTION_HEADS / NUM_KV_HEADS;
pub const ATTENTION_HEAD_DIM: usize = HIDDEN_DIM / NUM_ATTENTION_HEADS;
pub const ATTENTION_HEAD_DIM_OVER_2: usize = ATTENTION_HEAD_DIM / 2;
pub const ATTENTION_PROJECTION_DIM: usize = ATTENTION_HEAD_DIM * NUM_KV_HEADS;

// From examples/llama
// Currently just an argmax, do actual sampling here
fn sample_index(dist: &[f32]) -> usize {
    dist.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap()
        .0
}

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
        (graph.arange::<Const<ATTENTION_HEAD_DIM_OVER_2>>() * 2.0) / (ATTENTION_HEAD_DIM as f32);

    let frequencies = frequencies
        .pow2(ROPE_THETA)
        .recip()
        .reshape::<R2<1, ATTENTION_HEAD_DIM_OVER_2>>();
    let frequencies = frequencies.concat_along::<_, Axis<1>, _>(frequencies);
    let t = graph
        .arange::<SequenceLength>()
        .reshape::<(SequenceLength, Const<1>)>();
    let frequencies = t.matmul(frequencies);

    let cos = frequencies.cos();
    let sin = frequencies.sin();

    (cos, sin)
}

pub fn rotate_half<
    Batch: Dimension,
    SequenceLength: Dimension,
    NumAttentionHeads: Dimension,
    const ATTENTION_HEAD_DIM: usize,
>(
    x: GraphTensor<(
        Batch,
        SequenceLength,
        NumAttentionHeads,
        Const<ATTENTION_HEAD_DIM>,
    )>,
) -> GraphTensor<(
    Batch,
    SequenceLength,
    NumAttentionHeads,
    Const<ATTENTION_HEAD_DIM>,
)> {
    let x1 = x
        .slice((.., .., .., ..Expression::from(ATTENTION_HEAD_DIM / 2)))
        .contiguous();
    let x2 = x
        .slice((.., .., .., Expression::from(ATTENTION_HEAD_DIM / 2)..))
        .contiguous();

    (-x2).concat_along::<_, Axis<3>, _>(x1)
}

pub fn apply_rotary_embeddings<
    Batch: Dimension,
    SequenceLength: Dimension,
    NumAttentionHeads: Dimension,
    const ATTENTION_HEAD_DIM: usize,
>(
    input: GraphTensor<(
        Batch,
        NumAttentionHeads,
        SequenceLength,
        Const<ATTENTION_HEAD_DIM>,
    )>,
    frequencies: (
        GraphTensor<(SequenceLength, Const<ATTENTION_HEAD_DIM>)>,
        GraphTensor<(SequenceLength, Const<ATTENTION_HEAD_DIM>)>,
    ),
) -> GraphTensor<(
    Batch,
    NumAttentionHeads,
    SequenceLength,
    Const<ATTENTION_HEAD_DIM>,
)> {
    let (cos, sin) = frequencies;

    let input_half: GraphTensor<(
        Batch,
        NumAttentionHeads,
        SequenceLength,
        Const<ATTENTION_HEAD_DIM>,
    )> = rotate_half::<_, _, _, ATTENTION_HEAD_DIM>(input);

    (input * cos.expand()) + (input_half * sin.expand())
}

// Create the self-Attention layer
pub struct Attention {
    pub q_proj: GraphTensor<R2<HIDDEN_DIM, HIDDEN_DIM>>,
    pub k_proj: GraphTensor<R2<ATTENTION_PROJECTION_DIM, HIDDEN_DIM>>,
    pub v_proj: GraphTensor<R2<ATTENTION_PROJECTION_DIM, HIDDEN_DIM>>,
    pub o_proj: GraphTensor<R2<HIDDEN_DIM, HIDDEN_DIM>>,
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
    pub norm: RMSNorm<HIDDEN_DIM>,

    // LM Head Layer
    pub lm_head: GraphTensor<R2<VOCAB_SIZE, HIDDEN_DIM>>,

    // RoPE Embeddings
    pub rope_embeddings: (
        GraphTensor<R2<MAX_POSITION_EMBEDDINGS, ATTENTION_HEAD_DIM>>,
        GraphTensor<R2<MAX_POSITION_EMBEDDINGS, ATTENTION_HEAD_DIM>>,
    ),

    // Input Layer
    pub input: GraphTensor<(Const<1>, Dyn<'s'>)>,
}

pub fn slice_last_sequence<Batch: Dimension, SequenceLength: Dimension, Hidden: Dimension>(
    input: GraphTensor<(Batch, SequenceLength, Hidden)>,
) -> GraphTensor<(Batch, Const<1>, Hidden)> {
    let sequence_length = &input.shape.shape()[1];
    let sequence_length = Expression::from(sequence_length.clone());

    input
        .slice((.., (sequence_length - 1).., ..))
        .contiguous()
        .realize()
}

impl Mistral {
    pub fn debug_run(&mut self, prompt: &str) {
        // println!("Encoding Prompt");
        let input_data: Vec<f32> = self.encode(prompt);
        let sequence_length = input_data.len();
        // self.input.set_dyn(input_data, vec![1, sequence_length]);
        // println!("Defining Computation Graph");
        let input_embeddings = self.embedding.gather(self.input);
        let hidden_states = self.forward(input_embeddings);
        let logits = hidden_states.matmul(self.lm_head.permute()).retrieve();

        // Get the output ids
        // let output_ids = logits.argmax().retrieve();
        // let last_logit = logits
        //     .slice((.., Expression::from(sequence_length - 1).., ..))
        //     .contiguous()
        //     .reshape::<R1<VOCAB_SIZE>>()
        //     .retrieve();

        let last_logit = slice_last_sequence(logits)
            .reshape::<R1<VOCAB_SIZE>>()
            .retrieve();

        // println!("Compiling Graph");
        self.compile_forward_graph();

        let mut context_vector = self.encode(prompt);
        let mut completion = String::new();
        let max_new_tokens = 4;
        for i in 0..max_new_tokens {
            println!("########################### Iteration {i} ###########################");
            let _context = context_vector.clone();
            let sequence_length = _context.len();
            // self.graph.set_dyn_dim('s', sequence_length);
            println!("Sequence Length: {sequence_length}");
            println!("Input Vectors: {:?}", _context);
            self.input.set_dyn(_context, vec![1, sequence_length]);

            // println!("Executing Graph");
            // self.graph.execute_debug();
            self.graph.execute();

            println!("------- Last Logits -------");
            let last_logit_data = last_logit.data();
            let output_ids = sample_index(&last_logit_data);
            println!("Last Logits Shape: {:?}", last_logit.shape);
            println!("Last Logits Data Length: {:?}", last_logit_data.len());
            context_vector.push(output_ids as f32);

            // Decode token
            let output_token = self.decode(vec![output_ids as f32]);
            // context.push_str(output_token.as_str());
            completion.push_str(&output_token);

            println!("------- Logits -------");
            println!("Logits Shape: {:?}", logits.shape);
            println!("Logits Data Length: {:?}", logits.data().len());
            println!("Logits: {:?}", logits);
            // println!(
            //     "Tokens: {:?}",
            //     context_vector.iter().map(|x| *x as u32).collect_vec()
            // );

            println!("{}{}", prompt.on_black().white().bold(), completion.green());

            logits.drop();
            last_logit.drop();
        }
    }

    pub fn forward<Batch: Dimension, SequenceLength: Dimension>(
        &mut self,
        input_embeddings: GraphTensor<(Batch, SequenceLength, Const<HIDDEN_DIM>)>,
    ) -> GraphTensor<(Batch, SequenceLength, Const<HIDDEN_DIM>)> {
        // Get the sequence length as a run-time value
        let sequence_length = &input_embeddings.shape.shape()[1];

        // Start with the initial hidden states as the inputs
        let mut hidden_states = input_embeddings;

        // Extract the Rotary Embeddings
        let (cos, sin) = self.rope_embeddings;
        let cos = cos
            .slice((..Expression::from(sequence_length.clone()), ..))
            .contiguous();
        let sin = sin
            .slice((..Expression::from(sequence_length.clone()), ..))
            .contiguous();

        let rotary_embeddings = (cos.realize(), sin.realize());

        // Now, we loop over all layers
        for layer_index in 0..NUM_LAYERS {
            // Pull the weights we need
            let transformer_layer = &self.transformer_layers[layer_index];
            let q_proj = transformer_layer.attention.q_proj;
            let k_proj = transformer_layer.attention.k_proj;
            let v_proj = transformer_layer.attention.v_proj;
            let o_proj = transformer_layer.attention.o_proj;

            // Pre-attention norm
            let mut attention_output = transformer_layer.attention_norm.forward(hidden_states);

            // Prepare key, query, and value states for attention
            let query_states = attention_output.matmul(q_proj.permute());
            let key_states = attention_output.matmul(k_proj.permute());
            let value_states = attention_output.matmul(v_proj.permute());

            // Reshape
            let query_states = query_states
                .reshape::<(
                    Batch,
                    SequenceLength,
                    Const<NUM_ATTENTION_HEADS>,
                    Const<ATTENTION_HEAD_DIM>,
                )>()
                .permute::<_, Axes4<0, 2, 1, 3>>();

            let key_states = key_states
                .reshape::<(
                    Batch,
                    SequenceLength,
                    Const<NUM_KV_HEADS>,
                    Const<ATTENTION_HEAD_DIM>,
                )>()
                .permute::<_, Axes4<0, 2, 1, 3>>();

            let value_states = value_states
                .reshape::<(
                    Batch,
                    SequenceLength,
                    Const<NUM_KV_HEADS>,
                    Const<ATTENTION_HEAD_DIM>,
                )>()
                .permute::<_, Axes4<0, 2, 1, 3>>();

            // Apply the embeddings
            let query_states = apply_rotary_embeddings::<_, _, _, ATTENTION_HEAD_DIM>(
                query_states,
                rotary_embeddings,
            );

            let key_states = apply_rotary_embeddings::<_, _, _, ATTENTION_HEAD_DIM>(
                key_states,
                rotary_embeddings,
            );

            // Repeat the KV States for Grouped-Query Attention
            let key_states = key_states
                .expand::<(
                    Batch,
                    Const<NUM_KV_HEADS>,
                    Const<NUM_ATTENTION_GROUPS>,
                    SequenceLength,
                    Const<ATTENTION_HEAD_DIM>,
                ), Axis<2>>()
                .reshape::<(
                    Batch,
                    Const<NUM_ATTENTION_HEADS>,
                    SequenceLength,
                    Const<ATTENTION_HEAD_DIM>,
                )>()
                .permute::<_, Axes4<0, 1, 3, 2>>();

            let value_states = value_states
                .expand::<(
                    Batch,
                    Const<NUM_KV_HEADS>,
                    Const<NUM_ATTENTION_GROUPS>,
                    SequenceLength,
                    Const<ATTENTION_HEAD_DIM>,
                ), Axis<2>>()
                .reshape::<(
                    Batch,
                    Const<NUM_ATTENTION_HEADS>,
                    SequenceLength,
                    Const<ATTENTION_HEAD_DIM>,
                )>();

            // FINALLY, we actually compute attention
            let attention_weights = query_states.matmul(key_states);
            let attention_weights = attention_weights * (ATTENTION_HEAD_DIM as f32).sqrt().recip();

            // Attention Mask
            // let attention_mask =
            // self.graph.triu::<SequenceLength, SequenceLength>(1) * f16::MIN.to_f32();
            // let attention_mask = attention_mask.log();
            // let attention_weights = attention_weights + attention_mask.expand();

            let attention_weights = attention_weights.softmax::<3>();

            attention_output = attention_weights
                .matmul(value_states)
                .permute::<_, Axes4<0, 2, 1, 3>>()
                .reshape::<(Batch, SequenceLength, Const<HIDDEN_DIM>)>();

            attention_output = attention_output.matmul(o_proj.permute());

            // Add the residual
            hidden_states = hidden_states + attention_output;

            // Now for the projection layer
            let mut projection_layer_output =
                transformer_layer.feed_forward_norm.forward(hidden_states);

            projection_layer_output = transformer_layer
                .feed_forward
                .forward(projection_layer_output);

            // Add the residual
            hidden_states = hidden_states + projection_layer_output;
        }

        // Finally, we call the final norm
        hidden_states = self.norm.forward(hidden_states);

        // Return
        hidden_states
    }

    pub fn compile_forward_graph(&mut self) {
        self.graph
            .compile(<(PreGenericCompiler, MetalFp16Compiler, PostGenericCompiler)>::default());
    }

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

        // Create the norm
        let norm = RMSNorm::initialize(graph.as_mut());

        // Create the lm head
        let lm_head = graph.named_tensor("lm_head");

        // // Create the input node
        let input = graph.named_tensor::<(Const<1>, Dyn<'s'>)>("input_node");

        // Create the throwaway rope graph (just to compute RoPE)
        let mut rope_graph = Box::new(Graph::new());

        let (rope_cos, rope_sin) =
            compute_rotary_embedding_frequencies::<Const<MAX_POSITION_EMBEDDINGS>>(&mut rope_graph);

        rope_cos.retrieve();
        rope_sin.retrieve();

        rope_graph
            .compile(<(PreGenericCompiler, MetalFp32Compiler, PostGenericCompiler)>::default());

        rope_graph.execute();

        // Build the actual rope embeddings
        let (cos, sin) = (
            graph.named_tensor("rope_embeddings_cos"),
            graph.named_tensor("rope_embeddings_sin"),
        );
        cos.set(rope_cos.data());
        sin.set(rope_sin.data());

        let rope_embeddings = (cos, sin);

        Ok(Self {
            tokenizer,
            graph,
            embedding,
            transformer_layers,
            norm,
            lm_head,
            input,
            rope_embeddings,
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

        self.tokenizer
            .decode(token_ids, true, false)
            .replace("<0x0A>", "\n")
    }

    pub fn load(&mut self, file_paths: Vec<String>) -> Result<(), String> {
        // First, let's get a mapping from node to weight name
        let mut weight_name_tensor_mapping = HashMap::new();

        // Embeddings
        weight_name_tensor_mapping
            .insert("model.embed_tokens.weight".to_string(), self.embedding.id);

        // Transformer Layers
        for layer_index in 0..self.transformer_layers.len() {
            let weight_prefix = format!("model.layers.{layer_index}");

            // Query Projection
            let weight_name = format!("{weight_prefix}.self_attn.q_proj.weight");
            weight_name_tensor_mapping.insert(
                weight_name.clone(),
                self.transformer_layers[layer_index].attention.q_proj.id,
            );

            // Key Projection
            let weight_name = format!("{weight_prefix}.self_attn.k_proj.weight");
            weight_name_tensor_mapping.insert(
                weight_name.clone(),
                self.transformer_layers[layer_index].attention.k_proj.id,
            );

            // Value Projection
            let weight_name = format!("{weight_prefix}.self_attn.v_proj.weight");
            weight_name_tensor_mapping.insert(
                weight_name.clone(),
                self.transformer_layers[layer_index].attention.v_proj.id,
            );

            // Output Projection
            let weight_name = format!("{weight_prefix}.self_attn.o_proj.weight");
            weight_name_tensor_mapping.insert(
                weight_name.clone(),
                self.transformer_layers[layer_index].attention.o_proj.id,
            );

            // Pre-Attention Norm
            let weight_name = format!("{weight_prefix}.input_layernorm.weight");
            weight_name_tensor_mapping.insert(
                weight_name.clone(),
                self.transformer_layers[layer_index]
                    .attention_norm
                    .weight
                    .id,
            );

            // Feed Forward Norm
            let weight_name = format!("{weight_prefix}.post_attention_layernorm.weight");
            weight_name_tensor_mapping.insert(
                weight_name.clone(),
                self.transformer_layers[layer_index]
                    .feed_forward_norm
                    .weight
                    .id,
            );

            // Feed Forward Gate Projection
            let weight_name = format!("{weight_prefix}.mlp.gate_proj.weight");
            weight_name_tensor_mapping.insert(
                weight_name.clone(),
                self.transformer_layers[layer_index]
                    .feed_forward
                    .gate_proj
                    .id,
            );

            // Feed Forward Down Projection
            let weight_name = format!("{weight_prefix}.mlp.down_proj.weight");
            weight_name_tensor_mapping.insert(
                weight_name.clone(),
                self.transformer_layers[layer_index]
                    .feed_forward
                    .down_proj
                    .id,
            );

            // Feed Forward Up Projection
            let weight_name = format!("{weight_prefix}.mlp.up_proj.weight");
            weight_name_tensor_mapping.insert(
                weight_name.clone(),
                self.transformer_layers[layer_index].feed_forward.up_proj.id,
            );
        }

        // Final Norm
        weight_name_tensor_mapping.insert("model.norm.weight".to_string(), self.norm.weight.id);

        // LM Head
        weight_name_tensor_mapping.insert("lm_head.weight".to_string(), self.lm_head.id);

        // And then we loop over all of them an load
        for (weight_name, node_index) in weight_name_tensor_mapping.drain() {
            if let Some(loading_node) = self
                .graph
                .graph
                .node_weight_mut(node_index)
                .and_then(|op| op.as_any_mut().downcast_mut::<Function>())
            {
                let file_paths = file_paths.clone();
                loading_node.1 = Box::new(move |_| {
                    let mut output_vector = vec![];
                    // Do the loading here
                    for file_path in file_paths.iter() {
                        let file = File::open(file_path).unwrap();
                        let buffer = unsafe { MmapOptions::new().map(&file).unwrap() };
                        let safetensors = SafeTensors::deserialize(&buffer).unwrap();

                        if let Ok(tensor_view) = safetensors.tensor(weight_name.as_str()) {
                            output_vector = vec![Tensor {
                                data: Box::new(convert_vector_bf16_f32(&tensor_view)),
                            }];
                            break;
                        }
                    }

                    // Return
                    output_vector
                });
            }
        }
        Ok(())
    }
}

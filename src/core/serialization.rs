use crate::op::Function;
use crate::prelude::{Graph, GraphTensor, Shape, Tensor};
use half::{bf16, f16};
use memmap2::MmapOptions;
use petgraph::stable_graph::NodeIndex;
use safetensors::tensor::{Dtype, View};
use safetensors::{SafeTensorError, SafeTensors};
use std::borrow::Cow;
use std::collections::HashMap;
use std::fs::File;

use super::module::state_dict;

/// Tell luminal how to represent the module as a dict of (String, NodeIndex)'s
pub trait SerializeModule {
    fn serialize(&self, s: &mut Serializer);
}

/// Something that can load the state of a module into the graph
pub trait Loader {
    fn load<M: SerializeModule>(self, model: &M, graph: &mut Graph);
}

/// Something that can save the state of a module from the graph
pub trait Saver {
    type Saved;
    fn save<M: SerializeModule>(self, model: &M, graph: &mut Graph) -> Self::Saved;
}

/// Extract the state dict from a model
pub struct StateDictSaver;

impl Saver for StateDictSaver {
    type Saved = HashMap<String, Tensor>;
    fn save<M: SerializeModule>(self, model: &M, graph: &mut Graph) -> Self::Saved {
        let mut serializer = Serializer::default();
        model.serialize(&mut serializer);
        // Attempt to get all tensor data from the graph
        serializer
            .state
            .into_iter()
            .map(|(k, v)| (k, graph.get_tensor(v, 0).unwrap()))
            .collect()
    }
}

/// Save a model to a safetensor file
pub struct SafeTensorSaver {
    path: String,
}

impl SafeTensorSaver {
    pub fn new(path: &str) -> Self {
        Self {
            path: path.to_string(),
        }
    }
}

impl Saver for SafeTensorSaver {
    type Saved = Result<(), SafeTensorError>;
    fn save<M: SerializeModule>(self, model: &M, graph: &mut Graph) -> Self::Saved {
        let mut serializer = Serializer::default();
        model.serialize(&mut serializer);
        // Attempt to get all tensor data from the graph
        let state_dict: HashMap<_, _> = serializer
            .state
            .into_iter()
            .map(|(k, v)| (k, graph.get_tensor_ref(v, 0).unwrap()))
            .collect();
        safetensors::serialize_to_file(state_dict, &None, self.path.as_ref())
    }
}

/// Load the model from a state dict
pub struct StateDictLoader {
    state_dict: HashMap<String, Tensor>,
}

impl StateDictLoader {
    pub fn new(state_dict: HashMap<String, Tensor>) -> Self {
        Self { state_dict }
    }
}

impl Loader for StateDictLoader {
    fn load<M: SerializeModule>(mut self, model: &M, graph: &mut Graph) {
        let mut serializer = Serializer::default();
        model.serialize(&mut serializer);

        for (s, n) in serializer.state {
            let t = self.state_dict.remove(&s).unwrap();
            graph.no_delete.insert(n);
            graph.tensors.insert((n, 0), t);
        }
    }
}

/// Load the model from a safetensor file
pub struct SafeTensorLoader {
    /// The paths to the safetensors file
    paths: Vec<String>,
}

impl SafeTensorLoader {
    pub fn new<S: ToString>(paths: &[S]) -> Self {
        Self {
            paths: paths.into_iter().map(|s| s.to_string()).collect(),
        }
    }
}

impl Loader for SafeTensorLoader {
    fn load<M: SerializeModule>(self, model: &M, graph: &mut Graph) {
        for (weight_name, node_index) in state_dict(model) {
            if let Some(loading_node) = graph
                .graph
                .node_weight_mut(node_index)
                .and_then(|op| op.as_any_mut().downcast_mut::<Function>())
            {
                let file_paths = self.paths.clone();
                loading_node.1 = Box::new(move |_| {
                    for file_path in file_paths.iter() {
                        let file = File::open(file_path).unwrap();
                        let buffer = unsafe { MmapOptions::new().map(&file).unwrap() };
                        let safetensors = SafeTensors::deserialize(&buffer).unwrap();

                        if let Ok(tensor_view) = safetensors.tensor(&weight_name.replace('/', "."))
                        {
                            // Convert to fp32
                            let data: Vec<f32> = match tensor_view.dtype() {
                                Dtype::F32 => tensor_view
                                    .data()
                                    .chunks_exact(4)
                                    .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
                                    .collect(),
                                Dtype::F16 => tensor_view
                                    .data()
                                    .chunks_exact(2)
                                    .map(|c| f16::from_ne_bytes([c[0], c[1]]).to_f32())
                                    .collect(),
                                Dtype::BF16 => tensor_view
                                    .data()
                                    .chunks_exact(2)
                                    .map(|c| bf16::from_ne_bytes([c[0], c[1]]).to_f32())
                                    .collect(),
                                _ => panic!("{:?} is not a supported dtype", tensor_view.dtype()),
                            };
                            return vec![Tensor {
                                data: Box::new(data),
                            }];
                        }
                    }

                    panic!("Tensor \"{weight_name}\" not found in files");
                });
            }
        }
    }
}

#[derive(Debug, Default)]
pub struct Serializer {
    current_path: Vec<String>,
    pub state: HashMap<String, NodeIndex>,
}

impl Serializer {
    pub fn tensor<S: Shape>(&mut self, name: &str, tensor: GraphTensor<S>) {
        // Add new path component
        self.current_path.push(name.to_string());
        // Insert tensor id
        self.state.insert(self.current_path.join("/"), tensor.id);
        // Remove new path component
        self.current_path.pop();
    }
    pub fn module<T: SerializeModule>(&mut self, name: &str, module: &T) {
        // Add new path component
        self.current_path.push(name.to_string());
        // Serialize
        module.serialize(self);
        // Remove new path component
        self.current_path.pop();
    }
}

impl<'data> View for &'data Tensor {
    fn dtype(&self) -> Dtype {
        Dtype::F32 // For now just assume float, this should change in the future
    }
    fn shape(&self) -> &[usize] {
        &[]
    }
    fn data(&self) -> Cow<[u8]> {
        self.data
            .as_any()
            .downcast_ref::<Vec<f32>>()
            .unwrap()
            .iter()
            .flat_map(|f| f.to_le_bytes().into_iter())
            .collect::<Vec<_>>()
            .into()
    }
    fn data_len(&self) -> usize {
        self.data.as_any().downcast_ref::<Vec<f32>>().unwrap().len()
    }
}

impl<'a> std::convert::From<safetensors::tensor::TensorView<'a>> for Tensor {
    fn from(value: safetensors::tensor::TensorView<'a>) -> Self {
        let chunked = value.data().chunks_exact(std::mem::size_of::<f32>());

        Tensor {
            data: Box::new(
                chunked
                    .map(|chunk| unsafe {
                        std::mem::transmute::<[u8; 4], f32>([
                            chunk[0], chunk[1], chunk[2], chunk[3],
                        ])
                    })
                    .collect::<Vec<f32>>(),
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::{thread_rng, Rng};

    use crate::{nn::transformer::Transformer, prelude::*, tests::assert_close};

    use super::*;

    #[test]
    fn test_serialization() {
        let mut cx = Graph::new();
        let model: Transformer<32, 5, 4, 4, 3, 2> = InitModule::initialize(&mut cx);
        let enc = cx.tensor::<R2<24, 32>>();
        let trg = cx.tensor::<R2<20, 32>>();
        let out1 = model.forward((trg, enc));

        let mut rng = thread_rng();
        let enc_data = (0..(24 * 32)).map(|_| rng.gen()).collect::<Vec<f32>>();
        let trg_data = (0..(20 * 32)).map(|_| rng.gen()).collect::<Vec<f32>>();
        enc.set(enc_data.clone());
        trg.set(trg_data.clone());
        out1.retrieve();

        cx.execute_no_delete();

        let state_dict = StateDictSaver.save(&model, &mut cx);
        let out1 = out1.data();

        let mut cx = Graph::new();
        let model: Transformer<32, 5, 4, 4, 3, 2> = InitModule::initialize(&mut cx);
        StateDictLoader::new(state_dict).load(&model, &mut cx);
        let enc = cx.tensor::<R2<24, 32>>();
        let trg = cx.tensor::<R2<20, 32>>();
        let out2 = model.forward((trg, enc));

        enc.set(enc_data);
        trg.set(trg_data);
        out2.retrieve();

        cx.compile(<(CPUCompiler, PostGenericCompiler)>::default());
        cx.execute();

        let out2 = out2.data();
        assert_close(&out1, &out2);
    }
}

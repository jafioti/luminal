use crate::op::Function;
use crate::prelude::{Graph, GraphTensor, Shape, Tensor};
use memmap2::MmapOptions;
use petgraph::stable_graph::NodeIndex;
use safetensors::tensor::{Dtype, View};
use safetensors::SafeTensorError;
use std::borrow::Cow;
use std::collections::HashMap;

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

/// Load the entire model from a safetensor file all at once
pub struct SafeTensorLoader {
    /// The path to the safetensor file
    path: String,
}

impl SafeTensorLoader {
    pub fn new(path: &str) -> Self {
        Self {
            path: path.to_string(),
        }
    }
}

impl Loader for SafeTensorLoader {
    fn load<M: SerializeModule>(self, model: &M, graph: &mut Graph) {
        let data = std::fs::read(self.path).unwrap();
        let st = safetensors::SafeTensors::deserialize(&data).unwrap();
        let mut state_dict: HashMap<String, Tensor> = st
            .tensors()
            .into_iter()
            .map(|(k, v)| (k, v.into()))
            .collect();
        let mut serializer = Serializer::default();
        model.serialize(&mut serializer);

        for (s, n) in serializer.state {
            graph.tensors.insert((n, 0), state_dict.remove(&s).unwrap());
        }
    }
}

/// Load the entire model from a safetensor file, loading each tensor as needed
pub struct SafeTensorDeferredLoader {
    /// The path to the safetensor file
    path: String,
}

impl SafeTensorDeferredLoader {
    pub fn new(path: &str) -> Self {
        Self {
            path: path.to_string(),
        }
    }
}

impl Loader for SafeTensorDeferredLoader {
    fn load<M: SerializeModule>(self, model: &M, graph: &mut Graph) {
        let mut serializer = Serializer::default();
        model.serialize(&mut serializer);

        for (s, n) in serializer.state {
            if let Some(inp_func) = graph
                .graph
                .node_weight_mut(n)
                .unwrap()
                .as_any_mut()
                .downcast_mut::<Function>()
            {
                let path = self.path.clone();
                inp_func.1 = Box::new(move |_| {
                    // Get memmapped tensor
                    let file = std::fs::File::open(path.clone()).unwrap();
                    let buffer = unsafe { MmapOptions::new().map(&file).unwrap() };
                    let st = safetensors::SafeTensors::deserialize(&buffer).unwrap();
                    let tensor = st.tensor(&s).unwrap().into();

                    vec![tensor]
                });
            };
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
        let enc = cx.new_tensor::<R2<24, 32>>("EncInp");
        let trg = cx.new_tensor::<R2<20, 32>>("TrgInp");
        let out1 = model.forward((trg, enc));

        let mut rng = thread_rng();
        let enc_data = (0..(24 * 32)).map(|_| rng.gen()).collect::<Vec<f32>>();
        let trg_data = (0..(20 * 32)).map(|_| rng.gen()).collect::<Vec<f32>>();
        enc.set(enc_data.clone());
        trg.set(trg_data.clone());
        out1.mark();

        cx.execute_no_delete();

        let state_dict = StateDictSaver.save(&model, &mut cx);
        let out1 = out1.data();

        let mut cx = Graph::new();
        let model: Transformer<32, 5, 4, 4, 3, 2> = InitModule::initialize(&mut cx);
        StateDictLoader::new(state_dict).load(&model, &mut cx);
        let enc = cx.new_tensor::<R2<24, 32>>("EncInp");
        let trg = cx.new_tensor::<R2<20, 32>>("TrgInp");
        let out2 = model.forward((trg, enc));

        enc.set(enc_data);
        trg.set(trg_data);
        out2.mark();

        cx.compile(<(CPUCompiler, GenericCompiler)>::default());
        cx.execute();

        let out2 = out2.data();
        assert_close(&out1, &out2);
    }
}

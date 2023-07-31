use crate::prelude::{Graph, GraphTensor, Shape, ShapeTracker, Tensor, TensorView};
use itertools::Itertools;
use petgraph::stable_graph::NodeIndex;
use safetensors::tensor::{Dtype, View};
use std::collections::HashMap;
use std::{borrow::Cow, path::Path};

/// Tell luminal how to represent the module as a dict of (String, Tensor)'s
pub trait SerializeModule {
    fn serialize(&self, s: &mut Serializer);
}

/// A trait automatically derived on modules implementing SerializeModule, allowing access to `.save_to_file()`
pub trait SaveLoadModule: SerializeModule {
    /// Save the module to a file in the SafeTensors format
    fn save_to_file<P: AsRef<Path>>(&self, cx: &Graph, filename: P) {
        let tensors = self.state_dict(cx);
        safetensors::serialize_to_file(tensors, &None, filename.as_ref()).unwrap();
    }
    /// Get the state dict of the module
    fn state_dict<'a>(&'a self, cx: &'a Graph) -> HashMap<String, SaveTensorRef<'a>> {
        let mut serializer = Serializer {
            current_path: ".".to_string(),
            state: HashMap::default(),
        };
        self.serialize(&mut serializer);
        // Attempt to get all tensor data from the graph
        serializer
            .state
            .into_iter()
            .map(|(k, v)| {
                (
                    k,
                    SaveTensorRef(cx.get_tensor_ref(v).unwrap(), cx.views.get(&v).unwrap()),
                )
            })
            .collect()
    }
    /// Load module from state dict
    fn load_from_state_dict(
        &mut self,
        cx: &mut Graph,
        mut state_dict: HashMap<String, (Tensor, TensorView)>,
    ) {
        let mut serializer = Serializer {
            current_path: ".".to_string(),
            state: HashMap::default(),
        };
        self.serialize(&mut serializer);

        for (s, n) in serializer.state {
            let (t, v) = state_dict.remove(&s).unwrap();
            cx.tensors.insert(n, t);
            cx.views.insert(n, v);
        }
    }
    /// Load a module from a SafeTensors file
    fn load_from_file<P: AsRef<Path>>(&mut self, cx: &mut Graph, filename: P) {
        let data = std::fs::read(filename).unwrap();
        let st = safetensors::SafeTensors::deserialize(&data).unwrap();
        let state_dict = st
            .tensors()
            .into_iter()
            .map(|(k, v)| {
                let t: SaveTensor = v.into();
                (k, (t.0, t.1))
            })
            .collect();
        self.load_from_state_dict(cx, state_dict);
    }
}

impl<T: SerializeModule> SaveLoadModule for T {}

pub struct Serializer {
    current_path: String,
    state: HashMap<String, NodeIndex>,
}

impl Serializer {
    pub fn tensor<S: Shape>(&mut self, name: &str, tensor: GraphTensor<S>) {
        self.state
            .insert(format!("{}/{}", self.current_path, name), tensor.id);
    }
    pub fn module<T: SerializeModule>(&mut self, name: &str, module: &T) {
        // Add new path component
        self.current_path.push('/');
        self.current_path.push_str(name);
        // Serialize
        module.serialize(self);
        // Remove new path component
        let mut components = self.current_path.split('/').collect_vec();
        components.pop();
        self.current_path = components.join("/");
    }
}

#[derive(Clone, Debug)]
pub struct SaveTensorRef<'a>(&'a Tensor, &'a TensorView);
#[derive(Clone, Debug)]
pub struct SaveTensor(Tensor, TensorView);

impl<'data> View for SaveTensorRef<'data> {
    fn dtype(&self) -> Dtype {
        Dtype::F32 // For now just assume float, this should change in the future
    }
    fn shape(&self) -> &[usize] {
        self.1.shape.shape()
    }
    fn data(&self) -> Cow<[u8]> {
        self.0
            .data
            .as_any()
            .downcast_ref::<Vec<f32>>()
            .unwrap()
            .iter()
            .flat_map(|f| f.to_le_bytes().into_iter())
            .collect::<Vec<_>>()
            .into()
    }
    fn data_len(&self) -> usize {
        self.0
            .data
            .as_any()
            .downcast_ref::<Vec<f32>>()
            .unwrap()
            .len()
    }
}

impl<'a> std::convert::From<safetensors::tensor::TensorView<'a>> for SaveTensor {
    fn from(value: safetensors::tensor::TensorView<'a>) -> Self {
        let chunked = value.data().chunks_exact(std::mem::size_of::<f32>());

        SaveTensor(
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
            },
            TensorView {
                tensor_id: NodeIndex::default(),
                shape: ShapeTracker::new(value.shape().to_vec()),
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use rand::{thread_rng, Rng};
    use std::collections::HashMap;

    use crate::{nn::transformer::Transformer, prelude::*, tests::assert_close_data};

    use super::{SaveLoadModule, SaveTensorRef};

    #[test]
    fn test_serialization() {
        let mut cx = Graph::new();
        let model: Transformer<32, 200, 5, 5, 3, 2> = InitModule::initialize(&mut cx);
        let enc = cx.new_tensor::<R2<24, 32>>();
        let trg = cx.new_tensor::<R2<20, 32>>();
        let out1 = model.forward((trg, enc));

        let mut rng = thread_rng();
        let enc_data = (0..(24 * 32)).map(|_| rng.gen()).collect::<Vec<f32>>();
        let trg_data = (0..(20 * 32)).map(|_| rng.gen()).collect::<Vec<f32>>();
        enc.set(enc_data.clone());
        trg.set(trg_data.clone());
        out1.mark();

        cx.execute();
        let out1 = out1
            .retrieve()
            .unwrap()
            .real_data(out1.view().unwrap())
            .unwrap();

        let state_dict = model.state_dict(&cx);
        let state_dict: HashMap<_, _> = state_dict
            .into_iter()
            .map(|(k, SaveTensorRef(a, b))| (k, (a.clone(), b.clone())))
            .collect();

        let mut cx = Graph::new();
        let mut model: Transformer<32, 200, 5, 5, 3, 2> = InitModule::initialize(&mut cx);
        model.load_from_state_dict(&mut cx, state_dict);
        let enc = cx.new_tensor::<R2<24, 32>>();
        let trg = cx.new_tensor::<R2<20, 32>>();
        let out2 = model.forward((trg, enc));

        enc.set(enc_data);
        trg.set(trg_data);
        out2.mark();

        cx.optimize(<(CPUOptimizer, GenericOptimizer)>::default());
        cx.execute();

        let out2 = out2
            .retrieve()
            .unwrap()
            .real_data(out2.view().unwrap())
            .unwrap();
        assert_close_data(&out1, &out2);
    }
}

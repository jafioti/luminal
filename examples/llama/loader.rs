use std::collections::HashMap;

use luminal::{op::Function, prelude::*};
use memmap2::MmapOptions;

/// Load the model in the same way dfdx-llama does
pub struct DfdxDeferredLoader {
    /// The path to the model folder
    path: String,
}

impl DfdxDeferredLoader {
    pub fn new(path: &str) -> Self {
        Self {
            path: path.to_string(),
        }
    }
}

// impl Loader for DfdxDeferredLoader {
//     fn load<M: SerializeModule>(self, model: &M, graph: &mut Graph) {
//         let mut serializer = Serializer {
//             current_path: ".".to_string(),
//             state: HashMap::default(),
//         };
//         model.serialize(&mut serializer);

//         for (s, n) in serializer.state {
//             if let Some(inp_func) = graph
//                 .graph
//                 .node_weight_mut(n)
//                 .unwrap()
//                 .0
//                 .as_any_mut()
//                 .downcast_mut::<Function>()
//             {
//                 let path = self.path.clone();
//                 inp_func.1 = Box::new(move |_, i| {
//                     // Get memmapped tensor
//                     let file = std::fs::File::open(path.clone()).unwrap();
//                     let buffer = unsafe { MmapOptions::new().map(&file).unwrap() };
//                     let st = safetensors::SafeTensors::deserialize(&buffer).unwrap();
//                     let SaveTensor(tensor, mut view) = st.tensor(&s).unwrap().into();

//                     view.tensor_id = i;
//                     (Some(tensor), view)
//                 });
//             };
//         }
//     }
// }

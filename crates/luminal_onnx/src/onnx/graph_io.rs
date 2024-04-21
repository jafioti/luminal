


use std::collections::HashMap;
use luminal::prelude::*;

use crate::onnx::proto::TensorProto;

/// A node input or output.
#[derive(Debug, Clone)]
pub struct Argument {
    /// The name of the node input.
    pub name: String,

    /// The type of the argument.
    pub ty: ArgType,

    /// The data of the argument.
    pub value: Option<dyn Data>,

    /// True if the argument is passed to node, false otherwise. We use it mainly for informational purposes.
    /// The argument should contain a value if passed is false.
    pub passed: bool,
}

/// The type of an argument.
#[derive(Debug, Clone)]
pub enum ArgType {
    Scalar(ElementType),
    Shape(Dim),
    Tensor(TensorType),
}
impl Argument {
    /// Copy everything except the name from the other argument
    pub fn copy_value(&mut self, other_arg: &Argument) {
        self.ty = other_arg.ty.clone();
        self.value = other_arg.value.clone();
    }

    pub fn from_initializer(initializer: &TensorProto) -> Argument {
        let name = initializer.name.clone();
        let tensor = Tensor::try_from(initializer.clone())
            .unwrap_or_else(|_| panic!("invalid tensor {}", &initializer.name));

        if tensor.dim == 0 {
            // Convert zero dim tensor to scalar
            let value = if tensor.data.is_some() {
                Some(tensor.data.clone().unwrap().into_scalar())
            } else {
                None
            };
            let ty = ArgType::Scalar(tensor.elem_type);

            Self {
                name,
                ty,
                value,
                passed: false,
            }
        } else {
            Self {
                name,
                ty: ArgType::Tensor(TensorType {
                    elem_type: tensor.elem_type,
                    dim: tensor.dim,
                    shape: tensor.shape,
                }),
                value: tensor.data.clone(),
                passed: false,
            }
        }
    }
}

#[derive(Debug)]
pub(crate) enum IOEntry {
    In(usize),
    Out(usize),
    Node(usize),
}

pub(crate) struct OnnxGraphIO {
    /// The inputs for the Graph
    pub(crate) inputs: Vec<Argument>,
    /// The outputs for the Graph
    pub(crate) outputs: Vec<Argument>,
    /// Initializers
    pub(crate) initializers: HashMap<String, Argument>,
    ///updated names of outputs of node not stored in the graph
    node_out: Vec<Argument>,
    pub(crate) old_io_names: HashMap<String, IOEntry>,
}

impl OnnxGraphIO {
    pub(crate) fn new(
        inputs: &Vec<ValueInfoProto>,
        outputs: &Vec<ValueInfoProto>,
        initializers: &Vec<TensorProto>,
    ) -> Self {
        let mut old_io_names = HashMap::new();
        let mut in_count = 1;
        let constants = initializers
            .iter()
            .map(|x| (x.name.clone(), Argument::from_initializer(x)))
            .collect::<HashMap<String, Argument>>();

        let inputs = inputs
            .iter()
            .enumerate()
            .map(|(i, x)| {
                let in_name = format!("input{}", in_count);
                old_io_names.insert(x.name.clone(), IOEntry::In(i));
                let mut arg = Argument::try_from(x.clone()).unwrap();
                if let Some(initial_arg) = constants.get(&x.name) {
                    if arg.value.is_none() {
                        arg.copy_value(initial_arg);
                    }
                }

                in_count += 1;
                arg.name = in_name;
                arg
            })
            .collect::<Vec<Argument>>();

        let outputs = outputs
            .iter()
            .enumerate()
            .map(|(i, x)| {
                old_io_names.insert(x.name.clone(), IOEntry::Out(i));
                Argument::try_from(x.clone()).unwrap()
            })
            .collect::<Vec<Argument>>();

        let constants = initializers
            .iter()
            .map(|x| (x.name.clone(), Argument::from_initializer(x)))
            .collect::<HashMap<String, Argument>>();

        Self {
            inputs,
            outputs,
            initializers: constants,
            node_out: Vec::new(),
            old_io_names,
        }
    }

    fn update_name(&mut self, arg: &Argument, new_name: &str) {
        match self.old_io_names.get(&arg.name) {
            Some(IOEntry::In(_)) => {
                panic!("input names are set from the beginning");
            }
            Some(IOEntry::Out(i)) => {
                let arg = self.outputs.get_mut(*i).unwrap();
                arg.name = new_name.to_string();
            }
            Some(IOEntry::Node(i)) => {
                let arg = self.node_out.get_mut(*i).unwrap();
                arg.name = new_name.to_string();
            }
            None => {
                //Constants, Casts wound up here before API changes
                panic!(
                    "Tried to update the name of {} to {} but entry doesn't exist in the map",
                    arg.name, new_name
                )
            }
        }
    }

    /// Used to initialize the input arguments for nodes. Names need to remain the same because
    /// currently the old names are the key for accessing the Argument
    pub fn init_in(&self, proto_str: String) -> Argument {
        match self.old_io_names.get(&proto_str) {
            None => {
                if let Some(init_arg) = self.initializers.get(&proto_str) {
                    init_arg.clone()
                } else {
                    Argument::new(proto_str)
                }
            }

            Some(IOEntry::In(i)) => {
                let mut arg = self.inputs[*i].clone();

                arg.name = proto_str;
                arg.passed = true;
                arg
            }
            Some(IOEntry::Node(i)) => {
                let mut arg = self.node_out[*i].clone();
                arg.name = proto_str;
                arg
            }
            Some(IOEntry::Out(_)) => {
                panic!("graph output {} can't be a Node input", &proto_str)
            }
        }
    }

    fn insert(&mut self, arg: &Argument, new_name: &str) {
        if let Some(idx) = self.old_io_names.get(&arg.name) {
            if let IOEntry::Node(idx) = idx {
                if self.node_out[*idx].name == arg.name {
                    self.node_out[*idx].name = new_name.to_string();
                    return;
                }
            } else {
                panic!("arg entry with old name {} is a graph IO", &arg.name);
            }
        }

        let idx = self.node_out.len();
        self.old_io_names
            .insert(arg.name.clone(), IOEntry::Node(idx));
        self.node_out.push(arg.clone());
        self.node_out[idx].name = new_name.to_string();
    }

    /// Copies node outputs to graph IO. Used at the end of dim inference.
    pub(crate) fn update_tensor_output(&mut self, node: &Node) {
        for node_output in node.outputs.iter() {
            match self.old_io_names.get(&node_output.name) {
                Some(IOEntry::In(i)) => {
                    let arg = self.inputs.get_mut(*i).unwrap();
                    arg.copy_value(node_output);
                }
                Some(IOEntry::Out(i)) => {
                    let arg = self.outputs.get_mut(*i).unwrap();
                    arg.copy_value(node_output);
                    //Set the output to passed since it's been altered by a Node
                    arg.passed = true;
                }
                Some(IOEntry::Node(_)) => {
                    panic!("This output is from another node");
                }
                None => {
                    log::debug!("inserting with name {:?}", &node_output.name);
                    let idx = self.node_out.len();
                    self.old_io_names
                        .insert(node_output.name.clone(), IOEntry::Node(idx));
                    self.node_out.push(node_output.clone());
                }
            }
        }
    }

    /// Used by handle unsqeeze to remap the output of a node to a new name
    /// expected match if it exists is either a graph input or graph output
    pub(crate) fn get_node_output(&self, old_name: &str) -> Option<&Argument> {
        match self.old_io_names.get(old_name) {
            Some(IOEntry::In(i)) => self.inputs.get(*i),
            Some(IOEntry::Out(i)) => self.outputs.get(*i),
            Some(IOEntry::Node(_)) => panic!("This is a node output"),
            None => None,
        }
    }

    /// Get the updated name of a Node Input, which should be
    /// either a graph input or a node output.
    /// Will return None if the it isn't a graph input or node output(like an initializer)
    /// Will panic if it's a graph output
    fn get_new_name(&mut self, old_name: &str) -> Option<String> {
        match self.old_io_names.get(old_name) {
            Some(IOEntry::In(i)) => {
                //FIXME: technically in the spec, initializers are default values
                //for optional inputs, but implementing that would require reworking
                //the way the graph is built, and it's not clear burn users are using initializers
                //in that way
                // see https://github.com/onnx/onnx/issues/2660
                if self.initializers.contains_key(old_name) {
                    None
                } else {
                    //set the input as passed since a node is referencing it
                    self.inputs[*i].passed = true;
                    Some(self.inputs[*i].name.clone())
                }
            }
            Some(IOEntry::Out(_)) => {
                panic!(
                    "you just tried to get an updated name on a graph output: {}",
                    old_name
                )
            }
            Some(IOEntry::Node(i)) => Some(self.node_out[*i].name.clone()),
            None => None,
        }
    }
}
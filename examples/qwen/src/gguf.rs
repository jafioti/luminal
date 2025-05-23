//! Support for the GGUF file format.
//!
//! Spec: https://github.com/philpax/ggml/blob/gguf-spec/docs/gguf.md
#![allow(unused)]

use byteorder::{LittleEndian, ReadBytesExt};
use std::collections::HashMap;

pub const DEFAULT_ALIGNMENT: u64 = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Magic {
    Gguf,
}

impl TryFrom<u32> for Magic {
    type Error = ();
    fn try_from(value: u32) -> Result<Self, ()> {
        let magic = match value {
            0x46554747 | 0x47475546 => Self::Gguf,
            _ => panic!("unknown magic 0x{value:08x}"),
        };
        Ok(magic)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VersionedMagic {
    GgufV1,
    GgufV2,
    GgufV3,
}

impl VersionedMagic {
    pub fn read<R: std::io::Read>(reader: &mut R) -> Result<Self, ()> {
        let magic = reader.read_u32::<LittleEndian>().unwrap();
        let magic = Magic::try_from(magic).unwrap();
        let version = reader.read_u32::<LittleEndian>().unwrap();
        let versioned_magic = match (magic, version) {
            (Magic::Gguf, 1) => Self::GgufV1,
            (Magic::Gguf, 2) => Self::GgufV2,
            (Magic::Gguf, 3) => Self::GgufV3,
            _ => panic!("gguf: unsupported magic/version {magic:?}/{version}"),
        };
        Ok(versioned_magic)
    }
}

#[derive(Debug)]
pub struct Content {
    pub magic: VersionedMagic,
    pub metadata: HashMap<String, Value>,
    pub tensor_infos: HashMap<String, (usize, usize, GgmlDType)>, // buffer size and offset
    pub tensor_data_offset: u64,
}

pub fn read_string<R: std::io::Read>(reader: &mut R, magic: &VersionedMagic) -> Result<String, ()> {
    let len = match magic {
        VersionedMagic::GgufV1 => reader.read_u32::<LittleEndian>().unwrap() as usize,
        VersionedMagic::GgufV2 | VersionedMagic::GgufV3 => {
            reader.read_u64::<LittleEndian>().unwrap() as usize
        }
    };
    let mut v = vec![0u8; len];
    reader.read_exact(&mut v).unwrap();
    // GGUF strings are supposed to be non-null terminated but in practice this happens.
    while let Some(0) = v.last() {
        v.pop();
    }
    // GGUF strings are utf8 encoded but there are cases that don't seem to be valid.
    Ok(String::from_utf8_lossy(&v).into_owned())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ValueType {
    // The value is a 8-bit unsigned integer.
    U8,
    // The value is a 8-bit signed integer.
    I8,
    // The value is a 16-bit unsigned little-endian integer.
    U16,
    // The value is a 16-bit signed little-endian integer.
    I16,
    // The value is a 32-bit unsigned little-endian integer.
    U32,
    // The value is a 32-bit signed little-endian integer.
    I32,
    // The value is a 64-bit unsigned little-endian integer.
    U64,
    // The value is a 64-bit signed little-endian integer.
    I64,
    // The value is a 32-bit IEEE754 floating point number.
    F32,
    // The value is a 64-bit IEEE754 floating point number.
    F64,
    // The value is a boolean.
    // 1-byte value where 0 is false and 1 is true.
    // Anything else is invalid, and should be treated as either the model being invalid or the reader being buggy.
    Bool,
    // The value is a UTF-8 non-null-terminated string, with length prepended.
    String,
    // The value is an array of other values, with the length and type prepended.
    //
    // Arrays can be nested, and the length of the array is the number of elements in the array, not the number of bytes.
    Array,
    // The value is a 16-bit brain floating point number.
    BF16,
}

#[derive(Debug, Clone)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<Value>),
    BF16(u16), // BF16 stored as raw u16 bits
}

impl Value {
    pub fn read<R: std::io::Read>(
        reader: &mut R,
        value_type: ValueType,
        magic: &VersionedMagic,
    ) -> Result<Self, ()> {
        let v = match value_type {
            ValueType::U8 => Self::U8(reader.read_u8().unwrap()),
            ValueType::I8 => Self::I8(reader.read_i8().unwrap()),
            ValueType::U16 => Self::U16(reader.read_u16::<LittleEndian>().unwrap()),
            ValueType::I16 => Self::I16(reader.read_i16::<LittleEndian>().unwrap()),
            ValueType::U32 => Self::U32(reader.read_u32::<LittleEndian>().unwrap()),
            ValueType::I32 => Self::I32(reader.read_i32::<LittleEndian>().unwrap()),
            ValueType::U64 => Self::U64(reader.read_u64::<LittleEndian>().unwrap()),
            ValueType::I64 => Self::I64(reader.read_i64::<LittleEndian>().unwrap()),
            ValueType::F32 => Self::F32(reader.read_f32::<LittleEndian>().unwrap()),
            ValueType::F64 => Self::F64(reader.read_f64::<LittleEndian>().unwrap()),
            ValueType::BF16 => Self::BF16(reader.read_u16::<LittleEndian>().unwrap()),
            ValueType::Bool => match reader.read_u8().unwrap() {
                0 => Self::Bool(false),
                1 => Self::Bool(true),
                b => panic!("unexpected bool value {b}"),
            },
            ValueType::String => Self::String(read_string(reader, magic).unwrap()),
            ValueType::Array => {
                let value_type = reader.read_u32::<LittleEndian>().unwrap();
                let value_type = ValueType::from_u32(value_type).unwrap();
                let len = match magic {
                    VersionedMagic::GgufV1 => reader.read_u32::<LittleEndian>().unwrap() as usize,
                    VersionedMagic::GgufV2 | VersionedMagic::GgufV3 => {
                        reader.read_u64::<LittleEndian>().unwrap() as usize
                    }
                };
                let mut vs = Vec::with_capacity(len);
                for _ in 0..len {
                    vs.push(Value::read(reader, value_type, magic).unwrap())
                }
                Self::Array(vs)
            }
        };
        Ok(v)
    }
}

impl ValueType {
    pub fn from_u32(v: u32) -> Result<Self, ()> {
        let v = match v {
            0 => Self::U8,
            1 => Self::I8,
            2 => Self::U16,
            3 => Self::I16,
            4 => Self::U32,
            5 => Self::I32,
            6 => Self::F32,
            7 => Self::Bool,
            8 => Self::String,
            9 => Self::Array,
            10 => Self::U64,
            11 => Self::I64,
            12 => Self::F64,
            13 => Self::BF16,
            v => panic!("unrecognized value-type {v:#08x}"),
        };
        Ok(v)
    }
}

impl Content {
    pub fn read<R: std::io::Seek + std::io::Read>(reader: &mut R) -> Result<Self, ()> {
        let magic = VersionedMagic::read(reader).unwrap();

        let tensor_count = match magic {
            VersionedMagic::GgufV1 => reader.read_u32::<LittleEndian>().unwrap() as usize,
            VersionedMagic::GgufV2 | VersionedMagic::GgufV3 => {
                reader.read_u64::<LittleEndian>().unwrap() as usize
            }
        };
        let metadata_kv_count = match magic {
            VersionedMagic::GgufV1 => reader.read_u32::<LittleEndian>().unwrap() as usize,
            VersionedMagic::GgufV2 | VersionedMagic::GgufV3 => {
                reader.read_u64::<LittleEndian>().unwrap() as usize
            }
        };

        // Read metadata
        let mut metadata = HashMap::new();
        for _idx in 0..metadata_kv_count {
            let key = read_string(reader, &magic).unwrap();
            let value_type = reader.read_u32::<LittleEndian>().unwrap();
            let value_type = ValueType::from_u32(value_type).unwrap();
            let value = Value::read(reader, value_type, &magic).unwrap();
            metadata.insert(key, value);
        }
        // Read tensor infos
        let mut tensor_infos = HashMap::new();
        for _idx in 0..tensor_count {
            let tensor_name = read_string(reader, &magic).unwrap();
            let n_dimensions = reader.read_u32::<LittleEndian>().unwrap();
            let n_elements = match magic {
                VersionedMagic::GgufV1 => {
                    let mut dimensions = vec![0; n_dimensions as usize];
                    reader
                        .read_u32_into::<LittleEndian>(&mut dimensions)
                        .unwrap();
                    dimensions.into_iter().map(|c| c as usize).product()
                }
                VersionedMagic::GgufV2 | VersionedMagic::GgufV3 => {
                    let mut dimensions = vec![0; n_dimensions as usize];
                    reader
                        .read_u64_into::<LittleEndian>(&mut dimensions)
                        .unwrap();
                    dimensions.into_iter().map(|c| c as usize).product()
                }
            };

            let ggml_dtype = reader.read_u32::<LittleEndian>().unwrap();
            let offset = reader.read_u64::<LittleEndian>().unwrap();
            tensor_infos.insert(
                tensor_name,
                (n_elements, offset as usize, GgmlDType::from_u32(ggml_dtype)),
            );
        }
        let position = reader.stream_position().unwrap();
        let alignment = match metadata.get("general.alignment") {
            Some(Value::U8(v)) => *v as u64,
            Some(Value::U16(v)) => *v as u64,
            Some(Value::U32(v)) => *v as u64,
            Some(Value::I8(v)) if *v >= 0 => *v as u64,
            Some(Value::I16(v)) if *v >= 0 => *v as u64,
            Some(Value::I32(v)) if *v >= 0 => *v as u64,
            _ => DEFAULT_ALIGNMENT,
        };
        let tensor_data_offset = position.div_ceil(alignment) * alignment;
        Ok(Self {
            magic,
            metadata,
            tensor_infos,
            tensor_data_offset,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GgmlDType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
    BF16,
}

impl GgmlDType {
    fn from_u32(u: u32) -> Self {
        match u {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2K,
            11 => Self::Q3K,
            12 => Self::Q4K,
            13 => Self::Q5K,
            14 => Self::Q6K,
            15 => Self::Q8K,
            30 => Self::BF16,
            _ => panic!("unknown dtype for tensor {u}"),
        }
    }
}

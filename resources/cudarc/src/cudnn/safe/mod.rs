//! Safe wrappers around cuDNN.
//!
//! # Convolutions
//!
//! 1. Allocate tensor descriptors with [`Cudnn::create_4d_tensor()`]
//! 2. Allocate filter descriptors with [`Cudnn::create_4d_filter()`]
//! 3. Allocate conv descriptors with [`Cudnn::create_conv2d()`]
//! 4. Instantiate one of the following algorithms with the descriptors:
//!     a. [`Conv2dForward`]
//!     b. [`Conv2dBackwardData`] for computing gradient of image
//!     c. [`Conv2dBackwardFilter`] for computing gradient of filters
//! 5. Call the `pick_algorithm` method of the struct. Specify the number of options to compare with a const generic.
//! 6. Call the `get_workspace_size` method of the struct.
//! 7. Re-allocate the workspace to the appropriate size.
//! 8. Call the `launch` method of the struct.
//!
//! # Reductions

mod conv;
mod core;
mod reduce;

pub use self::conv::{
    Conv2dBackwardData, Conv2dBackwardFilter, Conv2dDescriptor, Conv2dForward, FilterDescriptor,
};
pub use self::core::{Cudnn, CudnnDataType, TensorDescriptor};
pub use self::reduce::{FlatIndices, NoIndices, ReduceTensor, ReductionDescriptor};
pub use super::result::CudnnError;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{cudnn, driver::CudaDevice};

    #[test]
    fn test_create_descriptors() -> Result<(), CudnnError> {
        let cudnn = Cudnn::new(CudaDevice::new(0).unwrap())?;
        let _ = cudnn.create_4d_tensor_ex::<f32>([1, 2, 3, 4], [24, 12, 4, 1])?;
        let _ = cudnn.create_nd_tensor::<f64>(&[1, 2, 3, 4, 5, 6], &[720, 360, 120, 30, 6, 1])?;
        let _ = cudnn.create_4d_filter::<f32>(
            cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
            [3, 3, 3, 3],
        )?;
        let _ = cudnn.create_reduction_flat_indices::<f32>(
            cudnn::sys::cudnnReduceTensorOp_t::CUDNN_REDUCE_TENSOR_ADD,
            cudnn::sys::cudnnNanPropagation_t::CUDNN_PROPAGATE_NAN,
        )?;
        let _ = cudnn.create_reduction_no_indices::<f32>(
            cudnn::sys::cudnnReduceTensorOp_t::CUDNN_REDUCE_TENSOR_ADD,
            cudnn::sys::cudnnNanPropagation_t::CUDNN_PROPAGATE_NAN,
        )?;
        Ok(())
    }

    #[test]
    fn test_conv_pick_algorithms() -> Result<(), CudnnError> {
        let cudnn = Cudnn::new(CudaDevice::new(0).unwrap())?;

        let conv = cudnn.create_conv2d::<f32>(
            [0; 2],
            [1; 2],
            [1; 2],
            cudnn::sys::cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION,
        )?;
        let x = cudnn.create_4d_tensor::<f32>(
            cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
            [100, 128, 224, 224],
        )?;
        let filter = cudnn.create_4d_filter::<f32>(
            cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
            [256, 128, 3, 3],
        )?;
        let y = cudnn.create_4d_tensor::<f32>(
            cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
            [100, 256, 222, 222],
        )?;

        {
            let op = Conv2dForward {
                conv: &conv,
                x: &x,
                w: &filter,
                y: &y,
            };
            let algo = op.pick_algorithm()?;
            assert_eq!(
                algo,
                cudnn::sys::cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM
            );
        }

        {
            let op = Conv2dBackwardData {
                conv: &conv,
                dx: &x,
                w: &filter,
                dy: &y,
            };
            let algo = op.pick_algorithm()?;
            assert_eq!(
                algo,
                cudnn::sys::cudnnConvolutionBwdDataAlgo_t::CUDNN_CONVOLUTION_BWD_DATA_ALGO_1
            );
        }

        {
            let op = Conv2dBackwardFilter {
                conv: &conv,
                x: &x,
                dw: &filter,
                dy: &y,
            };
            let algo = op.pick_algorithm()?;
            assert_eq!(
                algo,
                cudnn::sys::cudnnConvolutionBwdFilterAlgo_t::CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1
            );
        }

        Ok(())
    }

    #[test]
    fn test_reduction() {
        let dev = CudaDevice::new(0).unwrap();
        let a = dev
            .htod_copy(std::vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
        let mut c = dev.alloc_zeros::<f32>(1).unwrap();

        let cudnn = Cudnn::new(dev.clone()).unwrap();

        let reduce = cudnn
            .create_reduction_no_indices::<f32>(
                cudnn::sys::cudnnReduceTensorOp_t::CUDNN_REDUCE_TENSOR_ADD,
                cudnn::sys::cudnnNanPropagation_t::CUDNN_PROPAGATE_NAN,
            )
            .unwrap();
        let a_desc = cudnn
            .create_nd_tensor::<f32>(&[1, 1, 2, 3], &[0, 6, 3, 1])
            .unwrap();
        let c_desc = cudnn
            .create_nd_tensor::<f32>(&[1, 1, 1, 1], &[0, 0, 0, 1])
            .unwrap();
        let op = ReduceTensor {
            reduce: &reduce,
            a: &a_desc,
            c: &c_desc,
        };

        let workspace_size = op.get_workspace_size().unwrap();
        let mut workspace = dev.alloc_zeros::<u8>(workspace_size).unwrap();

        unsafe { op.launch(&mut workspace, (1.0, 0.0), &a, &mut c) }.unwrap();

        let c_host = dev.sync_reclaim(c).unwrap();
        assert_eq!(c_host.len(), 1);
        assert_eq!(c_host[0], 21.0);
    }
}

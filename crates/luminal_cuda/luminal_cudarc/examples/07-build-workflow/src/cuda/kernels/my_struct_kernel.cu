#include "../includes/my_struct.h"

extern "C" __global__ void my_struct_kernel(MyStruct *my_structs, const size_t n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n)
    {
        my_structs[i].data[0] += 1;
        my_structs[i].data[1] += 1;
        my_structs[i].data[2] += 1;
        my_structs[i].data[3] += 1;
    }
}
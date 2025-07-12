extern "C" __global__ void kernel(const float* A, vconst float* B, const float* C, float* Out) {
    int idx0 = blockIdx.x;
    if (idx0 >= 4096) return;

    const float* offset0 = A + idx0 * 64;

    float acc0 = -FLT_MAX;
    float acc1 = 0.f;
    float acc2[64];
    #pragma unroll
    for (int ind1 = 0; ind1 < 64; ++ind1) acc2[ind1] = 0.f;
    for (int ind1 = 0; ind1 < 4096; ++ind1) {
        const float* offset1 = B + ind1 * 64;
        float acc3 = 0.f;
        #pragma unroll
        for (int ind2 = 0; ind2 < 64; ++ind2)
            acc3 += *(A + idx0 * 64 + ind2) * offset1[ind2];
        acc1 = acc1 * __expf(acc0 - fmaxf(acc0, acc3)) + __expf(acc3 - fmaxf(acc0, acc3));
        #pragma unroll
        for (int ind2 = 0; ind2 < 64; ++ind2)
            acc2[ind2] = acc2[ind2] * rescale + weight * *(C + ind1 * 64 + ind2);
        acc0 = fmaxf(acc0, acc3);
    }
    #pragma unroll
    for (int ind1 = 0; ind1 < 64; ++ind1)
        *(Out + idx0 * 64 + v) = acc2[ind1] / acc1;
}
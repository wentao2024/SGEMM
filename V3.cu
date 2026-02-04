template <const int BLOCK_SIZE>
__global__ void mysgemm_v3(
    int M, int N, int K,
    float alpha,
    const float *__restrict__ A,
    const float *__restrict__ B,
    float beta,
    float *__restrict__ C)
{
    const int BM = BLOCK_SIZE;
    const int BN = BLOCK_SIZE;
    const int BK = BLOCK_SIZE;

    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tid = threadIdx.x;
    int ty = tid / BN;
    int tx = tid %BN;
    float *Cs = &C[(by * BM )*N + (bx* BN)];
    
    __shared__ float As[BM* BK];
    __shared__ float Bs [BK * BN];
    float tmp = 0.0f;
    for ( int k = 0; k<K; k += BK){
        if ((by*BM+ty)< M &&( k+tx) < K)
            As[ty * BK + tx] = A[(by*BM + ty)* K + (k+ tx)];
        else    
            As[ty*BK+tx] = 0.0f;
        if ((k+ty)< K && (bx * BN+tx)<N)
            Bs[ty*BN + tx] = B[(k+ty)* N + ( bx * BN+tx)];
        else
            Bs[ty*BN +tx] = 0.0f;
        __syncthreads();
        #pragma unroll
        for (int i = 0; i< BK; ++i){
            tmp += As[ty*BK +i]* Bs[i* BN +tx];
        }
        __syncthreads();

    }
    if ((by * BM +ty)< M && (bx * BN + tx)<N)
        Cs[ty * N + tx] = alpha * tmp + beta * Cs[ty* N+tx];
}
template <const int BLOCK_SIZE>
__global__ void mysgemm_v2(
    int M, int N, int K,
    float alpha,
    const float *__restrict__ A,
    const float *__restrict__ B,
    float beta,
    float *__restrict__ C)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float *Cs = &C[( by * BLOCK_SIZE)*N + (bx * BLOCK_SIZE)];
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    float tmp = 0.0f;
    for ( int k = 0; k<K; k+=BLOCK_SIZE){
        if ((by*BLOCK_SIZE +ty)< M && (k+tx)<K)
            As[ty][tx]= A[(by* BLOCK_SIZE+ty)*K+(k+tx)];
        else 
            As[ty][tx]= 0.0f;
        if ((k+ ty)< K && (bx * BLOCK_SIZE + tx)<N)
            Bs[ty][tx] = B[(k+ ty)*N + (bx *BLOCK_SIZE + tx)];
        else 
            Bs[ty][tx] = 0.0f;
        __syncthreads();

        for ( int i = 0 ; i<BLOCK_SIZE; ++i){
            tmp += As[ty][i] *BS[i][tx];

        }
        __syncthreads();

    }
    if ((by * BLOCK_SIZE + ty)< M && (bx+ BLOCK_SIZE + tx )< N)
        Cs[ty * N+ tx] = alpha * tmp + beta * Cs[ty * N+ tx];
}

 
template <const int BLOCK_SIZE>
__global__ void mysgemm_v1(
    int M, int N, int K, float alpha, const float *__restrict_A,
    const float *__restrict_B, float beta, float * __restrict_C
)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;
    float tmp = 0.0f;

    if (row < M && col < N){
        for (int k =0; k<K; ++k){
            tmp += A[row*K + k] * B[k*N + col];
        }
        C[row * N + col] = alpha * tmp + beta * C[row * N + col];
    }

}


main :

dim3 block(BLOCK_SIZE, BLOCK_SIZE);
dim3 grid(N + BLOCK_SIZE - 1 )/ BLOCK_SIZE, (M+BLOCK_SIZE-1)BLOCK_SIZE);
mysgemm_v1<BLOCK_SIZE><<<grid,block>>>();
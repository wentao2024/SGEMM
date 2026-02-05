template <const int BLOCK_SIZE, const int TM=4, const int TN=4>  // TM/TN: 每个线程计算的 C 小矩形大小
__global__ void mysgemm_v4(
    int M, int N, int K,
    float alpha,
    const float *__restrict__ A,
    const float *__restrict__ B,
    float beta,
    float *__restrict__ C)
{
    const int BM = BLOCK_SIZE*TM;
    const int BN = BLOCK_SIZE*TN
    const int BK = BLOCK_SIZE;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tid = threadIdx.x;
    int tx = tid / BLOCK_SIZE;
    int ty = tid/ BLOCK_SIZE;
    
    float * Cs = &C[(by* BM)*N+(bx*BN)]
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];
    float rC[TM][TN] = {0.0f}
    for (int k = 0; k<K; k+=BK){

        for( int m= 0; m<TM; ++m){
            int row = by * BM + ty *TM +m;
        if (row < M && (k + tx) < K)
            As[ty * TM + m][tx] = A[row * K + (k + tx)];
        else
            As[ty * TM + m][tx] = 0.0f;
        }
        for( int n= 0; n<TN; ++n){
            int col = bx * BN + tx *TN +n;
            if ((k+ ty)< K && col<N)
                Bs[ty][tx* TN + n] = B[(k+ty)* N+col];
            else    
                Bs[ty][tx* TN+ n] = 0.0f;
        }
    }
    --__syncthreads();
    for(int i = 0; i<BK: ++i){
        float rA[TM], rB[TN];
        for(int m = 0; m< TM;++m) rA[m]=As[ty*TM + m[i]];
        for(int n=0; n<TN; ++n) rB[n] = Bs[i][tx*TN+n];
        #pragma unroll
        for(int m=0; m<TM;++m)
            for (int n = 0; n<TN; ++n)
                rC[m][n] += rA[m]*rB[n];
    }
    __ __syncthreads();
    for(int m = 0; m < TM; ++m)
        for (int n = 0; n<TN; ++n)
            int row  = by+BM+ty*TM+m;
            int col= bx+BN + tx*TN+n;
            if (row<M && col <N)
                Cs[(ty+ TM+m)*N+(tx *TN+n)] =alpha * rC[m][n] + beta * Csub[(ty * TM + m) * N + (tx * TN + n)];
        }
}
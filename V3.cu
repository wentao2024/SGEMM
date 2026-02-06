

    %%writefile mysgemm_v3.cu
    #include <iostream>
    #include <vector>
    #include <cuda_runtime.h>

    template <const int BLOCK_SIZE>
    __global__ void mysgemm_v3(int M, int N, int K, float alpha, 
                            const float *__restrict__ A, 
                            const float *__restrict__ B, 
                            float beta, float *__restrict__ C) {
        const int BM = BLOCK_SIZE;
        const int BN = BLOCK_SIZE;
        const int BK = BLOCK_SIZE;

        int bx = blockIdx.x;
        int by = blockIdx.y;
        int tid = threadIdx.x;
        int ty = tid / BN;
        int tx = tid % BN;

        // C 子块指针
        float *Csub = &C[(by * BM) * N + (bx * BN)];

        __shared__ float As[BM * BK];
        __shared__ float Bs[BK * BN];

        float tmp = 0.0f;

        // 沿 K 维度分块循环
        for (int k = 0; k < K; k += BK) {
            // 搬运 A 块到 Shared Memory (带边界检查)
            if ((by * BM + ty) < M && (k + tx) < K)
                As[ty * BK + tx] = A[(by * BM + ty) * K + (k + tx)];
            else
                As[ty * BK + tx] = 0.0f;

            // 搬运 B 块到 Shared Memory
            if ((k + ty) < K && (bx * BN + tx) < N)
                Bs[ty * BN + tx] = B[(k + ty) * N + (bx * BN + tx)];
            else
                Bs[ty * BN + tx] = 0.0f;

            __syncthreads(); // 等待所有人搬运完

            // 计算当前分块的点积
            #pragma unroll
            for (int i = 0; i < BK; i++) {
                tmp += As[ty * BK + i] * Bs[i * BN + tx];
            }

            __syncthreads(); // 等待所有人算完再进入下一轮搬运
        }

        // 写回结果到 C
        if ((by * BM + ty) < M && (bx * BN + tx) < N) {
            Csub[ty * N + tx] = alpha * tmp + beta * Csub[ty * N + tx];
        }
    }

    // --- 驱动测试程序 ---
    int main() {
        int M = 1024, N = 1024, K = 1024;
        size_t size_A = M * K * sizeof(float);
        size_t size_B = K * N * sizeof(float);
        size_t size_C = M * N * sizeof(float);

        std::vector<float> h_A(M * K, 1.0f);
        std::vector<float> h_B(K * N, 2.0f);
        std::vector<float> h_C(M * N, 0.0f);

        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, size_A);
        cudaMalloc(&d_B, size_B);
        cudaMalloc(&d_C, size_C);

        cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B.data(), size_B, cudaMemcpyHostToDevice);

        const int BLOCK_SIZE = 32;
        dim3 threads(BLOCK_SIZE * BLOCK_SIZE); 
        dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
        
        std::cout << "正在运行 mysgemm_v3 (Tiling 版本)..." << std::endl;
        mysgemm_v3<BLOCK_SIZE><<<blocks, threads>>>(M, N, K, 1.0f, d_A, d_B, 0.0f, d_C);

        cudaDeviceSynchronize();
        cudaMemcpy(h_C.data(), d_C, size_C, cudaMemcpyDeviceToHost);

        std::cout << "计算完成！结果验证 (第一个元素): " << h_C[0] << " (预期值: 2048)" << std::endl;

        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        return 0;
    }
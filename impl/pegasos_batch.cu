// pegasos_batch.cu
#include <math.h>
#include <curand.h>
#include <curand_kernel.h>

__global__
void pegasos_batch_kernel(
    float *w,
    const float *X,
    const float *y,
    const int *A_t,
    int k,
    int d,
    float lam,
    float eta
)
{
    int bi = blockIdx.x
    int j  = threadIdx.x;

    if (bi >= k || j >= d) return;

    int i = A_t[bi];

    float margin = 0.0f;
    for (int jj = 0; jj < d; jj++)
        margin += w[jj] * X[i * d + jj];

    if (y[i] * margin < 1.0f) {
        float wij = w[j];
        float xij = X[i * d + j];
        float update = (1 - eta * lam) * wij + (eta / k) * (y[i] * xij);
        w[j] = update;
    }
}

void pegasos_batch_gpu(
    float *d_w,
    const float *d_X,
    const float *d_y,
    int n,
    int d,
    int k,
    float lam,
    int T
)
{
    int *d_A_t;
    cudaMalloc(&d_A_t, k * sizeof(int));

    dim3 block(d);
    dim3 grid(k);

    for (int t = 1; t <= T; t++) {
        int *A_t = (int*) malloc(k * sizeof(int));
        for (int i = 0; i < k; i++)
            A_t[i] = rand() % n;

        cudaMemcpy(d_A_t, A_t, k * sizeof(int), cudaMemcpyHostToDevice);

        float eta = 1.0f / (lam * t);

        pegasos_batch_kernel<<<grid, block>>>(
            d_w, d_X, d_y, d_A_t, k, d, lam, eta
        );

        free(A_t);
    }

    cudaFree(d_A_t);
}
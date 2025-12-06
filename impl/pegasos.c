#include <math.h>
#include <stdlib.h>

void pegasos_batch(
    float *w,
    const float *X,
    const float *y,
    int n,
    int d,
    int k,
    float lam,
    int T,
    int norm_flag
)
{
    float *update = (float*) malloc(sizeof(float) * d);

    for (int t = 1; t <= T; t++) {
        int *A_t = (int*) malloc(sizeof(int) * k);
        for (int i = 0; i < k; i++)
            A_t[i] = rand() % n;

        for (int j = 0; j < d; j++)
            update[j] = 0.0f;

        for (int bi = 0; bi < k; bi++) {
            int i = A_t[bi];

            float margin = 0.0f;
            const float *Xi = &X[i * d];
            for (int j = 0; j < d; j++)
                margin += w[j] * Xi[j];

            if (y[i] * margin < 1.0f) {
                for (int j = 0; j < d; j++)
                    update[j] += y[i] * Xi[j];
            }
        }

        float eta = 1.0f / (lam * t);

        for (int j = 0; j < d; j++)
            w[j] = (1 - eta * lam) * w[j] + (eta / k) * update[j];

        if (norm_flag) {
            float norm = 0.0f;
            for (int j = 0; j < d; j++)
                norm += w[j] * w[j];
            norm = sqrtf(norm);

            float factor = 1.0f;
            float thr = 1.0f / (sqrtf(lam) * norm);
            if (thr < 1.0f)
                factor = thr;

            for (int j = 0; j < d; j++)
                w[j] *= factor;
        }

        free(A_t);
    }

    free(update);
}

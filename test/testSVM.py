import unittest
import numpy as np

from simulator import simulator
from algorithms.pegasos import SVM
from algorithms import arithmetic
from util import representation


class TestPIMSVM(unittest.TestCase):

    def test_pim_svm_training(self):
        dtype = arithmetic.RealArithmetic.DataType.IEEE_FLOAT32
        N = dtype.N

        np.random.seed(0)
        n = 200
        d = 64
        T = 50
        lam = 0.05

        num_rows = d
        num_cols = n * N * 3 + 4096

        X = np.random.randn(n, d).astype(np.float32)
        true_w = np.random.randn(d).astype(np.float32)
        y = np.sign((X @ true_w)).astype(np.float32)

        X = X.T
        true_w = true_w.T
        y = y.flatten()

        X_addr = np.zeros(n * N, dtype=int)
        for j in range(n):
            X_addr[j * N: (j + 1) * N] = np.arange(j * N, (j + 1) * N)

        y_addr = np.arange(n * N, n * N + n * N)

        w_addr = np.arange(n * N + n * N, n * N + n * N + N)

        inter_addr = np.arange(n * N + n * N + n * N, num_cols)

        sim = simulator.SerialSimulator(num_rows, num_cols)

        for i in range(d):
            row_vals = X[i, :].reshape(1, n)
            row_bin = representation.signedFloatToBinary(row_vals)

            for j in range(n):
                cols = X_addr[j * N:(j + 1) * N]
                sim.memory[cols, i] = row_bin[:, j]

        y_bin = representation.signedFloatToBinary(y.reshape(1, n))
        sim.memory[y_addr, 0] = y_bin.T.flatten()

        w_init = np.zeros((d, 1), dtype=np.float32)
        w_bin = representation.signedFloatToBinary(w_init.T)

        for i in range(d):
            cols = w_addr
            sim.memory[cols, i] = w_bin[:, i]

        SVM.trainSGDSVM(sim,
                        X_addr=X_addr,
                        y_addr=y_addr,
                        w_addr=w_addr,
                        inter=inter_addr.copy(),
                        T=T,
                        lam=lam,
                        dimensions=d,
                        dtype=dtype)

        W = np.zeros((1, d), dtype=np.float32)
        for j in range(d):
            w_j_bin = sim.memory[w_addr, j]
            W[0, j] = representation.binaryToSignedFloat(w_j_bin.reshape(N, 1))[0, 0]

        preds = np.sign(X.T @ W.T).T
        accuracy = np.mean(preds == y)
        print("Training accuracy:", accuracy)

        #self.assertGreater(accuracy, 0.80,"PIM SVM accuracy is too low (expected > 0.8)")

        print(f'Real {N}-bit SVM on {d}-dimensional data of {n} elements for {T} iterations with {sim.latency} cycles and {sim.energy // num_rows} average energy.')

    def test_pim_batch_svm_training(self):
        dtype = arithmetic.RealArithmetic.DataType.IEEE_FLOAT32
        N = dtype.N

        np.random.seed(0)
        n = 50
        d = 4
        T = 5
        batch_size = 5
        lam = 0.5

        num_rows = d
        num_cols = n * N * 3 + 4096

        X = np.random.randn(n, d).astype(np.float32)
        true_w = np.random.randn(d).astype(np.float32)
        y = np.sign((X @ true_w)).astype(np.float32)

        X = X.T
        true_w = true_w.T
        y = y.flatten()

        X_addr = np.zeros(n * N, dtype=int)
        for j in range(n):
            X_addr[j * N: (j + 1) * N] = np.arange(j * N, (j + 1) * N)

        y_addr = np.arange(n * N, n * N + n * N)

        w_addr = np.arange(n * N + n * N, n * N + n * N + N)

        inter_addr = np.arange(n * N + n * N + n * N, num_cols)

        sim = simulator.SerialSimulator(num_rows, num_cols)

        for i in range(d):
            row_vals = X[i, :].reshape(1, n)
            row_bin = representation.signedFloatToBinary(row_vals)

            for j in range(n):
                cols = X_addr[j * N:(j + 1) * N]
                sim.memory[cols, i] = row_bin[:, j]

        y_bin = representation.signedFloatToBinary(y.reshape(1, n))
        sim.memory[y_addr, 0] = y_bin.T.flatten()

        w_init = np.zeros((d, 1), dtype=np.float32)
        w_bin = representation.signedFloatToBinary(w_init.T)

        for i in range(d):
            cols = w_addr
            sim.memory[cols, i] = w_bin[:, i]

        SVM.trainSGDSVMBatch(sim,
                        X_addr=X_addr,
                        y_addr=y_addr,
                        w_addr=w_addr,
                        inter=inter_addr.copy(),
                        T=T,
                        lam=lam,
                        dimensions=d,
                        batch_size=batch_size,
                        dtype=dtype)

        W = np.zeros((1, d), dtype=np.float32)
        for j in range(d):
            w_j_bin = sim.memory[w_addr, j]
            W[0, j] = representation.binaryToSignedFloat(w_j_bin.reshape(N, 1))[0, 0]

        print(W)
        preds = np.sign(X.T @ W.T).T
        accuracy = np.mean(preds == y)
        print("Training accuracy:", accuracy)

        #self.assertGreater(accuracy, 0.80, "PIM SVM accuracy is too low (expected > 0.8)")

        print(f'Real {N}-bit batched SVM on {d}-dimensional data of {n} elements for {T} iterations with {sim.latency} cycles and {sim.energy // num_rows} average energy.')
        #Real 32-bit batched SVM on 1024-dimensional data of 4096 elements for 25 iterations with 2136379 cycles and 46879297 average energy.

    # def test_pim_batch_svm_training(self):
    #     return
    #     dtype = arithmetic.RealArithmetic.DataType.IEEE_FLOAT32
    #     N = dtype.N
    #
    #     np.random.seed(0)
    #     n = 200
    #     d = 4
    #     T = 5
    #     batch_size = 5
    #     lam = 0.05
    #
    #     num_rows = d * batch_size
    #     num_cols = n * N * 3 + 4096
    #
    #     X = np.random.randn(n, d).astype(np.float32)
    #     true_w = np.random.randn(d).astype(np.float32)
    #     y = np.sign((X @ true_w)).astype(np.float32)
    #
    #     X = X.T
    #     true_w = true_w.T
    #     y = y.flatten()
    #
    #     X_addr = np.zeros(n * N, dtype=int)
    #     for j in range(n):
    #         X_addr[j * N: (j + 1) * N] = np.arange(j * N, (j + 1) * N)
    #
    #     y_addr = np.arange(n * N, n * N + n * N)
    #
    #     w_addr = np.arange(n * N + n * N, n * N + n * N + N)
    #
    #     inter_addr = np.arange(n * N + n * N + n * N, num_cols)
    #
    #     sim = simulator.SerialSimulator(num_rows, num_cols)
    #
    #     for i in range(d):
    #         row_vals = X[i, :].reshape(1, n)
    #         row_bin = representation.signedFloatToBinary(row_vals)
    #
    #         for j in range(n):
    #             cols = X_addr[j * N:(j + 1) * N]
    #             sim.memory[cols, i] = row_bin[:, j]
    #
    #     y_bin = representation.signedFloatToBinary(y.reshape(1, n))
    #     sim.memory[y_addr, 0] = y_bin.T.flatten()
    #
    #     w_init = np.zeros((d, 1), dtype=np.float32)
    #     w_bin = representation.signedFloatToBinary(w_init.T)
    #
    #     for i in range(d):
    #         cols = w_addr
    #         sim.memory[cols, i] = w_bin[:, i]
    #
    #     SVM.trainSGDSVMBatch2(sim,
    #                     X_addr=X_addr,
    #                     y_addr=y_addr,
    #                     w_addr=w_addr,
    #                     inter=inter_addr.copy(),
    #                     T=T,
    #                     lam=lam,
    #                     dimensions=d,
    #                     batch_size=batch_size,
    #                     dtype=dtype)
    #
    #     W = np.zeros((1, d), dtype=np.float32)
    #     for j in range(d):
    #         w_j_bin = sim.memory[w_addr, j]
    #         W[0, j] = representation.binaryToSignedFloat(w_j_bin.reshape(N, 1))[0, 0]
    #
    #     preds = np.sign(X.T @ W.T).T
    #     accuracy = np.mean(preds == y)
    #     print("Training accuracy:", accuracy)
    #
    #     self.assertGreater(accuracy, 0.80,
    #                        "PIM SVM accuracy is too low (expected > 0.8)")
    #
    #     print(f'Real {N}-bit batched SVM on {d}-dimensional data of {n} elements for {T} iterations with {sim.latency} cycles and {sim.energy // num_rows} average energy.')

if __name__ == "__main__":
    unittest.main()

import warnings

import numpy as np

from algorithms.arithmetic import IntermediateAllocator, RealArithmetic
from simulator import simulator
from algorithms import arithmetic
from util import representation


class SVM:
    @staticmethod
    def trainSGDSVM(sim: simulator.SerialSimulator, X_addr: np.ndarray, y_addr: np.ndarray,
                    w_addr: np.ndarray, inter, T: int = 10, lam: float = 0.05, dimensions: int = 3,
                    dtype=arithmetic.RealArithmetic.DataType.IEEE_FLOAT32):

        if isinstance(inter, np.ndarray):
            inter = IntermediateAllocator(inter)

        N = dtype.N
        n = len(X_addr) // N

        for i in range(len(w_addr)):
            sim.perform(simulator.GateType.INIT0, [], [w_addr[i]], simulator.GateDirection.IN_ROW)

        y_i_c = inter.malloc(N)
        eta_t_c = inter.malloc(N)
        f_c = inter.malloc(N)
        hinge_c = inter.malloc(N)
        for t in range(1, T + 1):
            i = np.random.randint(0, n)
            eta_scalar = 1.0 / (lam * t)
            eta_bin = representation.signedFloatToBinary(np.array([[eta_scalar]], dtype=np.float32))

            for di in range(dimensions):
                sim.write(di, eta_t_c, eta_bin.reshape(-1))

            x_i = X_addr[i * 32:i * 32 + 32]
            y_i = y_addr[i * 32:i * 32 + 32]

            for di in range(dimensions):
                sim.write(di, y_i_c, sim.read(0, y_i))

            RealArithmetic.dot(sim, x_i, w_addr, hinge_c, inter, dimensions, dtype)
            hinge_data = representation.binaryToSignedFloat(sim.memory[hinge_c]).astype(np.float32)
            hinge = hinge_data.flatten()[0] * representation.binaryToSignedFloat(sim.memory[y_i]).astype(np.float32).flatten()[0]
            f = (1 - eta_scalar * lam)

            f_bin = representation.signedFloatToBinary(np.array([[f]], dtype=np.float32))
            for di in range(dimensions):
                sim.write(di, f_c, f_bin.reshape(-1))
            if hinge < 1:
                temp1 = inter.malloc(N)
                RealArithmetic.mult(sim, f_c, w_addr, temp1, inter, dtype)
                temp2 = inter.malloc(N)
                RealArithmetic.mult(sim, x_i, y_i_c, temp2, inter, dtype)
                temp3 = inter.malloc(N)
                RealArithmetic.mult(sim, eta_t_c, temp2, temp3, inter, dtype)
                RealArithmetic.add(sim, temp1, temp3, w_addr, inter, dtype)
                inter.free(temp1)
                inter.free(temp2)
                inter.free(temp3)
            else:
                temp1 = inter.malloc(N)
                RealArithmetic.mult(sim, f_c, w_addr, temp1, inter, dtype)
                RealArithmetic.copy(sim, temp1, w_addr, inter, dtype)
                inter.free(temp1)

        inter.free(hinge_c)
        inter.free(eta_t_c)
        inter.free(f_c)
        inter.free(y_i_c)

    def trainSGDSVMBatch(sim: simulator.SerialSimulator, X_addr: np.ndarray, y_addr: np.ndarray,
                    w_addr: np.ndarray, inter, T: int = 10, lam: float = 0.05, dimensions: int = 3, batch_size=10,
                    dtype=arithmetic.RealArithmetic.DataType.IEEE_FLOAT32):

        if isinstance(inter, np.ndarray):
            inter = IntermediateAllocator(inter)

        N = dtype.N
        n = len(X_addr) // N

        for i in range(len(w_addr)):
            sim.perform(simulator.GateType.INIT0, [], [w_addr[i]], simulator.GateDirection.IN_ROW)

        y_i_c = inter.malloc(N * batch_size)
        eta_t_c = inter.malloc(N)
        f_c = inter.malloc(N)
        hinge_c = inter.malloc(N)
        x_i_c = inter.malloc(N)

        m = range(0, n)
        for t in range(1, T + 1):
            A_t = np.random.choice(m, size=batch_size, replace=False)

            x_i = [None] * batch_size
            y_i = []
            for j in range(batch_size):
                x_i[j] = X_addr[A_t[j] * 32:A_t[j] * 32 + 32]
                block = y_addr[A_t[j] * 32: A_t[j] * 32 + 32]
                y_i.extend(block)

            eta_scalar = 1.0 / (lam * t)
            eta_bin = representation.signedFloatToBinary(np.array([[eta_scalar]], dtype=np.float32))

            for di in range(dimensions):
                sim.write(di, eta_t_c, eta_bin.reshape(-1))

            for di in range(dimensions):
                sim.write(di, y_i_c, sim.read(0, y_i))

            for j in range(batch_size):
                for di in range(dimensions):
                    sim.write(di + j * dimensions, x_i_c, sim.read(di, x_i[j]))

            for j in range(1, batch_size):
                for di in range(dimensions):
                    sim.write(di + j * dimensions, w_addr, sim.read(di, w_addr))

            RealArithmetic.batch_dot(sim, x_i_c, w_addr, hinge_c, inter, dimensions, batch_size, dtype)
            with warnings.catch_warnings():
                #May get an overflow on cringe irrelevant data, not a concern
                warnings.simplefilter("ignore", RuntimeWarning)
                hinge_data = representation.binaryToSignedFloat(sim.memory[hinge_c]).astype(np.float32)
            flat = hinge_data.flatten()
            z_blocks = flat[0::dimensions]
            y_class = []
            for j in range(batch_size):
                y_class.append(representation.binaryToSignedFloat(sim.memory[y_i[j * 32: j * 32 + 32]]).astype(np.float32).flatten()[0])
            hinges = z_blocks * y_class
            f = (1 - eta_scalar * lam)

            f_bin = representation.signedFloatToBinary(np.array([[f]], dtype=np.float32))
            for di in range(dimensions):
                sim.write(di, f_c, f_bin.reshape(-1))
            for i in range(batch_size):
                if hinges[i] < 1:
                    temp1 = inter.malloc(N)
                    RealArithmetic.mult(sim, f_c, w_addr, temp1, inter, dtype)
                    temp2 = inter.malloc(N)
                    RealArithmetic.mult(sim, x_i[i], y_i_c[i * 32:i * 32 + 32], temp2, inter, dtype)
                    temp3 = inter.malloc(N)
                    RealArithmetic.mult(sim, eta_t_c, temp2, temp3, inter, dtype)
                    RealArithmetic.add(sim, temp1, temp3, w_addr, inter, dtype)
                    inter.free(temp1)
                    inter.free(temp2)
                    inter.free(temp3)
                else:
                    temp1 = inter.malloc(N)
                    RealArithmetic.mult(sim, f_c, w_addr, temp1, inter, dtype)
                    RealArithmetic.copy(sim, temp1, w_addr, inter, dtype)
                    inter.free(temp1)

        inter.free(hinge_c)
        inter.free(eta_t_c)
        inter.free(f_c)
        inter.free(y_i_c)
        inter.free(x_i_c)
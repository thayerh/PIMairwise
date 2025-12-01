import unittest
import numpy as np
from simulator import simulator
from algorithms import arithmetic
from util import representation


class TestArithmetic(unittest.TestCase):
    """
    Tests the arithmetic functions on real and complex data.
    """

    def test_realAdd(self):
        """
        Tests the real addition algorithm
        """

        # Parameters
        dtype = arithmetic.RealArithmetic.DataType.IEEE_FLOAT32
        N = dtype.N
        num_rows = 1024
        num_cols = 1024

        # Address allocation
        x_addr = np.arange(0, N)
        y_addr = np.arange(N, 2 * N)
        z_addr = np.arange(2 * N, 3 * N)
        inter_addr = np.arange(3 * N, num_cols)

        # Define the simulator
        sim = simulator.SerialSimulator(num_rows, num_cols)

        # Sample the inputs at random
        x = np.random.random((1, num_rows)).astype(np.float32)
        y = np.random.random((1, num_rows)).astype(np.float32)

        # Write the inputs to the memory
        sim.memory[x_addr] = representation.signedFloatToBinary(x)
        sim.memory[y_addr] = representation.signedFloatToBinary(y)

        # Perform the arithmetic algorithm
        arithmetic.RealArithmetic.add(sim, x_addr, y_addr, z_addr, inter_addr, dtype)

        # Read the output from the memory
        z = representation.binaryToSignedFloat(sim.memory[z_addr]).astype(np.float32)

        # Verify correctness
        np.seterr(over='ignore')
        expected = x + y
        # Generate mask to avoid cases where an overflow occurred
        mask = np.logical_and(np.logical_not(np.isinf(expected)), np.logical_or(expected == 0, np.abs(expected) >= np.finfo(np.float32).tiny))
        self.assertTrue(((z == expected)[mask]).all())

        print(f'Real {N}-bit Addition with {sim.latency} cycles and {sim.energy//num_rows} average energy.')

    def test_realSub(self):
        """
        Tests the real subtraction algorithm
        """

        # Parameters
        dtype = arithmetic.RealArithmetic.DataType.IEEE_FLOAT32
        N = dtype.N
        num_rows = 1024
        num_cols = 1024

        # Address allocation
        x_addr = np.arange(0, N)
        y_addr = np.arange(N, 2 * N)
        z_addr = np.arange(2 * N, 3 * N)
        inter_addr = np.arange(3 * N, num_cols)

        # Define the simulator
        sim = simulator.SerialSimulator(num_rows, num_cols)

        # Sample the inputs at random
        x = np.random.random((1, num_rows)).astype(np.float32)
        y = np.random.random((1, num_rows)).astype(np.float32)

        # Write the inputs to the memory
        sim.memory[x_addr] = representation.signedFloatToBinary(x)
        sim.memory[y_addr] = representation.signedFloatToBinary(y)

        # Perform the arithmetic algorithm
        arithmetic.RealArithmetic.sub(sim, x_addr, y_addr, z_addr, inter_addr, dtype)

        # Read the output from the memory
        z = representation.binaryToSignedFloat(sim.memory[z_addr]).astype(np.float32)

        # Verify correctness
        np.seterr(over='ignore')
        expected = x - y
        # Generate mask to avoid cases where an overflow occurred
        mask = np.logical_and(np.logical_not(np.isinf(expected)), np.logical_or(expected == 0, np.abs(expected) >= np.finfo(np.float32).tiny))
        self.assertTrue(((z == expected)[mask]).all())

        print(f'Real {N}-bit Subtraction with {sim.latency} cycles and {sim.energy//num_rows} average energy.')

    def test_realDot(self):
        """
        Tests the real dot algorithm
        """

        # Parameters
        dtype = arithmetic.RealArithmetic.DataType.IEEE_FLOAT32
        N = dtype.N
        num_rows = 1024
        num_cols = 1024

        # Address allocation
        x_addr = np.arange(0, N)
        y_addr = np.arange(N, 2 * N)
        z_addr = np.arange(2 * N, 3 * N)
        inter_addr = np.arange(3 * N, num_cols)

        # Define the simulator
        sim = simulator.SerialSimulator(num_rows, num_cols)

        # Sample the inputs at random
        x = np.random.random((1, num_rows)).astype(np.float32)
        y = np.random.random((1, num_rows)).astype(np.float32)

        # Write the inputs to the memory
        sim.memory[x_addr] = representation.signedFloatToBinary(x)
        sim.memory[y_addr] = representation.signedFloatToBinary(y)

        # Perform the arithmetic algorithm
        arithmetic.RealArithmetic.dot(sim, x_addr, y_addr, z_addr, inter_addr, num_rows, dtype)

        # Read the output from the memory
        z = representation.binaryToSignedFloat(sim.memory[z_addr]).astype(np.float32)

        # Verify correctness
        np.seterr(over='ignore')
        expected = np.dot(x.flatten(), y.flatten())
        # print(z.flatten()[0])
        # print(expected)
        self.assertTrue((z.flatten()[0] - expected < 1e-4).all())

        print(f'Real {N}-bit Dot Product with {sim.latency} cycles and {sim.energy//num_rows} average energy.')

    def test_realBatchDot(self):
        """
        Tests the real dot algorithm
        """

        # Parameters
        dtype = arithmetic.RealArithmetic.DataType.IEEE_FLOAT32
        N = dtype.N
        num_rows = 4*5
        num_cols = 1024

        block_size = 4
        num_blocks = 5

        # Address allocation
        x_addr = np.arange(0, N)
        y_addr = np.arange(N, 2 * N)
        z_addr = np.arange(2 * N, 3 * N)
        inter_addr = np.arange(3 * N, num_cols)

        # Define the simulator
        sim = simulator.SerialSimulator(num_rows, num_cols)

        # Sample the inputs at random
        x = np.random.random((1, num_rows)).astype(np.float32)
        y = np.random.random((1, num_rows)).astype(np.float32)

        # Write the inputs to the memory
        sim.memory[x_addr] = representation.signedFloatToBinary(x)
        sim.memory[y_addr] = representation.signedFloatToBinary(y)

        # Perform the arithmetic algorithm
        arithmetic.RealArithmetic.batch_dot(sim, x_addr, y_addr, z_addr, inter_addr, block_size, num_blocks, dtype)

        # Read the output from the memory
        z = representation.binaryToSignedFloat(sim.memory[z_addr]).astype(np.float32)

        # Verify correctness
        np.seterr(over='ignore')

        dot_products = []

        for i in range(num_blocks):
            start = i * block_size
            end = start + block_size

            dot_i = np.dot(x[0, start:end], y[0, start:end])
            dot_products.append(dot_i)

        expected_dot_products = np.array(dot_products)

        flat = z.flatten()
        z_blocks = flat[0::block_size]
        exp = expected_dot_products
        # print(z_blocks)
        # print(expected_dot_products)
        self.assertTrue(np.allclose(z_blocks, exp, atol=1e-4))

        print(f'Real {N}-bit Batched Dot Product with {sim.latency} cycles and {sim.energy//num_rows} average energy.')

    def test_realMult(self):
        """
        Tests the real subtraction algorithm
        """

        # Parameters
        dtype = arithmetic.RealArithmetic.DataType.IEEE_FLOAT32
        N = dtype.N
        num_rows = 1024
        num_cols = 1024

        # Address allocation
        x_addr = np.arange(0, N)
        y_addr = np.arange(N, 2 * N)
        z_addr = np.arange(2 * N, 3 * N)
        inter_addr = np.arange(3 * N, num_cols)

        # Define the simulator
        sim = simulator.SerialSimulator(num_rows, num_cols)

        # Sample the inputs at random
        x = np.random.random((1, num_rows)).astype(np.float32)
        y = np.random.random((1, num_rows)).astype(np.float32)

        # Write the inputs to the memory
        sim.memory[x_addr] = representation.signedFloatToBinary(x)
        sim.memory[y_addr] = representation.signedFloatToBinary(y)

        # Perform the arithmetic algorithm
        arithmetic.RealArithmetic.mult(sim, x_addr, y_addr, z_addr, inter_addr, dtype)

        # Read the output from the memory
        z = representation.binaryToSignedFloat(sim.memory[z_addr]).astype(np.float32)

        # Verify correctness
        np.seterr(over='ignore')
        expected = x * y
        # Generate mask to avoid cases where an overflow occurred
        mask = np.logical_and(np.logical_not(np.isinf(expected)), np.logical_or(expected == 0, np.abs(expected) >= np.finfo(np.float32).tiny))
        self.assertTrue(((z == expected)[mask]).all())

        print(f'Real {N}-bit Multiplication with {sim.latency} cycles and {sim.energy//num_rows} average energy.')

    def test_complexAdd(self):
        """
        Tests the complex addition algorithm
        """

        # Parameters
        dtype = arithmetic.ComplexArithmetic.DataType.IEEE_CFLOAT32
        N = dtype.N
        num_rows = 1024
        num_cols = 1024

        # Address allocation
        x_addr = np.arange(0, N)
        y_addr = np.arange(N, 2 * N)
        z_addr = np.arange(2 * N, 3 * N)
        inter_addr = np.arange(3 * N, num_cols)

        # Define the simulator
        sim = simulator.SerialSimulator(num_rows, num_cols)

        # Sample the inputs at random
        x = np.random.random((1, num_rows)).astype(np.csingle) + 1j * np.random.random((1, num_rows)).astype(np.csingle)
        y = np.random.random((1, num_rows)).astype(np.csingle) + 1j * np.random.random((1, num_rows)).astype(np.csingle)

        # Write the inputs to the memory
        sim.memory[x_addr] = representation.signedComplexFloatToBinary(x)
        sim.memory[y_addr] = representation.signedComplexFloatToBinary(y)

        # Perform the arithmetic algorithm
        arithmetic.ComplexArithmetic.add(sim, x_addr, y_addr, z_addr, inter_addr, dtype)

        # Read the output from the memory
        z = representation.binaryToSignedComplexFloat(sim.memory[z_addr]).astype(np.csingle)

        # Verify correctness
        np.seterr(over='ignore')
        expected = x + y
        # Generate mask to avoid cases where an overflow occurred
        mask = np.logical_and(np.logical_not(np.isinf(expected)), np.logical_or(expected == 0, np.abs(expected) >= np.finfo(np.float32).tiny))
        self.assertTrue(((z == expected)[mask]).all())

        print(f'Complex {N}-bit Addition with {sim.latency} cycles and {sim.energy//num_rows} average energy.')

    def test_complexSub(self):
        """
        Tests the complex subtraction algorithm
        """

        # Parameters
        dtype = arithmetic.ComplexArithmetic.DataType.IEEE_CFLOAT32
        N = dtype.N
        num_rows = 1024
        num_cols = 1024

        # Address allocation
        x_addr = np.arange(0, N)
        y_addr = np.arange(N, 2 * N)
        z_addr = np.arange(2 * N, 3 * N)
        inter_addr = np.arange(3 * N, num_cols)

        # Define the simulator
        sim = simulator.SerialSimulator(num_rows, num_cols)

        # Sample the inputs at random
        x = np.random.random((1, num_rows)).astype(np.csingle) + 1j * np.random.random((1, num_rows)).astype(np.csingle)
        y = np.random.random((1, num_rows)).astype(np.csingle) + 1j * np.random.random((1, num_rows)).astype(np.csingle)

        # Write the inputs to the memory
        sim.memory[x_addr] = representation.signedComplexFloatToBinary(x)
        sim.memory[y_addr] = representation.signedComplexFloatToBinary(y)

        # Perform the arithmetic algorithm
        arithmetic.ComplexArithmetic.sub(sim, x_addr, y_addr, z_addr, inter_addr, dtype)

        # Read the output from the memory
        z = representation.binaryToSignedComplexFloat(sim.memory[z_addr]).astype(np.csingle)

        # Verify correctness
        np.seterr(over='ignore')
        expected = x - y
        # Generate mask to avoid cases where an overflow occurred
        mask = np.logical_and(np.logical_not(np.isinf(expected)), np.logical_or(expected == 0, np.abs(expected) >= np.finfo(np.float32).tiny))
        self.assertTrue(((z == expected)[mask]).all())

        print(f'Complex {N}-bit Subtraction with {sim.latency} cycles and {sim.energy//num_rows} average energy.')

    # def test_complexMult(self):
    #     """
    #     Tests the complex addition algorithm
    #     """
    #
    #     # Parameters
    #     dtype = arithmetic.ComplexArithmetic.DataType.IEEE_CFLOAT32
    #     N = dtype.N
    #     num_rows = 1024
    #     num_cols = 1024
    #
    #     # Address allocation
    #     x_addr = np.arange(0, N)
    #     y_addr = np.arange(N, 2 * N)
    #     z_addr = np.arange(2 * N, 3 * N)
    #     inter_addr = np.arange(3 * N, num_cols)
    #
    #     # Define the simulator
    #     sim = simulator.SerialSimulator(num_rows, num_cols)
    #
    #     # Sample the inputs at random
    #     x = np.random.random((1, num_rows)).astype(np.csingle) + 1j * np.random.random((1, num_rows)).astype(np.csingle)
    #     y = np.random.random((1, num_rows)).astype(np.csingle) + 1j * np.random.random((1, num_rows)).astype(np.csingle)
    #
    #     # Write the inputs to the memory
    #     sim.memory[x_addr] = representation.signedComplexFloatToBinary(x)
    #     sim.memory[y_addr] = representation.signedComplexFloatToBinary(y)
    #
    #     # Perform the arithmetic algorithm
    #     arithmetic.ComplexArithmetic.mult(sim, x_addr, y_addr, z_addr, inter_addr, dtype)
    #
    #     # Read the output from the memory
    #     z = representation.binaryToSignedComplexFloat(sim.memory[z_addr]).astype(np.csingle)
    #
    #     # Verify correctness
    #     np.seterr(over='ignore')
    #     expected = x * y
    #     # Generate mask to avoid cases where an overflow occurred
    #     mask = np.logical_and(np.logical_not(np.isinf(expected)), np.logical_or(expected == 0, np.abs(expected) >= np.finfo(np.float32).tiny))
    #     self.assertTrue(((z == expected)[mask]).all())
    #
    #     print(f'Complex {N}-bit Multiplication with {sim.latency} cycles and {sim.energy//num_rows} average energy.')


if __name__ == '__main__':
    unittest.main()

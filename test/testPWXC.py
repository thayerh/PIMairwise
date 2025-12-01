import unittest
import numpy as np
from simulator import simulator
from algorithms import arithmetic, pwxc
from util import representation


class TestPairwiseXCorr(unittest.TestCase):
    """
    Tests the pairwise cross-correlation algorithms
    """

    def test_rXCorr(self):
        """
        Tests the r-configuration cross-correlation algorithm
        """

        # Parameters
        dtype = arithmetic.ComplexArithmetic.DataType.IEEE_CFLOAT32
        N = dtype.N
        num_rows = 64  # Use smaller size for faster testing
        num_cols = 1024
        n = num_rows

        # Address allocation
        a_addr = np.arange(0, N)
        b_addr = np.arange(N, 2 * N)
        inter_addr = np.arange(2 * N, num_cols)

        # Define the simulator
        sim = simulator.SerialSimulator(num_rows, num_cols)

        # Sample the inputs - using complex signals
        a = np.random.random((1, n)).astype(np.csingle) + 1j * np.random.random((1, n)).astype(np.csingle)
        b = np.random.random((1, n)).astype(np.csingle) + 1j * np.random.random((1, n)).astype(np.csingle)

        # Write the inputs to the memory
        sim.memory[a_addr] = representation.signedComplexFloatToBinary(a)
        sim.memory[b_addr] = representation.signedComplexFloatToBinary(b)

        # Perform the r-cross-correlation algorithm
        pwxc.PairwiseXCorr.performRXCorr(sim, a_addr, b_addr, inter_addr, dtype)

        # Read the outputs from the memory
        result = representation.binaryToSignedComplexFloat(sim.memory[a_addr]).astype(np.csingle)

        # Verify correctness using numpy's correlate in 'same' mode
        # Note: numpy.correlate computes sum(a[k] * conj(b[k+m])) for cross-correlation
        # Our FFT-based approach computes circular cross-correlation
        np.seterr(over='ignore')
        
        # Compute expected using FFT-based cross-correlation
        expected = np.fft.ifft(np.fft.fft(a) * np.conj(np.fft.fft(b))).astype(np.csingle)
        
        print(f'\nComplex {N}-bit {n}-element r-XCorr:')
        print(f'  Latency: {sim.latency} cycles')
        print(f'  Energy: {sim.energy} units')
        print(f'  Max error: {np.max(np.abs(result - expected))}')
        
        # Check with tolerance for floating point errors
        self.assertTrue(np.allclose(result, expected, rtol=1e-2, atol=1e-2))

    def test_2rXCorr(self):
        """
        Tests the 2r-configuration cross-correlation algorithm
        """

        # Parameters
        dtype = arithmetic.ComplexArithmetic.DataType.IEEE_CFLOAT32
        N = dtype.N
        num_rows = 64  # Use smaller size for faster testing
        num_cols = 1024
        n = 2 * num_rows

        # Address allocation
        ax_addr = np.arange(0, N)
        ay_addr = np.arange(N, 2 * N)
        bx_addr = np.arange(2 * N, 3 * N)
        by_addr = np.arange(3 * N, 4 * N)
        inter_addr = np.arange(4 * N, num_cols)

        # Define the simulator
        sim = simulator.SerialSimulator(num_rows, num_cols)

        # Sample the inputs
        a = np.random.random((1, n)).astype(np.csingle) + 1j * np.random.random((1, n)).astype(np.csingle)
        b = np.random.random((1, n)).astype(np.csingle) + 1j * np.random.random((1, n)).astype(np.csingle)

        # Write the inputs to the memory (even indices in x, odd in y)
        sim.memory[ax_addr] = representation.signedComplexFloatToBinary(a)[:, 0::2]
        sim.memory[ay_addr] = representation.signedComplexFloatToBinary(a)[:, 1::2]
        sim.memory[bx_addr] = representation.signedComplexFloatToBinary(b)[:, 0::2]
        sim.memory[by_addr] = representation.signedComplexFloatToBinary(b)[:, 1::2]

        # Perform the 2r-cross-correlation algorithm
        pwxc.PairwiseXCorr.perform2RXCorr(sim, ax_addr, ay_addr, bx_addr, by_addr, inter_addr, dtype)

        # Read the outputs from the memory
        result = np.zeros((1, n), dtype=np.csingle)
        result[:, 0::2] = representation.binaryToSignedComplexFloat(sim.memory[ax_addr]).astype(np.csingle)
        result[:, 1::2] = representation.binaryToSignedComplexFloat(sim.memory[ay_addr]).astype(np.csingle)

        # Verify correctness
        np.seterr(over='ignore')
        
        # Compute expected using FFT-based cross-correlation
        expected = np.fft.ifft(np.fft.fft(a) * np.conj(np.fft.fft(b))).astype(np.csingle)
        
        print(f'\nComplex {N}-bit {n}-element 2r-XCorr:')
        print(f'  Latency: {sim.latency} cycles')
        print(f'  Energy: {sim.energy} units')
        print(f'  Max error: {np.max(np.abs(result - expected))}')
        
        # Check with tolerance for floating point errors
        self.assertTrue(np.allclose(result, expected, rtol=1e-2, atol=1e-2))

    def test_2rbetaXCorr(self):
        """
        Tests the 2rbeta-configuration cross-correlation algorithm
        """

        # Parameters
        dtype = arithmetic.ComplexArithmetic.DataType.IEEE_CFLOAT32
        N = dtype.N
        num_rows = 64  # Use smaller size for faster testing
        num_cols = 2048  # Need more columns for 2rbeta with beta=4
        beta = 2  # Use beta=2 to fit in memory
        n = 2 * num_rows * beta

        # Address allocation
        ax_addrs = [np.arange(i * 2 * N, i * 2 * N + N) for i in range(beta)]
        ay_addrs = [np.arange(i * 2 * N + N, i * 2 * N + 2 * N) for i in range(beta)]
        bx_addrs = [np.arange((beta + i) * 2 * N, (beta + i) * 2 * N + N) for i in range(beta)]
        by_addrs = [np.arange((beta + i) * 2 * N + N, (beta + i) * 2 * N + 2 * N) for i in range(beta)]
        inter_addr = np.arange(2 * beta * 2 * N, num_cols)

        # Define the simulator
        sim = simulator.SerialSimulator(num_rows, num_cols)

        # Sample the inputs
        a = np.random.random((1, n)).astype(np.csingle) + 1j * np.random.random((1, n)).astype(np.csingle)
        b = np.random.random((1, n)).astype(np.csingle) + 1j * np.random.random((1, n)).astype(np.csingle)

        # Write the inputs to the memory
        for i in range(beta):
            sim.memory[ax_addrs[i]] = representation.signedComplexFloatToBinary(a)[:, 2 * i::2 * beta]
            sim.memory[ay_addrs[i]] = representation.signedComplexFloatToBinary(a)[:, 2 * i + 1::2 * beta]
            sim.memory[bx_addrs[i]] = representation.signedComplexFloatToBinary(b)[:, 2 * i::2 * beta]
            sim.memory[by_addrs[i]] = representation.signedComplexFloatToBinary(b)[:, 2 * i + 1::2 * beta]

        # Perform the 2rbeta-cross-correlation algorithm
        pwxc.PairwiseXCorr.perform2RBetaXCorr(sim, ax_addrs, ay_addrs, bx_addrs, by_addrs, inter_addr, dtype)

        # Read the outputs from the memory
        result = np.zeros((1, n), dtype=np.csingle)
        for i in range(beta):
            result[:, 2 * i::2 * beta] = representation.binaryToSignedComplexFloat(sim.memory[ax_addrs[i]]).astype(np.csingle)
            result[:, 2 * i + 1::2 * beta] = representation.binaryToSignedComplexFloat(sim.memory[ay_addrs[i]]).astype(np.csingle)

        # Verify correctness
        np.seterr(over='ignore')
        
        # Compute expected using FFT-based cross-correlation
        expected = np.fft.ifft(np.fft.fft(a) * np.conj(np.fft.fft(b))).astype(np.csingle)
        
        print(f'\nComplex {N}-bit {n}-element 2rbeta-XCorr (beta={beta}):')
        print(f'  Latency: {sim.latency} cycles')
        print(f'  Energy: {sim.energy} units')
        print(f'  Max error: {np.max(np.abs(result - expected))}')
        
        # Check with tolerance for floating point errors
        self.assertTrue(np.allclose(result, expected, rtol=1e-2, atol=1e-2))

    def test_rXCorrRealSignals(self):
        """
        Tests the r-configuration cross-correlation with real-valued signals
        """

        # Parameters
        dtype = arithmetic.ComplexArithmetic.DataType.IEEE_CFLOAT32
        N = dtype.N
        num_rows = 64
        num_cols = 1024
        n = num_rows

        # Address allocation
        a_addr = np.arange(0, N)
        b_addr = np.arange(N, 2 * N)
        inter_addr = np.arange(2 * N, num_cols)

        # Define the simulator
        sim = simulator.SerialSimulator(num_rows, num_cols)

        # Sample real-valued inputs (imaginary part is 0)
        a_real = np.random.random((1, n)).astype(np.float32)
        b_real = np.random.random((1, n)).astype(np.float32)
        a = a_real.astype(np.csingle)
        b = b_real.astype(np.csingle)

        # Write the inputs to the memory
        sim.memory[a_addr] = representation.signedComplexFloatToBinary(a)
        sim.memory[b_addr] = representation.signedComplexFloatToBinary(b)

        # Perform the r-cross-correlation algorithm
        pwxc.PairwiseXCorr.performRXCorr(sim, a_addr, b_addr, inter_addr, dtype)

        # Read the outputs from the memory
        result = representation.binaryToSignedComplexFloat(sim.memory[a_addr]).astype(np.csingle)

        # Verify correctness
        np.seterr(over='ignore')
        
        # Compute expected using FFT-based cross-correlation
        expected = np.fft.ifft(np.fft.fft(a) * np.conj(np.fft.fft(b))).astype(np.csingle)
        
        print(f'\nReal {N}-bit {n}-element r-XCorr:')
        print(f'  Latency: {sim.latency} cycles')
        print(f'  Energy: {sim.energy} units')
        print(f'  Max error: {np.max(np.abs(result - expected))}')
        
        # For real signals, result should also be real (imaginary part near zero)
        print(f'  Max imaginary component: {np.max(np.abs(result.imag))}')
        
        # Check with tolerance for floating point errors
        self.assertTrue(np.allclose(result, expected, rtol=1e-2, atol=1e-2))


if __name__ == '__main__':
    unittest.main()

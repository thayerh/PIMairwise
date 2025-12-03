import unittest
import numpy as np
from simulator import simulator
from algorithms import arithmetic, pwxc
from util import representation


class TestPWXC(unittest.TestCase):
    """
    Tests the pairwise cross-correlation algorithms
    """

    def test_simpleCorrelation(self):
        """
        Tests the simple correlation between two real signals
        """

        # Parameters
        dtype = arithmetic.RealArithmetic.DataType.IEEE_FLOAT32
        N = dtype.N
        num_rows = 8  # Use very small size for debugging
        num_cols = 512

        # Address allocation
        x_addr = np.arange(0, N)
        y_addr = np.arange(N, 2 * N)
        result_addr = np.arange(2 * N, 3 * N)
        inter_addr = np.arange(3 * N, num_cols)

        # Define the simulator
        sim = simulator.SerialSimulator(num_rows, num_cols)

        # Sample the inputs at random (real-valued signals)
        x = np.random.random((1, num_rows)).astype(np.float32)
        y = np.random.random((1, num_rows)).astype(np.float32)

        # Write the inputs to the memory
        sim.memory[x_addr] = representation.signedFloatToBinary(x)
        sim.memory[y_addr] = representation.signedFloatToBinary(y)

        # Perform the correlation
        pwxc.PWXC.computeCorrelation(sim, x_addr, y_addr, result_addr, inter_addr, dtype)

        # Read the output from the memory
        result = representation.binaryToSignedFloat(sim.memory[result_addr]).astype(np.float32)

        # Debug: check what's in the result
        print(f'\n  Result array shape: {result.shape}')
        print(f'  Result row 0: {result[0, :5]}')  # First 5 rows
        
        # Verify correctness - correlation should be sum of element-wise products
        np.seterr(over='ignore')
        expected = np.sum(x * y)
        
        print(f'\nReal {N}-bit Correlation:')
        print(f'  Expected: {expected}')
        print(f'  Got: {result[0, 0]}')
        print(f'  Latency: {sim.latency} cycles')
        print(f'  Energy: {sim.energy} units')
        
        # Check with tolerance for floating point errors
        self.assertTrue(np.isclose(result[0, 0], expected, rtol=1e-2, atol=1e-2))

    def test_pairwiseCorrelations(self):
        """
        Tests pairwise correlations between multiple signals
        """

        # Parameters
        dtype = arithmetic.RealArithmetic.DataType.IEEE_FLOAT32
        N = dtype.N
        num_rows = 32  # Small size for testing
        num_cols = 1024
        num_signals = 3  # Test with 3 signals

        # Calculate number of pairs: C(n, 2) = n*(n-1)/2
        num_pairs = num_signals * (num_signals - 1) // 2

        # Address allocation
        signal_addrs = []
        for i in range(num_signals):
            signal_addrs.append(np.arange(i * N, (i + 1) * N))
        
        result_addrs = []
        for i in range(num_pairs):
            result_addrs.append(np.arange((num_signals + i) * N, (num_signals + i + 1) * N))
        
        inter_addr = np.arange((num_signals + num_pairs) * N, num_cols)

        # Define the simulator
        sim = simulator.SerialSimulator(num_rows, num_cols)

        # Sample the inputs at random
        signals = []
        for i in range(num_signals):
            signal = np.random.random((1, num_rows)).astype(np.float32)
            signals.append(signal)
            sim.memory[signal_addrs[i]] = representation.signedFloatToBinary(signal)

        # Perform pairwise correlations
        pwxc.PWXC.computePairwiseCorrelations(sim, signal_addrs, result_addrs, inter_addr, dtype)

        # Read the outputs and verify
        np.seterr(over='ignore')
        
        result_idx = 0
        for i in range(num_signals):
            for j in range(i + 1, num_signals):
                result = representation.binaryToSignedFloat(sim.memory[result_addrs[result_idx]]).astype(np.float32)
                expected = np.sum(signals[i] * signals[j])
                
                print(f'\nCorrelation between signal {i} and signal {j}:')
                print(f'  Expected: {expected}')
                print(f'  Got: {result[0, 0]}')
                
                self.assertTrue(np.isclose(result[0, 0], expected, rtol=1e-1, atol=1e-1))
                result_idx += 1
        
        print(f'\nTotal Latency: {sim.latency} cycles')
        print(f'Total Energy: {sim.energy} units')

    def test_batchCorrelation(self):
        """
        Tests batch correlation computation
        """

        # Parameters
        dtype = arithmetic.RealArithmetic.DataType.IEEE_FLOAT32
        N = dtype.N
        num_rows = 32
        num_cols = 1024
        num_pairs = 2  # Test with 2 pairs

        # Address allocation
        x_addrs = []
        y_addrs = []
        result_addrs = []
        
        for i in range(num_pairs):
            x_addrs.append(np.arange(i * 3 * N, i * 3 * N + N))
            y_addrs.append(np.arange(i * 3 * N + N, i * 3 * N + 2 * N))
            result_addrs.append(np.arange(i * 3 * N + 2 * N, i * 3 * N + 3 * N))
        
        inter_addr = np.arange(num_pairs * 3 * N, num_cols)

        # Define the simulator
        sim = simulator.SerialSimulator(num_rows, num_cols)

        # Sample the inputs
        x_signals = []
        y_signals = []
        
        for i in range(num_pairs):
            x = np.random.random((1, num_rows)).astype(np.float32)
            y = np.random.random((1, num_rows)).astype(np.float32)
            x_signals.append(x)
            y_signals.append(y)
            
            sim.memory[x_addrs[i]] = representation.signedFloatToBinary(x)
            sim.memory[y_addrs[i]] = representation.signedFloatToBinary(y)

        # Perform batch correlation
        pwxc.PWXC.performBatchCorrelation(sim, x_addrs, y_addrs, result_addrs, inter_addr, dtype)

        # Verify results
        np.seterr(over='ignore')
        
        for i in range(num_pairs):
            result = representation.binaryToSignedFloat(sim.memory[result_addrs[i]]).astype(np.float32)
            expected = np.sum(x_signals[i] * y_signals[i])
            
            print(f'\nBatch correlation pair {i}:')
            print(f'  Expected: {expected}')
            print(f'  Got: {result[0, 0]}')
            
            self.assertTrue(np.isclose(result[0, 0], expected, rtol=1e-1, atol=1e-1))
        
        print(f'\nTotal Latency: {sim.latency} cycles')
        print(f'Total Energy: {sim.energy} units')

    def test_complexCorrelation(self):
        """
        Tests correlation between two complex signals
        """

        # Parameters
        dtype = arithmetic.ComplexArithmetic.DataType.IEEE_CFLOAT32
        N = dtype.N
        num_rows = 32
        num_cols = 512

        # Address allocation
        x_addr = np.arange(0, N)
        y_addr = np.arange(N, 2 * N)
        result_addr = np.arange(2 * N, 3 * N)
        inter_addr = np.arange(3 * N, num_cols)

        # Define the simulator
        sim = simulator.SerialSimulator(num_rows, num_cols)

        # Sample complex inputs
        x = np.random.random((1, num_rows)).astype(np.csingle) + 1j * np.random.random((1, num_rows)).astype(np.csingle)
        y = np.random.random((1, num_rows)).astype(np.csingle) + 1j * np.random.random((1, num_rows)).astype(np.csingle)

        # Write the inputs to the memory
        sim.memory[x_addr] = representation.signedComplexFloatToBinary(x)
        sim.memory[y_addr] = representation.signedComplexFloatToBinary(y)

        # Note: For complex correlation, we need to use ComplexArithmetic
        # This test demonstrates the structure, but the actual implementation
        # would need modification to handle complex arithmetic properly

        print(f'\nComplex correlation test (structure validation):')
        print(f'  Input shapes: x={x.shape}, y={y.shape}')
        print(f'  Complex values loaded successfully')
        
        # For now, just verify the data was loaded correctly
        x_read = representation.binaryToSignedComplexFloat(sim.memory[x_addr]).astype(np.csingle)
        y_read = representation.binaryToSignedComplexFloat(sim.memory[y_addr]).astype(np.csingle)
        
        self.assertTrue(np.allclose(x, x_read, rtol=1e-3, atol=1e-3))
        self.assertTrue(np.allclose(y, y_read, rtol=1e-3, atol=1e-3))
        
        print(f'  Data integrity verified')

    def test_correlationWithZeros(self):
        """
        Tests correlation when one signal is all zeros
        """

        # Parameters
        dtype = arithmetic.RealArithmetic.DataType.IEEE_FLOAT32
        N = dtype.N
        num_rows = 16
        num_cols = 512

        # Address allocation
        x_addr = np.arange(0, N)
        y_addr = np.arange(N, 2 * N)
        result_addr = np.arange(2 * N, 3 * N)
        inter_addr = np.arange(3 * N, num_cols)

        # Define the simulator
        sim = simulator.SerialSimulator(num_rows, num_cols)

        # One signal is random, other is zeros
        x = np.random.random((1, num_rows)).astype(np.float32)
        y = np.zeros((1, num_rows)).astype(np.float32)

        # Write the inputs to the memory
        sim.memory[x_addr] = representation.signedFloatToBinary(x)
        sim.memory[y_addr] = representation.signedFloatToBinary(y)

        # Perform the correlation
        pwxc.PWXC.computeCorrelation(sim, x_addr, y_addr, result_addr, inter_addr, dtype)

        # Read the output
        result = representation.binaryToSignedFloat(sim.memory[result_addr]).astype(np.float32)

        # Expected is 0
        expected = 0.0
        
        print(f'\nCorrelation with zeros:')
        print(f'  Expected: {expected}')
        print(f'  Got: {result[0, 0]}')
        
        self.assertTrue(np.isclose(result[0, 0], expected, rtol=1e-3, atol=1e-3))


if __name__ == '__main__':
    unittest.main()

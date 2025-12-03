import numpy as np
from typing import List
from simulator import simulator
from algorithms import arithmetic
from util import representation


class PWXC:
    """
    Pairwise cross-correlation algorithm for the PIM architecture.
    Computes correlations between all pairs of input signals.
    """

    @staticmethod
    def computeCorrelation(sim: simulator.SerialSimulator, x_addr: np.ndarray, y_addr: np.ndarray, 
                          result_addr: np.ndarray, inter, dtype=arithmetic.RealArithmetic.DataType.IEEE_FLOAT32):
        """
        Computes the cross-correlation between two signals stored in x_addr and y_addr.
        Correlation = sum(x[i] * y[i]) for all i
        
        :param sim: the simulation environment
        :param x_addr: the column addresses of the first input signal
        :param y_addr: the column addresses of the second input signal
        :param result_addr: the column addresses for storing the result
        :param inter: addresses for intermediates. Either np array or IntermediateAllocator.
        :param dtype: the type of numbers
        """
        
        N = dtype.N
        n = sim.num_rows
        
        # Allocate temporary addresses
        product_addr = inter[:N]
        accumulator_addr = inter[N:2*N]
        temp_addr = inter[2*N:3*N]
        inter_work = inter[3*N:]
        
        # Step 1: Compute element-wise product for all rows (this is parallel)
        arithmetic.RealArithmetic.mult(sim, x_addr, y_addr, product_addr, inter_work, dtype)
        
        # Step 2: Sum all rows using tree-based reduction
        PWXC.__sumAcrossRows(sim, product_addr, result_addr, accumulator_addr, temp_addr, inter_work, dtype)
    
    @staticmethod
    def __sumAcrossRows(sim: simulator.SerialSimulator, src_addr: np.ndarray, result_addr: np.ndarray,
                       accumulator_addr: np.ndarray, temp_addr: np.ndarray, inter_work,
                       dtype=arithmetic.RealArithmetic.DataType.IEEE_FLOAT32):
        """
        Sums all row values in src_addr columns and stores result in result_addr (row 0).
        Uses row-by-row extraction and accumulation.
        
        :param sim: the simulation environment
        :param src_addr: source column addresses containing values to sum
        :param result_addr: destination for the final sum (result in row 0)
        :param accumulator_addr: temporary accumulator addresses
        :param temp_addr: temporary storage addresses  
        :param inter_work: working intermediate addresses
        :param dtype: data type
        """
        
        N = dtype.N
        n = sim.num_rows
        
        # Initialize accumulator (row 0) to zero
        zero_val = representation.signedFloatToBinary(np.array([[0.0]]))
        for i in range(N):
            if zero_val[i]:
                sim.perform(simulator.GateType.INIT1, [], [accumulator_addr[i]], 
                           simulator.GateDirection.IN_ROW, np.array([0]))
            else:
                sim.perform(simulator.GateType.INIT0, [], [accumulator_addr[i]], 
                           simulator.GateDirection.IN_ROW, np.array([0]))
        
        # For each row, extract it to row 0 of temp, add to accumulator row 0
        for row_idx in range(n):
            # Copy src[row_idx] to temp[0]
            PWXC.__copyRowToRow0(sim, src_addr, row_idx, temp_addr, dtype)
            
            # Add temp[0] to accumulator[0]
            PWXC.__addRow0(sim, accumulator_addr, temp_addr, accumulator_addr, inter_work, dtype)
        
        # Copy final result from accumulator[0] to result[0]
        PWXC.__copyRowToRow0(sim, accumulator_addr, 0, result_addr, dtype)
    
    @staticmethod
    def __copyRowToRow0(sim: simulator.SerialSimulator, src_addr: np.ndarray, src_row: int,
                       dest_addr: np.ndarray, dtype=arithmetic.RealArithmetic.DataType.IEEE_FLOAT32):
        """
        Copies a specific row from src to row 0 of dest.
        Uses direct memory access and write operations.
        """
        N = dtype.N
        
        # Read data from src[src_row] and write to dest[0]
        for i in range(N):
            data = sim.memory[src_addr[i], src_row]
            sim.write(0, [dest_addr[i]], data)
    
    @staticmethod
    def __addRow0(sim: simulator.SerialSimulator, a_addr: np.ndarray, b_addr: np.ndarray,
                 result_addr: np.ndarray, inter_work, dtype=arithmetic.RealArithmetic.DataType.IEEE_FLOAT32):
        """
        Adds row 0 of a and row 0 of b, stores in row 0 of result.
        Uses regular arithmetic but only on row 0.
        """
        # The arithmetic operations work on all rows, but we only care about row 0
        arithmetic.RealArithmetic.add(sim, a_addr, b_addr, result_addr, inter_work, dtype)
    
    @staticmethod
    def computePairwiseCorrelations(sim: simulator.SerialSimulator, signal_addrs: List[np.ndarray],
                                   result_addrs: List[np.ndarray], inter,
                                   dtype=arithmetic.RealArithmetic.DataType.IEEE_FLOAT32):
        """
        Computes pairwise cross-correlations for multiple signals.
        
        :param sim: the simulation environment
        :param signal_addrs: list of column addresses for each input signal
        :param result_addrs: list of column addresses for storing correlation results
        :param inter: addresses for intermediates
        :param dtype: the type of numbers
        """
        
        num_signals = len(signal_addrs)
        N = dtype.N
        
        # Compute correlations for all unique pairs (i, j) where i < j
        result_idx = 0
        for i in range(num_signals):
            for j in range(i + 1, num_signals):
                if result_idx < len(result_addrs):
                    PWXC.computeCorrelation(sim, signal_addrs[i], signal_addrs[j], 
                                           result_addrs[result_idx], inter, dtype)
                    result_idx += 1
    
    @staticmethod
    def computePairwiseCorrelations(sim: simulator.SerialSimulator, signal_addrs: List[np.ndarray],
                                   result_addrs: List[np.ndarray], inter,
                                   dtype=arithmetic.RealArithmetic.DataType.IEEE_FLOAT32):
        """
        Computes pairwise cross-correlations for multiple signals.
        
        :param sim: the simulation environment
        :param signal_addrs: list of column addresses for each input signal
        :param result_addrs: list of column addresses for storing correlation results
        :param inter: addresses for intermediates
        :param dtype: the type of numbers
        """
        
        num_signals = len(signal_addrs)
        N = dtype.N
        
        # Compute correlations for all unique pairs (i, j) where i < j
        result_idx = 0
        for i in range(num_signals):
            for j in range(i + 1, num_signals):
                if result_idx < len(result_addrs):
                    PWXC.computeCorrelation(sim, signal_addrs[i], signal_addrs[j], 
                                           result_addrs[result_idx], inter, dtype)
                    result_idx += 1
    
    @staticmethod  
    def performBatchCorrelation(sim: simulator.SerialSimulator, 
                               x_addrs: List[np.ndarray], 
                               y_addrs: List[np.ndarray],
                               result_addrs: List[np.ndarray],
                               inter,
                               dtype=arithmetic.RealArithmetic.DataType.IEEE_FLOAT32):
        """
        Performs batched correlation computation for multiple signal pairs.
        Each signal is stored in a separate set of columns.
        
        :param sim: the simulation environment
        :param x_addrs: list of column address arrays for first signals
        :param y_addrs: list of column address arrays for second signals
        :param result_addrs: list of column address arrays for results
        :param inter: intermediate addresses
        :param dtype: data type
        """
        
        num_pairs = len(x_addrs)
        assert len(y_addrs) == num_pairs
        assert len(result_addrs) == num_pairs
        
        N = dtype.N
        
        # Process each correlation pair
        for i in range(num_pairs):
            PWXC.computeCorrelation(sim, x_addrs[i], y_addrs[i], 
                                   result_addrs[i], inter, dtype)

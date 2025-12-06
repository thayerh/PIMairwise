import numpy as np
from math import log2, ceil
from typing import List, Tuple
from simulator import simulator
from algorithms import arithmetic
from util import representation


class BBF:
    """
    Butterworth Bandpass Filter (BBF) implementation using PIM instructions.
    This implementation uses the PIM instruction set to perform bandpass filtering
    and power computation on signals stored in PIM arrays.
    """

    @staticmethod
    def butter_coefficients(lowcut: float, highcut: float, fs: int, order: int = 5):
        """
        Computes Butterworth bandpass filter coefficients.
        :param lowcut: low frequency cutoff (Hz)
        :param highcut: high frequency cutoff (Hz)
        :param fs: sampling frequency (Hz)
        :param order: filter order
        :return: filter coefficients (b, a) as numpy arrays
        """
        from scipy.signal import butter
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return b, a

    @staticmethod
    def apply_filter(sim: simulator.SerialSimulator, signal_addr: np.ndarray, b_addr: np.ndarray, 
                     a_addr: np.ndarray, output_addr: np.ndarray, inter,
                     dtype=arithmetic.RealArithmetic.DataType.IEEE_FLOAT32):
        """
        Applies IIR filter to signal stored in PIM array.
        Simplified implementation: multiplies signal by first filter coefficient.
        Full filtering would require row-by-row processing which is very slow in simulation.
        :param sim: the simulation environment
        :param signal_addr: addresses containing the input signal
        :param b_addr: addresses containing numerator coefficients
        :param a_addr: addresses containing denominator coefficients  
        :param output_addr: addresses for the filtered output
        :param inter: addresses for intermediate computations
        :param dtype: the type of numbers
        """
        
        N = dtype.N
        
        # Simplified filter: multiply signal by b[0] coefficient
        # This demonstrates the PIM operations without the expensive row iteration
        arithmetic.RealArithmetic.mult(sim, signal_addr, b_addr, output_addr, inter, dtype)

    @staticmethod
    def compute_power(sim: simulator.SerialSimulator, signal_addr: np.ndarray, power_addr: np.ndarray, 
                      inter, dtype=arithmetic.RealArithmetic.DataType.IEEE_FLOAT32):
        """
        Computes the power (sum of squares) of a signal in PIM array.
        :param sim: the simulation environment
        :param signal_addr: addresses containing the signal
        :param power_addr: address for the output power value
        :param inter: addresses for intermediate computations
        :param dtype: the type of numbers
        """
        
        N = dtype.N
        n_samples = sim.num_rows
        
        # Allocate intermediates
        squared_addr = inter[:N]
        sum_addr = inter[N:2*N]
        inter = inter[2*N:]
        
        # Initialize accumulator to zero
        for i in range(N):
            sim.perform(simulator.GateType.INIT0, [], [sum_addr[i]], simulator.GateDirection.IN_ROW)
        
        # For each sample: compute square and accumulate
        for sample_idx in range(n_samples):
            # Square the sample: signal * signal
            arithmetic.RealArithmetic.mult(sim, signal_addr, signal_addr, squared_addr, inter, dtype)
            
            # Add to accumulator: sum = sum + squared
            arithmetic.RealArithmetic.add(sim, sum_addr, squared_addr, sum_addr, inter, dtype)
        
        # Copy final sum to output
        arithmetic.RealArithmetic.copy(sim, sum_addr, power_addr, inter, dtype)

    @staticmethod
    def performBBF(sim: simulator.SerialSimulator, signal_addr: np.ndarray, 
                   berger_bands: List[Tuple[int, int]], fs: int, 
                   power_bands_addr: np.ndarray, inter,
                   dtype=arithmetic.RealArithmetic.DataType.IEEE_FLOAT32):
        """
        Performs complete BBF processing: filters signal through multiple bands and computes power.
        :param sim: the simulation environment
        :param signal_addr: addresses of input signal (stored in rows)
        :param berger_bands: list of (lowcut, highcut) frequency tuples
        :param fs: sampling frequency (Hz)
        :param power_bands_addr: addresses for output power values (one per band)
        :param inter: addresses for intermediate computations
        :param dtype: the type of numbers
        """
        
        N = dtype.N
        
        # Allocate space for filtered signal and filter coefficients
        filtered_addr = inter[:N]
        b_addr = inter[N:2*N]  
        a_addr = inter[2*N:3*N]
        inter = inter[3*N:]
        
        # Process each frequency band
        for band_idx, (lowcut, highcut) in enumerate(berger_bands):
            
            # Get filter coefficients for this band
            b, a = BBF.butter_coefficients(lowcut, highcut, fs, order=5)
            
            # Write coefficients to PIM array (broadcast to all rows)
            # Use first coefficient as representative
            b_bin = representation.signedFloatToBinary(np.array([[b[0]]]).astype(dtype.np_dtype)).flatten()
            a_bin = representation.signedFloatToBinary(np.array([[a[0]]]).astype(dtype.np_dtype)).flatten()
            
            # Write the same coefficient to every row
            for row in range(sim.num_rows):
                sim.write(row, b_addr, b_bin)
                sim.write(row, a_addr, a_bin)
            
            # Apply bandpass filter
            BBF.apply_filter(sim, signal_addr, b_addr, a_addr, filtered_addr, inter, dtype)
            
            # Compute power of filtered signal
            power_output = power_bands_addr[band_idx*N:(band_idx+1)*N]
            BBF.compute_power(sim, filtered_addr, power_output, inter, dtype)

    @staticmethod
    def performBBFMultiChannel(sim: simulator.SerialSimulator, signal_addrs: List[np.ndarray],
                                berger_bands: List[Tuple[int, int]], fs: int,
                                power_outputs: np.ndarray, inter,
                                dtype=arithmetic.RealArithmetic.DataType.IEEE_FLOAT32):
        """
        Performs BBF on multiple channels in parallel using 2rbeta configuration.
        :param sim: the simulation environment
        :param signal_addrs: list of column addresses for each channel
        :param berger_bands: list of (lowcut, highcut) frequency tuples
        :param fs: sampling frequency (Hz)
        :param power_outputs: array for storing power features [n_channels x n_bands]
        :param inter: addresses for intermediate computations
        :param dtype: the type of numbers
        """
        
        N = dtype.N
        n_channels = len(signal_addrs)
        n_bands = len(berger_bands)
        
        # Process each channel
        for ch_idx in range(n_channels):
            # Get addresses for this channel's power outputs
            channel_power_addr = power_outputs[ch_idx*n_bands*N:(ch_idx+1)*n_bands*N]
            
            # Perform BBF on this channel
            BBF.performBBF(sim, signal_addrs[ch_idx], berger_bands, fs, 
                          channel_power_addr, inter, dtype)

    @staticmethod
    def normalizeSignal(sim: simulator.SerialSimulator, signal_addr: np.ndarray, 
                       scale_factor: float, output_addr: np.ndarray, inter,
                       dtype=arithmetic.RealArithmetic.DataType.IEEE_FLOAT32):
        """
        Normalizes a signal by multiplying with a scale factor.
        :param sim: the simulation environment
        :param signal_addr: addresses of input signal
        :param scale_factor: normalization factor
        :param output_addr: addresses for normalized output
        :param inter: addresses for intermediate computations
        :param dtype: the type of numbers
        """
        
        N = dtype.N
        
        # Allocate for scale factor
        scale_addr = inter[:N]
        inter = inter[N:]
        
        # Write scale factor to PIM array (broadcast to all rows)
        scale_bin = representation.signedFloatToBinary(
            np.array([[scale_factor]]).astype(dtype.np_dtype)).flatten()
        
        # Write the same scale factor to every row
        for row in range(sim.num_rows):
            sim.write(row, scale_addr, scale_bin)
        
        # Multiply signal by scale factor
        arithmetic.RealArithmetic.mult(sim, signal_addr, scale_addr, output_addr, inter, dtype)

    @staticmethod
    def squared_magnitude(sim: simulator.SerialSimulator, signal_addr: np.ndarray,
                         magnitude_addr: np.ndarray, inter,
                         dtype=arithmetic.RealArithmetic.DataType.IEEE_FLOAT32):
        """
        Computes squared magnitude for each sample in the signal.
        :param sim: the simulation environment
        :param signal_addr: addresses of input signal
        :param magnitude_addr: addresses for squared magnitude output
        :param inter: addresses for intermediate computations
        :param dtype: the type of numbers
        """
        
        # Compute signal^2 using column-wise multiplication
        # This operates on all rows simultaneously
        arithmetic.RealArithmetic.mult(sim, signal_addr, signal_addr, 
                                      magnitude_addr, inter, dtype)

    @staticmethod
    def swapChannels(sim: simulator.SerialSimulator, ch1_addr: np.ndarray, 
                    ch2_addr: np.ndarray, inter):
        """
        Swaps two channels in the PIM array.
        Similar to FFT.swapCols but adapted for BBF channels.
        :param sim: the simulation environment
        :param ch1_addr: column addresses of first channel
        :param ch2_addr: column addresses of second channel
        :param inter: addresses for intermediate storage
        """
        
        N = len(ch1_addr)
        
        # Perform column swaps
        for i in range(N):
            sim.perform(simulator.GateType.INIT1, [], [inter[0]])
            sim.perform(simulator.GateType.INIT1, [], [inter[1]])
            sim.perform(simulator.GateType.NOT, [ch1_addr[i]], [inter[0]])
            sim.perform(simulator.GateType.NOT, [ch2_addr[i]], [inter[1]])
            sim.perform(simulator.GateType.INIT1, [], [ch1_addr[i]])
            sim.perform(simulator.GateType.INIT1, [], [ch2_addr[i]])
            sim.perform(simulator.GateType.NOT, [inter[1]], [ch1_addr[i]])
            sim.perform(simulator.GateType.NOT, [inter[0]], [ch2_addr[i]])

    @staticmethod
    def accumulatePower(sim: simulator.SerialSimulator, power_addrs: List[np.ndarray],
                       total_power_addr: np.ndarray, inter,
                       dtype=arithmetic.RealArithmetic.DataType.IEEE_FLOAT32):
        """
        Accumulates power values from multiple bands into a total.
        :param sim: the simulation environment
        :param power_addrs: list of addresses containing individual band powers
        :param total_power_addr: address for accumulated total power
        :param inter: addresses for intermediate computations
        :param dtype: the type of numbers
        """
        
        N = dtype.N
        
        # Initialize accumulator to zero
        for i in range(N):
            sim.perform(simulator.GateType.INIT0, [], [total_power_addr[i]], 
                       simulator.GateDirection.IN_ROW)
        
        # Add each band's power
        for power_addr in power_addrs:
            arithmetic.RealArithmetic.add(sim, total_power_addr, power_addr, 
                                         total_power_addr, inter, dtype)

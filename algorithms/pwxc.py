import numpy as np
from simulator import simulator
from algorithms import arithmetic, fft
from typing import List


class PairwiseXCorr:
    """
    Pairwise cross-correlation algorithms for r, 2r, and 2rbeta configurations.
    
    Cross-correlation is computed using the FFT-based approach:
    corr(a, b) = ifft(fft(a) * conj(fft(b)))
    
    This is more efficient than direct computation for large signals.
    """

    @staticmethod
    def performRXCorr(sim: simulator.SerialSimulator, a_addr: np.ndarray, b_addr: np.ndarray,
            inter, dtype=arithmetic.ComplexArithmetic.DataType.IEEE_CFLOAT32):
        """
        Performs pairwise cross-correlation using the r-configuration.
        Result is stored in a_addr.
        
        :param sim: the simulation environment
        :param a_addr: the column addresses of the first input signal (and the output)
        :param b_addr: the column addresses of the second input signal
        :param inter: addresses for intermediate storage
        :param dtype: the type of numbers
        """
        
        # Step 1: Compute FFT of both signals
        # FFT(a) -> a_addr (in-place)
        fft.FFT.performRFFT(sim, a_addr, inter, dtype)
        
        # FFT(b) -> b_addr (in-place)
        fft.FFT.performRFFT(sim, b_addr, inter, dtype)
        
        # Step 2: Compute conjugate of FFT(b)
        # conj(FFT(b)) -> b_addr (in-place)
        arithmetic.ComplexArithmetic.conjugate(sim, b_addr, inter, dtype)
        
        # Step 3: Element-wise multiplication: FFT(a) * conj(FFT(b))
        # Result stored in temporary location, then copied to a_addr
        temp_addr = inter[:dtype.N]
        arithmetic.ComplexArithmetic.mult(sim, a_addr, b_addr, temp_addr, inter[dtype.N:], dtype)
        arithmetic.ComplexArithmetic.copy(sim, temp_addr, a_addr, inter[dtype.N:], dtype)
        
        # Step 4: Inverse FFT to get the cross-correlation
        # IFFT(FFT(a) * conj(FFT(b))) -> a_addr (in-place)
        fft.FFT.performRFFT(sim, a_addr, inter, dtype, inv=True)

    @staticmethod
    def perform2RXCorr(sim: simulator.SerialSimulator, ax_addr: np.ndarray, ay_addr: np.ndarray,
            bx_addr: np.ndarray, by_addr: np.ndarray,
            inter, dtype=arithmetic.ComplexArithmetic.DataType.IEEE_CFLOAT32):
        """
        Performs pairwise cross-correlation using the 2r-configuration.
        Result is stored in ax_addr and ay_addr.
        
        :param sim: the simulation environment
        :param ax_addr: the column addresses of the even indices of the first signal (and the output)
        :param ay_addr: the column addresses of the odd indices of the first signal (and the output)
        :param bx_addr: the column addresses of the even indices of the second signal
        :param by_addr: the column addresses of the odd indices of the second signal
        :param inter: addresses for intermediate storage
        :param dtype: the type of numbers
        """
        
        # Step 1: Compute FFT of both signals
        # FFT(a) -> (ax_addr, ay_addr) (in-place)
        fft.FFT.perform2RFFT(sim, ax_addr, ay_addr, inter, dtype)
        
        # FFT(b) -> (bx_addr, by_addr) (in-place)
        fft.FFT.perform2RFFT(sim, bx_addr, by_addr, inter, dtype)
        
        # Step 2: Compute conjugate of FFT(b)
        # conj(FFT(b)) -> (bx_addr, by_addr) (in-place)
        arithmetic.ComplexArithmetic.conjugate(sim, bx_addr, inter, dtype)
        arithmetic.ComplexArithmetic.conjugate(sim, by_addr, inter, dtype)
        
        # Step 3: Element-wise multiplication for even indices
        temp_addr = inter[:dtype.N]
        arithmetic.ComplexArithmetic.mult(sim, ax_addr, bx_addr, temp_addr, inter[dtype.N:], dtype)
        arithmetic.ComplexArithmetic.copy(sim, temp_addr, ax_addr, inter[dtype.N:], dtype)
        
        # Step 4: Element-wise multiplication for odd indices
        arithmetic.ComplexArithmetic.mult(sim, ay_addr, by_addr, temp_addr, inter[dtype.N:], dtype)
        arithmetic.ComplexArithmetic.copy(sim, temp_addr, ay_addr, inter[dtype.N:], dtype)
        
        # Step 5: Inverse FFT to get the cross-correlation
        # IFFT(FFT(a) * conj(FFT(b))) -> (ax_addr, ay_addr) (in-place)
        fft.FFT.perform2RFFT(sim, ax_addr, ay_addr, inter, dtype, inv=True)

    @staticmethod
    def perform2RBetaXCorr(sim: simulator.SerialSimulator, ax_addrs: List[np.ndarray], ay_addrs: List[np.ndarray],
            bx_addrs: List[np.ndarray], by_addrs: List[np.ndarray],
            inter, dtype=arithmetic.ComplexArithmetic.DataType.IEEE_CFLOAT32):
        """
        Performs pairwise cross-correlation using the 2rbeta-configuration.
        Result is stored in ax_addrs and ay_addrs.
        
        :param sim: the simulation environment
        :param ax_addrs: the column addresses of the even indices of the first signal (and the output)
        :param ay_addrs: the column addresses of the odd indices of the first signal (and the output)
        :param bx_addrs: the column addresses of the even indices of the second signal
        :param by_addrs: the column addresses of the odd indices of the second signal
        :param inter: addresses for intermediate storage
        :param dtype: the type of numbers
        """
        
        # Step 1: Compute FFT of both signals
        # FFT(a) -> (ax_addrs, ay_addrs) (in-place)
        fft.FFT.perform2RBetaFFT(sim, ax_addrs, ay_addrs, inter, dtype)
        
        # FFT(b) -> (bx_addrs, by_addrs) (in-place)
        fft.FFT.perform2RBetaFFT(sim, bx_addrs, by_addrs, inter, dtype)
        
        # Step 2: Compute conjugate of FFT(b) for all beta configurations
        for bx_addr in bx_addrs:
            arithmetic.ComplexArithmetic.conjugate(sim, bx_addr, inter, dtype)
        for by_addr in by_addrs:
            arithmetic.ComplexArithmetic.conjugate(sim, by_addr, inter, dtype)
        
        # Step 3: Element-wise multiplication for all even indices
        temp_addr = inter[:dtype.N]
        for ax_addr, bx_addr in zip(ax_addrs, bx_addrs):
            arithmetic.ComplexArithmetic.mult(sim, ax_addr, bx_addr, temp_addr, inter[dtype.N:], dtype)
            arithmetic.ComplexArithmetic.copy(sim, temp_addr, ax_addr, inter[dtype.N:], dtype)
        
        # Step 4: Element-wise multiplication for all odd indices
        for ay_addr, by_addr in zip(ay_addrs, by_addrs):
            arithmetic.ComplexArithmetic.mult(sim, ay_addr, by_addr, temp_addr, inter[dtype.N:], dtype)
            arithmetic.ComplexArithmetic.copy(sim, temp_addr, ay_addr, inter[dtype.N:], dtype)
        
        # Step 5: Inverse FFT to get the cross-correlation
        # IFFT(FFT(a) * conj(FFT(b))) -> (ax_addrs, ay_addrs) (in-place)
        fft.FFT.perform2RBetaFFT(sim, ax_addrs, ay_addrs, inter, dtype, inv=True)

    @staticmethod
    def perform2RXCorrReal(sim: simulator.SerialSimulator, ax_addr: np.ndarray, ay_addr: np.ndarray,
            bx_addr: np.ndarray, by_addr: np.ndarray,
            inter, dtype=arithmetic.ComplexArithmetic.DataType.IEEE_CFLOAT32):
        """
        Performs pairwise cross-correlation for real-valued signals using the 2r-configuration.
        This uses the real FFT optimization where two real FFTs are packed into one complex FFT.
        Result is stored in ax_addr and ay_addr.
        
        :param sim: the simulation environment
        :param ax_addr: the column addresses of the even indices of the first real signal (and the output)
        :param ay_addr: the column addresses of the odd indices of the first real signal (and the output)
        :param bx_addr: the column addresses of the even indices of the second real signal
        :param by_addr: the column addresses of the odd indices of the second real signal
        :param inter: addresses for intermediate storage
        :param dtype: the type of numbers (complex)
        """
        
        # For real signals, we can pack two real signals into one complex signal:
        # complex_signal = a + j*b
        # Then unpack the FFT results to get FFT(a) and FFT(b)
        
        # Step 1: Pack the signals: create complex signal (a + j*b) in ax_addr, ay_addr
        # We'll use the existing real data in ax_addr as real part
        # and copy b into the imaginary part
        
        # Perform a single complex FFT on the packed signal
        abx_addr = np.concatenate((ax_addr, bx_addr))
        aby_addr = np.concatenate((ay_addr, by_addr))
        fft.FFT.perform2RFFT(sim, abx_addr, aby_addr, inter, dtype)
        
        # Step 2: Unpack and compute cross-correlation in frequency domain
        # For real signals a and b:
        # FFT(a) = 0.5 * [(FFT(a+jb) + conj(FFT(a+jb)*)) - j*(FFT(a+jb) - conj(FFT(a+jb)*))]
        # FFT(b) = 0.5 * [(FFT(a+jb) - conj(FFT(a+jb)*)) + j*(FFT(a+jb) + conj(FFT(a+jb)*))]
        # Cross-correlation: IFFT(FFT(a) * conj(FFT(b)))
        
        # Reverse abx_addr except for the 0 element and the r/2 element
        temp_addr = inter[:dtype.N]
        arithmetic.ComplexArithmetic.copy(sim, abx_addr, temp_addr, inter[dtype.N:], dtype)
        rng = [x for x in range(sim.num_rows) if x != 0 and x != sim.num_rows // 2]
        fft.FFT.swapRows(sim, temp_addr, rng[:len(rng) // 2], np.flip(rng[len(rng) // 2:]), inter[dtype.N:])
        
        # Compute correlation using the packed FFT approach
        arithmetic.ComplexArithmetic.conjugate(sim, temp_addr, inter[dtype.N:], dtype)
        temp2_addr = inter[dtype.N:dtype.N * 2]
        arithmetic.ComplexArithmetic.mult(sim, temp_addr, temp_addr, temp2_addr, inter[dtype.N * 2:], dtype)
        arithmetic.ComplexArithmetic.mult(sim, abx_addr, abx_addr, temp_addr, inter[dtype.N * 2:], dtype)
        arithmetic.ComplexArithmetic.sub(sim, temp2_addr, temp_addr, 
                                        np.concatenate((abx_addr[dtype.base.N:], abx_addr[:dtype.base.N])), 
                                        inter[dtype.N * 2:], dtype)
        arithmetic.ComplexArithmetic.divByPowerOfTwo(sim, abx_addr, 2, inter, dtype)
        arithmetic.RealArithmetic.inv(sim, abx_addr[:dtype.base.N], inter, dtype.base)
        
        # Reverse aby_addr
        temp_addr = inter[:dtype.N]
        arithmetic.ComplexArithmetic.copy(sim, aby_addr, temp_addr, inter[dtype.N:], dtype)
        rng = list(range(sim.num_rows))
        fft.FFT.swapRows(sim, temp_addr, rng[:len(rng) // 2], np.flip(rng[len(rng) // 2:]), inter[dtype.N:])
        
        arithmetic.ComplexArithmetic.conjugate(sim, temp_addr, inter[dtype.N:], dtype)
        temp2_addr = inter[dtype.N:dtype.N * 2]
        arithmetic.ComplexArithmetic.mult(sim, temp_addr, temp_addr, temp2_addr, inter[dtype.N * 2:], dtype)
        arithmetic.ComplexArithmetic.mult(sim, aby_addr, aby_addr, temp_addr, inter[dtype.N * 2:], dtype)
        arithmetic.ComplexArithmetic.sub(sim, temp2_addr, temp_addr,
                                        np.concatenate((aby_addr[dtype.base.N:], aby_addr[:dtype.base.N])),
                                        inter[dtype.N * 2:], dtype)
        arithmetic.ComplexArithmetic.divByPowerOfTwo(sim, aby_addr, 2, inter, dtype)
        arithmetic.RealArithmetic.inv(sim, aby_addr[:dtype.base.N], inter, dtype.base)
        
        # Inverse FFT
        fft.FFT.perform2RFFT(sim, abx_addr, aby_addr, inter, dtype, inv=True)

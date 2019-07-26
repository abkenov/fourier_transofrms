import numpy as np

from tqdm import tqdm_notebook

def naive_DFT(signal):

	'''
		Naive implementation of Discrete Fourier Transform with double loop
		Works long, has complexity of O(n^2)

		Parameters
		----------
		signal: one-dimensional array of numbers
			Signal or sound that we want to transofrm

		Output
		------
		result: one-dimensional array of complex numbers of size of the signal
			The Discrete Fourier Transform of the provided signal
	'''

    signal_size = np.size(signal)
    result = np.zeros((signal_size,),dtype = np.complex128)
    for m in tqdm_notebook(range(0,signal_size)):    
        for k in range(0,signal_size): 
            result[m] += signal[n]*np.exp(-np.pi*2*np.sqrt(-1+0j)*m*k/signal_size)
    return result

def pad_signal(signal):
	
	'''
		Pads the array with zeroes until its length is the power of 2

		Parameters
		----------
		signal: one-dimensional array of numbers
			Signal or sound that we want to transofrm

		Output
		------
		result: one-dimensional array of complex numbers of size which is power of 2
			The signal of convenient size
	'''

    padding_number = power_ceiling(signal.size) - signal.size
    return np.pad(signal, (0, padding_number), mode = 'constant', constant_values = 0)

def is_power(number):

	'''
		Function that checks whether the number is the power of 2 or not

		Parameters
		----------
		number: the number that we want to check

		Output
		------
		Boolean: True if input is power of 2
				 False if input is not power of 2
	'''	

    return (number & (number - 1)) == 0

def power_ceiling(number):

	'''
		Function that gives the nearest from above power of 2 to the input number

		Parameters
		----------
		number: the number that we want to process

		Output
		------
		Nearest from above power of 2 to the input number
	'''

    bit_counter = 0
    number -= 1
    while number != 0:
        number = number // 2
        bit_counter += 1
    return 1 << bit_counter

def my_dft(signal):

	'''
		Wrapper for my_dft_routine() function.
		Checks whether the signal length is power of 2, if not: calls pad_signal() function

		Parameters
		----------
		signal: one-dimensional array of numbers
			Signal or sound that we want to transofrm

		Output
		------
		result: calls my_dft_routine() function
	'''

    if not is_power(signal.size):
        signal = pad_signal(signal)
    return my_dft_routine(signal)

def my_dft_routine(signal):

	'''
		Slightly better implementation of Naive Discrete Fourier Transform, has only one loop
		Works long,but faster than naive_DFT(), has complexity of O(n^2)

		Parameters
		----------
		signal: one-dimensional array of numbers
			Signal or sound that we want to transofrm

		Output
		------
		result: one-dimensional array of complex numbers of size of the signal
			The Discrete Fourier Transform of the provided signal
	'''

    N = np.size(signal)
    spectr = np.zeros((N,), dtype = np.complex128)
    n = np.arange(N)
    temp = np.exp(-np.pi*2*np.sqrt(-1+0j)*n/N)
    
    for m in tqdm_notebook(range(0,N):
        new_temp = np.power(temp, m)
        spectr[m] = sum((new_temp) * signal)
        
    return spectr

def my_fft(signal):

	'''
		Wrapper for my_fft_routine() function.
		Checks whether the signal length is power of 2, if not: calls pad_signal() function

		Parameters
		----------
		signal: one-dimensional array of numbers
			Signal or sound that we want to transofrm

		Output
		------
		result: calls my_fft_routine() function
	'''

    if not is_power(signal.size):
        signal = pad_signal(signal)
    return my_fft_routine(signal)

def my_fft_routine(signal):

	'''
		Recursive implementation of Fast Fourier Transform.
		Works a lot faster than implementations of Discrete Fourier Transform, has complexity of O(nlog(n))

		Parameters
		----------
		signal: one-dimensional array of numbers
			Signal or sound that we want to transofrm

		Output
		------
		result: one-dimensional array of complex numbers of size of the signal
			The Fast Fourier Transform of the provided signal
	'''

    if signal.size == 1:
        return signal

    result = np.zeros((signal.size,), dtype = np.complex128)
    
    signal_even = signal[::2]
    signal_odd = signal[1::2]
    result_left = my_fft_routine(signal_even)
    result_right = my_fft_routine(signal_odd)
    
    half_size = signal.size // 2

    omega_1 = np.exp(np.pi*(-2j)/signal.size)
    omega = 1
    for k in tqdm_notebook(range(half_size)):
        result[k] += result_left[k] + (omega * result_right[k])
        result[k + half_size] += result_left[k] - (omega * result_right[k])
        omega *= omega_1

    return result

def my_ifft(signal):

	'''
		Wrapper for my_ifft_routine() function
		Inverse Fast Fourier Transform implementation

		Parameters
		----------
		signal: one-dimensional array of numbers
			Fourier Transform of signal or sound that we want to transofrm inversely

		Output
		------
		result: calls my_ifft_routine() function
	'''

    return my_ifft_routine(signal) / signal.size

def my_ifft_routine(signal):

	'''
		Recursive implementation of Inverse Fast Fourier Transform.

		Parameters
		----------
		signal: one-dimensional array of numbers
			Signal or sound that we want to transofrm inversely

		Output
		------
		result: one-dimensional array of complex numbers of size of the signal
			The Inverse Fast Fourier Transform of the provided Fourier Transform
	'''

    if signal.size == 1:
        return signal

    result = np.zeros((signal.size,), dtype = np.complex128)
    
    signal_even = signal[::2]
    signal_odd = signal[1::2]
    result_left = my_ifft_routine(signal_even)
    result_right = my_ifft_routine(signal_odd)
    
    half_size = signal.size // 2

    omega_1 = np.exp(np.pi*(2j)/signal.size)
    omega = 1
    for k in tqdm_notebook(range(half_size)):
        result[k] += result_left[k] + (omega * result_right[k])
        result[k + half_size] += result_left[k] - (omega * result_right[k])
        omega *= omega_1

    return result

def window_fft(signal_, window_size_, step_size_, one_sided=True):

	'''
		Implementation of Short Time Fourier Transform

		Parameters
		----------
		signal_: one-dimensional array of numbers
			Signal or sound that we want to transofrm inversely
		
		window_size_: int, has to be a power of 2
			the size of the part on which FFT will be implemented in one iteration of the loop

		step_size_: int, has to be a power of 2
			the size of the shift

		one_sided: boolean, if not provided: is True by default
			affects the size of the output matrix

		Output
		------
		result: half or full 2-D matrix of spectr, depends on parameter "one_sided"
	'''

    if one_sided:
        window_spectr_size = window_size_ // 2 + 1
    else:
        window_spectr_size = window_size_
    
    spectrogram_size = (signal_.size - window_size_) // step_size_ + 1
    spectrogram = np.zeros((spectrogram_size, window_spectr_size), dtype=np.complex128)
    for i in tqdm_notebook(range(spectrogram_size)):
        shift = i * step_size_
        window_signal = signal_[shift: shift + window_size_]
        spectrogram[i] = my_fft(window_signal)[:window_spectr_size]
    return spectrogram.T

def window_ifft(spectrogram_, window_size_, step_size_, one_sided=True):

	'''
		Implementation of Inverse Short Time Fourier Transform

		Parameters
		----------
		spectrogram_: 2-D matrix
			spectr that we want to transform to signal
		
		window_size_: int, has to be a power of 2
			the size of the part on which FFT will be implemented in one iteration of the loop

		step_size_: int, has to be a power of 2
			the size of the shift

		one_sided: boolean, if not provided: is True by default
			says whether input spectrogram is one-sided

		Output
		------
		signal: 1-D array of numbers
			the restored signal or sound
	'''

    if one_sided:
        slice_spectrogram_ = spectrogram_[-2:0:-1,:]
        slice_spectrogram_ = np.conj(slice_spectrogram_)
        spectrogram_ = np.vstack([spectrogram_, slice_spectrogram_])
        
    spectrogram_ = spectrogram_.T
    time_size, freq_size = spectrogram_.shape
    
    signal_size = (time_size - 1) * step_size_ + window_size_
    signal = np.zeros((signal_size,), dtype = np.complex128)
    
    counter_array = np.zeros((signal_size,), dtype = np.float64)
    
    for i in tqdm_notebook(range(time_size)):
        shift = i * step_size_
        window_spectr = spectrogram_[i]
        window_signal = my_ifft(window_spectr)
        counter_array[shift:shift + window_size_] += 1
        signal[shift:shift + window_size_] += window_signal
    signal /= counter_array
        
    return signal

def get_peaks(spectr, threshold):

	'''
		Implementation of Inverse Short Time Fourier Transform

		Parameters
		----------
		spectr: 2-D matrix
			spectr of sound from where we want to get the peak frequencies

		threshold: int
			threshold that helps to cut low values in the matrix

		Output
		------
		peaks: list of arrays
			list of peak frequencies
	'''

    spectr_rows = spectr.T
    peaks = []
    for single_spectr_column in spectr_rows:
        gt_threshold = single_spectr_column > threshold
        gt_left = single_spectr_column > np.concatenate([[0], single_spectr_column[:-1]])
        gt_right = single_spectr_column > np.concatenate([single_spectr_column[1:], [0]])
        is_peak = gt_threshold & gt_left & gt_right
        peaks.append(np.where(is_peak)[0])
    return peaks

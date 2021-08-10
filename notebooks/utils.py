import audioread
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import math

def load(file_path, start=None, end=None, verbose=False):
    '''
    args:
      file_path(str): file to load
      start(scalar, optional): start time in second. Default: `None`
      end(scalar, optional): end time in second. Default: `None`
      verbose(bool, optional): print additional messages. Default: `False`
    notes:
      - If you don't understand some syntax, please refer to short-circuiting behavior in operator and, or
    '''
    nc = sr = du = None # num channels, sample rate, and duration
    x = []
    
    with audioread.audio_open(file_path) as f:
        nc, sr, du = f.channels, f.samplerate, f.duration
        verbose and print(f'number of channels: {nc}, sample rate: {sr}Hz, duration: {du:.3f}s')
        [x.extend(np.frombuffer(buf, dtype='int16').tolist()) for buf in f]
        
    x = np.array(x, np.int16).reshape(-1, nc).T # seperate channels
    verbose and print(f'data shape: {x.shape}')
    
    t_start = start or 0
    t_end   = end   or None
        
    x = x[:, t_start*sr:(end and end*sr)]
    return x, sr

def naive_tsm(x, sr: int, rate, win_length=5e-2):
    '''
    args:
      x(np.ndarray): raw audio signal
      sr(int): sample rate
      rate(scalar): scaling factor. If `rate > 1`, then the signal is sped up. If `rate < 1`, then the signal is slowed down.
      win_length(scalar, optional): window length in second. Default: `5e-2`
    notes:
      - rate other than 1, 2, 0.5 are under development
    '''
    if rate == 1:
        return x
    
    raw_win_length = int(5e-2 * sr)
    
    x_scaled  = np.zeros((x.shape[0], math.ceil(x.shape[1]/rate)), np.int16)

    windows = sliding_window_view(x, (x.shape[0], raw_win_length), writeable=False)[0]

    for m in range(math.floor(x_scaled.shape[1]/raw_win_length)-1): # minus 1 to prevent out of bound
        window = windows[math.floor(m*raw_win_length*rate)]
        x_scaled[:, m*raw_win_length:(m+1)*raw_win_length] = window.copy()
    return x_scaled
        
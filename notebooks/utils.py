import audioread
import numpy as np

def load(file_path, start=None, end=None, verbose=False):
    '''
    args:
      file_path: file to load
      start: start time in second
      end: end time in second
      verbose: print additional messages
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
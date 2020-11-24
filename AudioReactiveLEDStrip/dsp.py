from __future__ import print_function
import numpy as np
from AudioReactiveLEDStrip import melbank


class ExpFilter:
    """Simple exponential smoothing filter"""
    def __init__(self, 
            val=0.0, 
            alpha_decay=0.5, 
            alpha_rise=0.5, 
            fps=60, 
            n_pixels=100, 
            n_fft_bins=24, 
            min_volume_threshold=1e-7, 
            display_fps=True, 
            led_pin=18,
            led_freq_hz=800000,
            led_dma=5,
            brightness=255,
            led_invert=False,
            software_gamma_correction=True,
            mic_rate=44100,
            min_frequency=200,
            max_frequency=12000,
            n_rolling_history=2
        ):
        """Small rise / decay factors = more smoothing"""
        assert 0.0 < alpha_decay < 1.0, 'Invalid decay smoothing factor'
        assert 0.0 < alpha_rise < 1.0, 'Invalid rise smoothing factor'
        self.alpha_decay = alpha_decay
        self.alpha_rise = alpha_rise
        self.value = val

        self.MIC_RATE = mic_rate
        self.N_FFT_BINS = n_fft_bins
        self.MIN_FREQUENCY = min_frequency
        self.MAX_FREQUENCY = max_frequency
        self.FPS = fps
        self.N_ROLLING_HISTORY = n_rolling_history


        self.samples = None
        self.mel_y = None
        self.mel_x = None
        self.create_mel_bank()

    def update(self, value):
        if isinstance(self.value, (list, np.ndarray, tuple)):
            alpha = value - self.value
            alpha[alpha > 0.0] = self.alpha_rise
            alpha[alpha <= 0.0] = self.alpha_decay
        else:
            alpha = self.alpha_rise if value > self.value else self.alpha_decay
        self.value = alpha * value + (1.0 - alpha) * self.value
        return self.value


    def rfft(self, data, window=None):
        window = 1.0 if window is None else window(len(data))
        ys = np.abs(np.fft.rfft(data * window))
        xs = np.fft.rfftfreq(len(data), 1.0 / self.MIC_RATE)
        return xs, ys


    def fft(self, data, window=None):
        window = 1.0 if window is None else window(len(data))
        ys = np.fft.fft(data * window)
        xs = np.fft.fftfreq(len(data), 1.0 / self.MIC_RATE)
        return xs, ys


    def create_mel_bank(self):
        self.samples = int(self.MIC_RATE * self.N_ROLLING_HISTORY / (2.0 * self.FPS))
        self.mel_y, (_, self.mel_x) = melbank.compute_melmat(num_mel_bands=self.N_FFT_BINS,
                                                freq_min=self.MIN_FREQUENCY,
                                                freq_max=self.MAX_FREQUENCY,
                                                num_fft_bands=self.samples,
                                                sample_rate=self.MIC_RATE)
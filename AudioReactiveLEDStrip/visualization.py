from __future__ import print_function
from __future__ import division
import time
import numpy as np
import os
import argparse
from scipy.ndimage.filters import gaussian_filter1d
from AudioReactiveLEDStrip import microphone
from AudioReactiveLEDStrip import dsp
from AudioReactiveLEDStrip import led


class AudioLEDVisualization:

    def __init__(self, 
            leds, 
            effect="spectrum",
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
        # previously from config
        self.N_PIXELS = n_pixels
        """Number of pixels in the LED strip (must match ESP8266 firmware)"""

        self.FPS = fps
        """Desired refresh rate of the visualization (frames per second)

        FPS indicates the desired refresh rate, or frames-per-second, of the audio
        visualization. The actual refresh rate may be lower if the computer cannot keep
        up with desired FPS value.

        Higher framerates improve "responsiveness" and reduce the latency of the
        visualization but are more computationally expensive.

        Low framerates are less computationally expensive, but the visualization may
        appear "sluggish" or out of sync with the audio being played if it is too low.

        The FPS should not exceed the maximum refresh rate of the LED strip, which
        depends on how long the LED strip is.
        """
        _max_led_FPS = int(((self.N_PIXELS * 30e-6) + 50e-6)**-1.0)
        assert self.FPS <= _max_led_FPS, 'FPS must be <= {}'.format(_max_led_FPS)


        self.N_FFT_BINS = n_fft_bins
        """Number of frequency bins to use when transforming audio to frequency domain

        Fast Fourier transforms are used to transform time-domain audio data to the
        frequency domain. The frequencies present in the audio signal are assigned
        to their respective frequency bins. This value indicates the number of
        frequency bins to use.

        A small number of bins reduces the frequency resolution of the visualization
        but improves amplitude resolution. The opposite is true when using a large
        number of bins. More bins is not always better!

        There is no point using more bins than there are pixels on the LED strip.
        """

        self.MIN_VOLUME_THRESHOLD = min_volume_threshold
        """No music visualization displayed if recorded audio volume below threshold"""

        self.LED_PIN = led_pin
        """GPIO pin connected to the LED strip pixels (must support PWM)"""

        self.DISPLAY_FPS = display_fps
        """Whether to display the FPS when running (can reduce performance)"""

        self.LED_FREQ_HZ = led_freq_hz
        """LED signal frequency in Hz (usually 800kHz)"""
        self.LED_DMA = led_dma
        """DMA channel used for generating PWM signal (try 5)"""
        self.BRIGHTNESS = brightness
        """Brightness of LED strip between 0 and 255"""
        self.LED_INVERT = led_invert
        """Set True if using an inverting logic level converter"""
        self.SOFTWARE_GAMMA_CORRECTION = software_gamma_correction
        """Set to True because Raspberry Pi doesn't use hardware dithering"""



        self.GAMMA_TABLE_PATH = os.path.join(os.path.dirname(__file__), 'gamma_table.npy')
        """Location of the gamma correction table"""

        self.MIC_RATE = mic_rate
        """Sampling frequency of the microphone in Hz"""

        self.MIN_FREQUENCY = min_frequency
        """Frequencies below this value will be removed during audio processing"""

        self.MAX_FREQUENCY = max_frequency
        """Frequencies above this value will be removed during audio processing"""

        self.N_ROLLING_HISTORY = n_rolling_history
        """Number of past audio frames to include in the rolling window"""

        self.leds = leds
        self._time_prev = time.time() * 1000.0
        """The previous time that the frames_per_second() function was called"""

        # pass in args passed to visualzation to pass to ExpFilter class
        extra_dps_args = {
            "mic_rate": self.MIC_RATE,
            "n_fft_bins": self.N_FFT_BINS,
            "min_frequency": self.MIN_FREQUENCY,
            "max_frequency": self.MAX_FREQUENCY,
            "fps": self.FPS,
            "n_rolling_history": self.N_ROLLING_HISTORY
        }

        self._fps = dsp.ExpFilter(val=self.FPS, alpha_decay=0.2, alpha_rise=0.2, **extra_dps_args)
        """The low-pass filter used to estimate frames-per-second"""

        self.r_filt = dsp.ExpFilter(np.tile(0.01, self.N_PIXELS // 2),
                        alpha_decay=0.2, alpha_rise=0.99, **extra_dps_args)
        self.g_filt = dsp.ExpFilter(np.tile(0.01, self.N_PIXELS // 2),
                            alpha_decay=0.05, alpha_rise=0.3, **extra_dps_args)
        self.b_filt = dsp.ExpFilter(np.tile(0.01, self.N_PIXELS // 2),
                            alpha_decay=0.1, alpha_rise=0.5, **extra_dps_args)
        self.common_mode = dsp.ExpFilter(np.tile(0.01, self.N_PIXELS // 2),
                            alpha_decay=0.99, alpha_rise=0.01, **extra_dps_args)
        self.p_filt = dsp.ExpFilter(np.tile(1, (3, self.N_PIXELS // 2)),
                            alpha_decay=0.1, alpha_rise=0.99, **extra_dps_args)
        self.p = np.tile(1.0, (3, self.N_PIXELS // 2))
        self.gain = dsp.ExpFilter(np.tile(0.01, self.N_FFT_BINS),
                            alpha_decay=0.001, alpha_rise=0.99, **extra_dps_args)
        
        self._prev_spectrum = np.tile(0.01, self.N_PIXELS // 2)

        self.fft_plot_filter = dsp.ExpFilter(np.tile(1e-1, self.N_FFT_BINS),
                            alpha_decay=0.5, alpha_rise=0.99, **extra_dps_args)
        self.mel_gain = dsp.ExpFilter(np.tile(1e-1, self.N_FFT_BINS),
                                alpha_decay=0.01, alpha_rise=0.99, **extra_dps_args)
        self.mel_smoothing = dsp.ExpFilter(np.tile(1e-1, self.N_FFT_BINS),
                                alpha_decay=0.5, alpha_rise=0.99, **extra_dps_args)
        self.volume = dsp.ExpFilter(self.MIN_VOLUME_THRESHOLD,
                            alpha_decay=0.02, alpha_rise=0.02, **extra_dps_args)
        self.fft_window = np.hamming(int(self.MIC_RATE / self.FPS) * self.N_ROLLING_HISTORY)
        self.prev_fps_update = time.time()

         # Number of audio samples to read every time frame
        self.samples_per_frame = int(self.MIC_RATE / self.FPS)

        # Array containing the rolling audio sample window
        self.y_roll = np.random.rand(self.N_ROLLING_HISTORY, self.samples_per_frame) / 1e16

        if effect == "energy":
            self.visualization_effect = self.visualize_energy
        elif effect == "scroll":
            self.visualization_effect = self.visualize_scroll
        else:
            self.visualization_effect = self.visualize_spectrum
        """Visualization effect to display on the LED strip"""


    def frames_per_second(self):
        """Return the estimated frames per second

        Returns the current estimate for frames-per-second (FPS).
        FPS is estimated by measured the amount of time that has elapsed since
        this function was previously called. The FPS estimate is low-pass filtered
        to reduce noise.

        This function is intended to be called one time for every iteration of
        the program's main loop.

        Returns
        -------
        fps : float
            Estimated frames-per-second. This value is low-pass filtered
            to reduce noise.
        """
        time_now = time.time() * 1000.0
        dt = time_now - self._time_prev
        self._time_prev = time_now
        if dt == 0.0:
            return self._fps.value
        return self._fps.update(1000.0 / dt)

    def _normalized_linspace(self, size):
        return np.linspace(0, 1, size)


    def interpolate(self, y, new_length):
        """Intelligently resizes the array by linearly interpolating the values

        Parameters
        ----------
        y : np.array
            Array that should be resized

        new_length : int
            The length of the new interpolated array

        Returns
        -------
        z : np.array
            New array with length of new_length that contains the interpolated
            values of y.
        """
        if len(y) == new_length:
            return y
        x_old = self._normalized_linspace(len(y))
        x_new = self._normalized_linspace(new_length)
        z = np.interp(x_new, x_old, y)
        return z





    def visualize_scroll(self, y):
        """Effect that originates in the center and scrolls outwards"""
        y = y**2.0
        self.gain.update(y)
        y /= self.gain.value
        y *= 255.0
        r = int(np.max(y[:len(y) // 3]))
        g = int(np.max(y[len(y) // 3: 2 * len(y) // 3]))
        b = int(np.max(y[2 * len(y) // 3:]))
        # Scrolling effect window
        self.p[:, 1:] = self.p[:, :-1]
        self.p *= 0.98
        self.p = gaussian_filter1d(self.p, sigma=0.2)
        # Create new color originating at the center
        self.p[0, 0] = r
        self.p[1, 0] = g
        self.p[2, 0] = b
        # Update the LED strip
        return np.concatenate((self.p[:, ::-1], self.p), axis=1)


    def visualize_energy(self, y):
        """Effect that expands from the center with increasing sound energy"""
        y = np.copy(y)
        self.gain.update(y)
        y /= self.gain.value
        # Scale by the width of the LED strip
        y *= float((self.N_PIXELS // 2) - 1)
        # Map color channels according to energy in the different freq bands
        scale = 0.9
        r = int(np.mean(y[:len(y) // 3]**scale))
        g = int(np.mean(y[len(y) // 3: 2 * len(y) // 3]**scale))
        b = int(np.mean(y[2 * len(y) // 3:]**scale))
        # Assign color to different frequency regions
        self.p[0, :r] = 255.0
        self.p[0, r:] = 0.0
        self.p[1, :g] = 255.0
        self.p[1, g:] = 0.0
        self.p[2, :b] = 255.0
        self.p[2, b:] = 0.0
        self.p_filt.update(self.p)
        self.p = np.round(self.p_filt.value)
        # Apply substantial blur to smooth the edges
        self.p[0, :] = gaussian_filter1d(self.p[0, :], sigma=4.0)
        self.p[1, :] = gaussian_filter1d(self.p[1, :], sigma=4.0)
        self.p[2, :] = gaussian_filter1d(self.p[2, :], sigma=4.0)
        # Set the new pixel value
        return np.concatenate((self.p[:, ::-1], self.p), axis=1)




    def visualize_spectrum(self, y):
        """Effect that maps the Mel filterbank frequencies onto the LED strip"""
        y = np.copy(self.interpolate(y, self.N_PIXELS // 2))
        self.common_mode.update(y)
        diff = y - self._prev_spectrum
        self._prev_spectrum = np.copy(y)
        # Color channel mappings
        r = self.r_filt.update(y - self.common_mode.value)
        g = np.abs(diff)
        b = self.b_filt.update(np.copy(y))
        # Mirror the color channels for symmetric output
        r = np.concatenate((r[::-1], r))
        g = np.concatenate((g[::-1], g))
        b = np.concatenate((b[::-1], b))
        output = np.array([r, g,b]) * 255
        return output


    def microphone_update(self, audio_samples):
        # Normalize samples between 0 and 1
        y = audio_samples / 2.0**15
        # Construct a rolling window of audio samples
        self.y_roll[:-1] = self.y_roll[1:]
        self.y_roll[-1, :] = np.copy(y)
        self.y_data = np.concatenate(self.y_roll, axis=0).astype(np.float32)
        
        vol = np.max(np.abs(self.y_data))
        if vol < self.MIN_VOLUME_THRESHOLD:
            print('No audio input. Volume below threshold. Volume:', vol)
            self.leds.pixels = np.tile(0, (3, self.N_PIXELS))
            self.leds.update()
        else:
            # Transform audio input into the frequency domain
            N = len(self.y_data)
            N_zeros = 2**int(np.ceil(np.log2(N))) - N
            # Pad with zeros until the next power of two
            self.y_data *= self.fft_window
            y_padded = np.pad(self.y_data, (0, N_zeros), mode='constant')
            YS = np.abs(np.fft.rfft(y_padded)[:N // 2])
            # Construct a Mel filterbank from the FFT data
            mel = np.atleast_2d(YS).T * self.b_filt.mel_y.T
            # Scale data to values more suitable for visualization
            # mel = np.sum(mel, axis=0)
            mel = np.sum(mel, axis=0)
            mel = mel**2.0
            # Gain normalization
            self.mel_gain.update(np.max(gaussian_filter1d(mel, sigma=1.0)))
            mel /= self.mel_gain.value
            mel = self.mel_smoothing.update(mel)
            # Map filterbank output onto LED strip
            output = self.visualization_effect(mel)
            self.leds.pixels = output
            self.leds.update()
        if self.DISPLAY_FPS:
            fps = self.frames_per_second()
            if time.time() - 0.5 > self.prev_fps_update:
                self.prev_fps_update = time.time()
                print('FPS {:.0f} / {:.0f}'.format(fps, self.FPS))


if __name__ == '__main__':
    # Initialize LEDs
    gamma_table_path = os.path.join(os.path.dirname(__file__), 'gamma_table.npy')
    leds = led.LED(gamma_table_path=gamma_table_path)
    leds.update()
    # Start listening to live audio stream
    audio = AudioLEDVisualization(leds, effect="scroll")
    mic = microphone.Microphone()
    mic.start_stream(audio.microphone_update, mic_rate=audio.MIC_RATE, fps=audio.FPS)

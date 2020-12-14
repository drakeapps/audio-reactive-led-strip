import time
import numpy as np
import pyaudio

class Microphone:
    def __init__(self):
        self.stream = None
        self.p = None
        self.continue_mic_stream = True

    def start_stream(self, callback, mic_rate=44100, fps=60):
        self.p = pyaudio.PyAudio()
        frames_per_buffer = int(mic_rate / fps)
        self.stream = self.p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=mic_rate,
                        input=True,
                        frames_per_buffer=frames_per_buffer)
        overflows = 0
        prev_ovf_time = time.time()
        while self.continue_mic_stream:
            try:
                y = np.fromstring(self.stream.read(frames_per_buffer, exception_on_overflow=False), dtype=np.int16)
                y = y.astype(np.float32)
                self.stream.read(self.stream.get_read_available(), exception_on_overflow=False)
                callback(y)
            except IOError:
                overflows += 1
                if time.time() > prev_ovf_time + 1:
                    prev_ovf_time = time.time()
                    print('Audio buffer has overflowed {} times'.format(overflows))
            except:
                pass
        try:
            self.stream.stop_stream()
            self.stream.close()
            self.p.terminate()
        except:
            pass

    def stop_stream(self):
        self.continue_mic_stream = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.p:
            self.p.terminate()

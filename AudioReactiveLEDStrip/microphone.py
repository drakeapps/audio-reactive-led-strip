import time
import numpy as np
import pyaudio

CONTINUE_MIC_STREAM = True

class Microphone:
    def __init__(self):
        self.stream = None
        self.continue_mic_stream = True

    def start_stream(self, callback, mic_rate=44100, fps=60):
        global CONTINUE_MIC_STREAM
        self.p = pyaudio.PyAudio()
        frames_per_buffer = int(mic_rate / fps)
        self.stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=mic_rate,
                        input=True,
                        frames_per_buffer=frames_per_buffer)
        overflows = 0
        prev_ovf_time = time.time()
        while self.continue_mic_stream:
            try:
                y = np.fromstring(stream.read(frames_per_buffer, exception_on_overflow=False), dtype=np.int16)
                y = y.astype(np.float32)
                self.stream.read(stream.get_read_available(), exception_on_overflow=False)
                callback(y)
            except IOError:
                overflows += 1
                if time.time() > prev_ovf_time + 1:
                    prev_ovf_time = time.time()
                    print('Audio buffer has overflowed {} times'.format(overflows))
            except:
                pass
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

    def stop_stream(self):
        self.continue_mic_stream = False
        if self.stream:
            self.stream.stop_stream()
            self.stream_close()
        if self.p:
            self.p.terminate()

from __future__ import print_function
from __future__ import division

import platform
import numpy as np
import config

import rpi_ws281x as neopixel

class LED:

    def __init__(self):
        self.strip = neopixel.Adafruit_NeoPixel(config.N_PIXELS, config.LED_PIN,
                                       config.LED_FREQ_HZ, config.LED_DMA,
                                       config.LED_INVERT, config.BRIGHTNESS)
        self.strip.begin()

        self._gamma = np.load(config.GAMMA_TABLE_PATH)
        """Gamma lookup table used for nonlinear brightness correction"""

        self._prev_pixels = np.tile(253, (3, config.N_PIXELS))
        """Pixel values that were most recently displayed on the LED strip"""

        self.pixels = np.tile(1, (3, config.N_PIXELS))
        """Pixel values for the LED strip"""

    def _update_pi(self):
        """Writes new LED values to the Raspberry Pi's LED strip

        Raspberry Pi uses the rpi_ws281x to control the LED strip directly.
        This function updates the LED strip with new values.
        """
        # Truncate values and cast to integer
        self.pixels = np.clip(self.pixels, 0, 255).astype(int)
        # Optional gamma correction
        p = self._gamma[self.pixels] if config.SOFTWARE_GAMMA_CORRECTION else np.copy(self.pixels)
        # Encode 24-bit LED values in 32 bit integers
        r = np.left_shift(p[0][:].astype(int), 8)
        g = np.left_shift(p[1][:].astype(int), 16)
        b = p[2][:].astype(int)
        rgb = np.bitwise_or(np.bitwise_or(r, g), b)
        # Update the pixels
        for i in range(config.N_PIXELS):
            # Ignore pixels if they haven't changed (saves bandwidth)
            if np.array_equal(p[:, i], self._prev_pixels[:, i]):
                continue
            #strip._led_data[i] = rgb[i]
            self.strip._led_data[i] = int(rgb[i])
        self._prev_pixels = np.copy(p)
        self.strip.show()

    def update(self):
        """Updates the LED strip values"""
        self._update_pi()


# Execute this file to run a LED strand test
# If everything is working, you should see a red, green, and blue pixel scroll
# across the LED strip continously
if __name__ == '__main__':
    import time
    # Turn all pixels off
    leds = LED()
    leds.pixels *= 0
    leds.pixels[0, 0] = 255  # Set 1st pixel red
    leds.pixels[1, 1] = 255  # Set 2nd pixel green
    leds.pixels[2, 2] = 255  # Set 3rd pixel blue
    print('Starting LED strand test')
    while True:
        leds.pixels = np.roll(leds.pixels, 1, axis=1)
        leds.update()
        time.sleep(.1)

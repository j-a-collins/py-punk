'''
Core module for PyPunk: a long-term project aimed at
training a self-driving car in Cyberpunk 2077
J-A-Collins
'''

# Imports
import numpy as np
from PIL import ImageGrab
import cv2
import time


def screen_record():
    """
    Captures a screen recording of the region defined by the bbox parameter and displays it in a window.

    The recording will continue until the user presses the 'q' key.

    Parameters:
    -----------
    None

    Returns:
    --------
    None
    """
    last_time = time.time()
    while(True):
        printscreen = np.array(ImageGrab.grab(bbox=(0, 40, 1200, 600)))
        print(f"Loop time={time.time()-last_time}")
        last_time = time.time()
        cv2.imshow('window', cv2.cvtColor(printscreen, cv2.COLOR_BGR2RGB))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

if __name__ == "__main__":
    screen_record()



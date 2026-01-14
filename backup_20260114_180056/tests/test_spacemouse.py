import time

import pyspacemouse

success = pyspacemouse.open()
if success:
    while 1:
        state = pyspacemouse.read()
        print(state.buttons)
        time.sleep(0.01)

import time

from xlib.device.keyboard import KeyboardReader

kb = KeyboardReader()
try:
    print("Press 'q' to quit")
    while True:
        start_time = time.perf_counter()
        if kb.is_pressed('q'):
            print("Quit pressed")
            break
        # 模拟其他日志输出
        print("Running...")
        elapsed_time = time.perf_counter() - start_time
        print(f"Loop time: {elapsed_time:.4f} seconds")
        time.sleep(0.01)
finally:
    kb.restore()
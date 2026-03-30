import serial
import time
from queue import Queue
import threading


class ArduinoSerial:
    def __init__(self, port="/dev/ttyACM0", baudrate=115200, timeout=0.2):
        self.serial = serial.Serial(port, baudrate=baudrate, timeout=timeout)
        time.sleep(2)
        self.serial.reset_input_buffer()
        print(f"[Arduino] Connected on {port}")

    def send_command(self, command, delay_us=None, wait_done=True,
                     done_tokens=("DONE", "INTERRUPTED"), overall_timeout=10.0):
        full_command = command if delay_us is None else f"{command} {delay_us}"
        self.serial.reset_input_buffer()
        self.serial.write((full_command + "\n").encode("utf-8"))

        if not wait_done:
            return

        start = time.time()
        try:
            while True:
                line = self.serial.readline().decode("utf-8").strip()
                if line:
                    print(f"[Arduino] Response: {line}")
                    if line in done_tokens:
                        break
                if (time.time() - start) > overall_timeout:
                    print("[Arduino] Warning: overall timeout waiting for DONE")
                    break
        except Exception as e:
            print(f"[Arduino] Error during send_command: {e}")

    def close(self):
        if self.serial and self.serial.is_open:
            self.serial.close()
            print("[Arduino] Serial closed.")

class AdvancerUnit:
    def __init__(self, port="/dev/ttyACM0", baudrate=115200, timeout=1):
        self.arduino = ArduinoSerial(port=port, baudrate=baudrate, timeout=timeout)

        self.queue = Queue()
        self._stop_event = threading.Event()

        self.thread = threading.Thread(target=self._arduino_worker, daemon=True)
        self.thread.start()
    def _arduino_worker(self):
        while not self._stop_event.is_set():
            item = self.queue.get()
            if item is None:
                self.queue.task_done()
                break
            command, delay = item
            self.arduino.send_command(command, delay)  
            self.queue.task_done()


    @staticmethod
    def distance_to_steps(distance_mm):
        mm_per_step = 0.166
        return int(distance_mm / mm_per_step)

    def forward(self, distance_mm, delay_us=20):
        steps = self.distance_to_steps(distance_mm)
        self.queue.put((f"ON {steps}", delay_us))
        self.queue.join()

    def backward(self, distance_mm, delay_us=20):
        steps = self.distance_to_steps(distance_mm)
        self.queue.put((f"REV {steps}", delay_us))
        self.queue.join()


    def shutdown(self):
        self.queue.put(None)
        self._stop_event.set()
        self.thread.join(timeout=2.0)
        self.arduino.close()
        print("[AdvancerUnit] Shutdown complete.")

if __name__ == "__main__":
    ady = AdvancerUnit(port="/dev/ttyACM0")
    ady.forward(5)

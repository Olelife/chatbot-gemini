# utils/timer.py
import time

class Timer:
    def __init__(self):
        self.timestamps = {}
        self.results = {}

    def start(self, label: str):
        self.timestamps[label] = time.time()

    def end(self, label: str):
        if label in self.timestamps:
            self.results[label] = time.time() - self.timestamps[label]
        else:
            self.results[label] = None  # fallback

    def summary(self):
        # Calcula "total"
        if "start" in self.timestamps:
            self.results["total"] = time.time() - self.timestamps["start"]

        return self.results

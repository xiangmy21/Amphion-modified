import time

class Timer:
    def __enter__(self):
        self.start_time = time.time()
        self.name = ""
        return self

    def __exit__(self, *args):
        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time
        print(f"{self.name} time: {self.elapsed_time:.3f} seconds")
import os
import time

class FreqMonitor:
    def __init__(self, name:str, display_time:float, logger, verbose = False):
        self.logger = logger
        self.itt = 0
        self.name = name
        self.display_time = display_time
        self.start_time = time.time()
        self.verbose = verbose
        self.frequency = 0
    
    def tic(self):
        self.itt += 1
        now = time.time()
        delta_t = now - self.start_time
        if delta_t > self.display_time:
            self.frequency = self.itt / delta_t
            if self.verbose:
                self.logger.info(f'{self.name}: {self.frequency:.2f} Hz')
            self.itt = 0
            self.start_time = now

def workspace_dir():
    workspace_dir = os.path.dirname(os.path.abspath(__file__))

    workspace_dir, tail = os.path.split(workspace_dir)
    while tail not in ["src", "venv", "install", "build"]:
        workspace_dir, tail = os.path.split(workspace_dir)

    return workspace_dir

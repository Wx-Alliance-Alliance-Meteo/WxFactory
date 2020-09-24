
from time import time

class Timer:
   def __init__(self, initial_time):
      self.times = []
      self.start_times = []
      self.start_time = 0.0
      self.stop_time  = 0.0

      self.initial_time = initial_time

   def start(self):
      self.start_time = time()
      self.start_times.append(self.start_time)

   def stop(self):
      self.stop_time = time()
      self.times.append(self.stop_time - self.start_time)

   def last_time(self):
       return self.times[-1]


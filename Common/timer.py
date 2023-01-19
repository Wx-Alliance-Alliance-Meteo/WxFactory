
from time import time

class Timer:
   def __init__(self, initial_time = 0.0):
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

   def average_time(self):
      return sum(self.times) / len(self.times)

class TimerGroup:
   def __init__(self, num_timers, initial_time):
      self.num_timers = num_timers
      self.timers = [Timer(initial_time) for i in range(num_timers)]

   def __getitem__(self, key):
      return self.timers[key]


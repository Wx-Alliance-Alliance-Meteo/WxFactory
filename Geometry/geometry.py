
class Geometry:
   """
   Abstract class that groups different geometries
   """
   def __init__(self, grid_type: str) -> None:
      self.grid_type = grid_type

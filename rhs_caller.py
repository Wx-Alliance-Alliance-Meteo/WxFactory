

class RhsCaller:
   def __init__(self, function, geometry, operators, metric, topo, ptopo, nb_sol_pts, nb_elements, case_num,
                use_filter, timers):
      self.function   = function
      self.geometry   = geometry
      self.operators  = operators
      self.metric     = metric
      self.topo       = topo
      self.ptopo      = ptopo
      self.nb_sol_pts = nb_sol_pts
      self.nb_elem    = nb_elements
      self.case_num   = case_num
      self.use_filter = use_filter
      self.timers     = timers

   def __call__(self, field):
      return self.function(field, self.geometry, self.operators, self.metric, self.topo, self.ptopo,
                           self.nb_sol_pts, self.nb_elem, self.case_num, self.use_filter, self.timers)
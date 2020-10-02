

class RhsCaller:
   #rhs_handle = lambda q: rhs_sw(q, geom, mtrx, metric, topo, comm_dist_graph, param.nbsolpts, param.nb_elements, param.case_number, rhs_timers)

   def __init__(self, function, geometry, operators, metric, topo, comm_graph, nb_sol_pts, nb_elements, case_num, timers):
      self.function   = function
      self.geometry   = geometry
      self.operators  = operators
      self.metric     = metric
      self.topo       = topo
      self.comm_graph = comm_graph
      self.nb_sol_pts = nb_sol_pts
      self.nb_elem    = nb_elements
      self.case_num   = case_num
      self.timers     = timers

   def __call__(self, field):
      return self.function(field, self.geometry, self.operators, self.metric, self.topo, self.comm_graph,
                           self.nb_sol_pts, self.nb_elem, self.case_num, self.timers)
import netCDF4 as nc
import numpy as np 


#tol_comp = "1e-12"
tf = "900"
nbsolpts = "7"
nb_elements = "168"
num_of_cores = "24"
grid_size = nbsolpts + "x" + nb_elements + "x" +nb_elements
EPImethod="epi2"
dt_comp = "900"
read_from_line = 150
num_of_read_line = 21
controller="deadbeat"

METHODS = ['ERK4(3)4L'] #'RK23', 'RK45', 'MERSON4', 'ARK3(2)4L[2]SA-ERK' ,'ERK3(2)3L', 'ERK4(3)4L']
methods = [item.lower() for item in METHODS] #['kiops_epi2', 'rk23', 'rk45', 'merson4', 'ark3(2)4l[2]sa-erk' ,'erk3(2)3l', 'erl4(3)4l']

for method in METHODS:
    #data_file = "compare_controller_data_"+EPImethod+"_" + method + "_dt_" + dt_comp + "_step_size_control.txt"
    #with open(data_file, "w") as datafile:

        for controller in ["deadbeat", "h211b","pi3040"]:
            for tol_comp in ["1e-5"]: #,"1e-8", "1e-12"]:   

                file_loc = "/space/hall0/work/eccc/mrd/rpnatm/siw001/gef_data/case6/compare_controller/"+ EPImethod+"_"+controller 
                #"/space/hall0/work/eccc/mrd/rpnatm/siw001/work_precision/dt_"+dt_comp + "/tol_"+tol_comp 

                subfile_loc = file_loc+"/" + method.lower() + "/tol_" + tol_comp + "_dt_" + dt_comp + "_tf_" + tf + "_grid_" + nbsolpts + "x" + nb_elements + "x" +nb_elements + "_with_"+num_of_cores+"_cores"
                
                timing_file = subfile_loc+"/" +method.lower()+ "_case6_tol_"+ tol_comp + "_dt_" + dt_comp + "_tf_" + tf + "_grid_" + nbsolpts + "x" + nb_elements + "x" +nb_elements + "_using_"+num_of_cores+"_cores.txt" 
                
                
                if method == 'ARK3(2)4L[2]SA-ERK':
                    Method = 'ARK32'
                elif method == 'ERK3(2)3L':
                    Method = 'ERK32'
                elif method == "ERK4(3)4L":
                    Method = "ERK43"
                else: 
                    Method = method

                data_file_accepted = "./step_size_control_data/"+Method.lower()+"/compare_controller_data_"+EPImethod+"_" + Method.lower() + "_dt_" + dt_comp + "_tol_"+tol_comp+"_controller_"+controller+"_accepted_step_size_control.txt"
                data_file_rejected = "./step_size_control_data/"+Method.lower()+"/compare_controller_data_"+EPImethod+"_" + Method.lower() + "_dt_" + dt_comp + "_tol_"+tol_comp+"_controller_"+controller+"_rejected_step_size_control.txt"

                with open(data_file_rejected,"w") as datafilerej:
                    with open(timing_file) as f:
                         for line in (f.readlines()[read_from_line:-1]):
                             if line[0:13] == "rejected step": 
                                 datafilerej.write(line[14:])
                with open(data_file_accepted,"w") as datafileaccept:
                    with open(timing_file) as f:
                         for line in (f.readlines()[read_from_line:-1]):
                             if line[0:13] == "accepted step":
                                 datafileaccept.write(line[14:]) 
                









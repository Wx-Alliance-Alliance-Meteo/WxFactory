

import csv
import array as arr
#import numpy as np

EPImethod="epi4"
file_dir = "/home/siw001/gef/vicky/EXODE_testoutput_case6/"+EPImethod+"/strong_scaling"
nbsolpts=7
method_opt=["KIOPS", "RK23", "RK45", "MERSON4", "ARK3(2)4L[2]SA-ERK", "ERK3(2)3L", "ERK4(3)4L"]
tol="1e-12"
dt="900"
tf="14400"
cores=[294, 864, 1176, 2646]
nb_elements_sel = [168]
run_num="1"

data_file = "case6_tol_" + tol + "_dt_" + dt + "_tf_" + tf  + "_strong_scaling_"+EPImethod+"_run1.txt"
with open(data_file, "w") as datafile:
    for method in method_opt:
    #data_file = "case6_tol_" + tol + "_dt_" + dt + "_tf_" + tf  + "_strong_scaling_"+EPImethod+"_run"+run_num+"_"+method+".txt"
    #with open(data_file, "w") as datafile:
        datafile.write(method)
        datafile.write('\n')
        datafile.write("grid_size num_of_cores timing \n")
        for nb_elements in nb_elements_sel:
            sub_dir = str(nbsolpts) + "x" + str(nb_elements) + "x" + str(nb_elements)
            #datafile.write("grid size " + sub_dir + " with ")
            #datafile.write('\n')
            #nb_element_str = str(nb_elements) + " "
            #datafile.write(nb_element_str)
            for num_of_core in cores:
                file_name = method + "_case6_tol_" + tol + "_dt_" + dt + "_tf_" + tf + "_grid_" + sub_dir + "_using_" + str(num_of_core) + "_cores.txt"
                file_path = file_dir + "/" + sub_dir + "/" + file_name  
                #datafile.write(str(num_of_core) +" cores ")
                #datafile.write('\n')
                num_of_core_str = str(num_of_core) + " "
                datafile.write(num_of_core_str)
                with open(file_path) as f:
                    for line in (f.readlines()[-16:]):
                        datafile.write(line.rstrip('\n'))
                        datafile.write(" ")
                datafile.write('\n')
        datafile.write('\n')

    datafile.write('\n')
    datafile.write('Repeat for MATLAB entry \n')  
    for method in method_opt:
        datafile.write(method + ' = [ \n')
        for nb_elements in nb_elements_sel:
            sub_dir = str(nbsolpts) + "x" + str(nb_elements) + "x" + str(nb_elements)
            #datafile.write("grid size " + sub_dir + " with ")
            #datafile.write('\n')
            #nb_element_str = str(nb_elements) + " "
            #datafile.write(nb_element_str)
            for num_of_core in cores:
                file_name = method + "_case6_tol_" + tol + "_dt_" + dt + "_tf_" + tf + "_grid_" + sub_dir + "_using_" + str(num_of_core) + "_cores.txt"
                file_path = file_dir + "/" + sub_dir + "/" + file_name
                #datafile.write(str(num_of_core) +" cores ")
                #datafile.write('\n')
                num_of_core_str = str(num_of_core) + " "
                datafile.write(num_of_core_str)
                with open(file_path) as f:
                    for line in (f.readlines()[-16:]):
                        datafile.write(line.rstrip('\n'))
                        datafile.write(" ")
                datafile.write('\n')
        datafile.write(']; \n')
datafile.close() 
    
                
        

























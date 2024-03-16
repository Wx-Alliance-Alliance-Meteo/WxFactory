import csv
import array as arr
#import numpy as np

file_dir = "/home/siw001/gef/vicky/EXODE_testoutput_case6/weak_scaling"
nbsolpts=4
method_opt=["KIOPS", "RK23", "RK45", "MERSON4", "ARK3(2)4L[2]SA-ERK", "ERK3(2)3L", "ERK4(3)4L"]
tol="1e-7"
dt="900"
tf="9000"
cores=[6,24,96,384,1536]
nb_elements_sel = [48,96,192,384,768]
data_file = "case6_tol_" + tol + "_dt_" + dt + "_tf_" + tf  + "_weak_scaling.txt"


with open(data_file, "w") as datafile: 

    for method in method_opt:
        datafile.write(method)
        datafile.write('\n')
        datafile.write("grid_size num_of_cores timing \n")
        for nb_elements in nb_elements_sel:
            sub_dir = str(nbsolpts) + "x" + str(nb_elements) + "x" + str(nb_elements)
            #datafile.write("grid size " + sub_dir + " with ")
            #datafile.write('\n')
            nb_element_str = str(nb_elements) + " "
            datafile.write(nb_element_str) 
            if nb_elements == 48:
                num_of_core = 6
            elif nb_elements == 96:
                num_of_core = 24
            elif nb_elements == 192:
                num_of_core = 96
            elif nb_elements == 384:
                num_of_core = 384
            elif nb_elements == 768:
                num_of_core = 1536
            else:
                print("check problem size")
                

            file_name = method + "_case6_tol_" + tol + "_dt_" + dt + "_tf_" + tf + "_grid_" + sub_dir + "_using_" + str(num_of_core) + "_cores.txt"
            file_path = file_dir + "/" + sub_dir + "/" + file_name  
            #datafile.write(str(num_of_core) +" cores ")
            #datafile.write('\n')
            num_of_core_str = str(num_of_core) + " " 
            datafile.write(num_of_core_str) 
            with open(file_path) as f:
                for line in (f.readlines()[-10:]):
                    datafile.write(line.rstrip('\n'))
                    datafile.write(" ")
            datafile.write('\n')
datafile.close() 
    
                
        

























import netCDF4 as nc
import numpy as np 

'''
ref_sol_file= '/home/siw001/gef/vicky/ref_sol_case6/tol_1e-12_dt_10_tf_14400_grid_7x168x168/out.nc' 
ds_ref = nc.Dataset(ref_sol_file)

#print(ds_ref)
#time = ds_ref['time'][:]
#print(time) 

h_ref = ds_ref['h'][-1,:,:,:]
U_ref = ds_ref['U'][-1,:,:,:]
V_ref = ds_ref['V'][-1,:,:,:]
RV_ref = ds_ref['RV'][-1,:,:,:]
PV_ref = ds_ref['PV'][-1,:,:,:]

ref_sol_file2 = '/home/siw001/gef/vicky/ref_sol_case6/tol_1e-12_dt_5_tf_14400_grid_7x168x168/out.nc'
ds_ref2 = nc.Dataset(ref_sol_file2)

#print(ds_ref2)
#time = ds_ref2['time'][:]
#print(time)

h_ref2 = ds_ref2['h'][-1,:,:,:]
U_ref2 = ds_ref2['U'][-1,:,:,:]
V_ref2 = ds_ref2['V'][-1,:,:,:]
RV_ref2 = ds_ref2['RV'][-1,:,:,:]
PV_ref2 = ds_ref2['PV'][-1,:,:,:]

h = np.max(np.absolute( h_ref - h_ref2)) 
U = np.max(np.absolute( U_ref - U_ref2))
V = np.max(np.absolute( V_ref - V_ref2))
RV = np.max(np.absolute( RV_ref - RV_ref2))
PV = np.max(np.absolute( PV_ref - PV_ref2))


print('accuracy of ref solution:', h, U, V, RV, PV)
''' 


#tol_comp = "1e-12"
tf = "9000"
nbsolpts = "7"
nb_elements = "168"
num_of_cores = "24"
grid_size = nbsolpts + "x" + nb_elements + "x" +nb_elements
EPImethod="epi2"
dt_comp = "900"
num_of_read_line = 21
controller="deadbeat"

METHODS = ['RK23', 'RK45', 'MERSON4', 'ARK3(2)4L[2]SA-ERK' ,'ERK3(2)3L', 'ERK4(3)4L']
methods = [item.lower() for item in METHODS] #['kiops_epi2', 'rk23', 'rk45', 'merson4', 'ark3(2)4l[2]sa-erk' ,'erk3(2)3l', 'erl4(3)4l']

for method in METHODS:
    data_file = "compare_controller_data_"+EPImethod+"_" + method + "_dt_" + dt_comp + ".txt"
    with open(data_file, "w") as datafile:

        for controller in ["deadbeat", "h110", "h211b", "h211d", "h211pi", "pi3040", "pi4020"]:
    # compare by controller 
    # data_file = "compare_controller_data_"+EPImethod+"_" + method + "_dt_" + dt_comp + ".txt"
    
    # compare by method 
    # data_file = "compare_controller_data_"+EPImethod+"_dt_" + dt_comp + "_" + controller + ".txt"
    # with open(data_file, "w") as datafile:
    #    for method in METHODS:
            datafile.write("\n")
            datafile.write(controller + "\n") 
            #datafile.write(method+" \n")
            for tol_comp in ["1e-5","1e-8", "1e-12"]:   
                
                file_loc = "/space/hall0/work/eccc/mrd/rpnatm/siw001/gef_data/case6/compare_controller/"+ EPImethod+"_"+controller 
                #"/space/hall0/work/eccc/mrd/rpnatm/siw001/work_precision/dt_"+dt_comp + "/tol_"+tol_comp 

                subfile_loc = file_loc+"/" + method.lower() + "/tol_" + tol_comp + "_dt_" + dt_comp + "_tf_" + tf + "_grid_" + nbsolpts + "x" + nb_elements + "x" +nb_elements + "_with_"+num_of_cores+"_cores"
                
                timing_file = subfile_loc+"/" +method.lower()+ "_case6_tol_"+ tol_comp + "_dt_" + dt_comp + "_tf_" + tf + "_grid_" + nbsolpts + "x" + nb_elements + "x" +nb_elements + "_using_"+num_of_cores+"_cores.txt" 
                datafile.write("tolerance = " +  tol_comp + "\n") 

                with open(timing_file) as f:
                     for line in (f.readlines()[-num_of_read_line:-1]):
                         if line[0:7] == "Running":
                             datafile.write("-------------------------------------------- \n")
                         else:
                             datafile.write(line) 

                
                









import matplotlib.pyplot as plt
import numpy as np 


def read_data(filename):
    time_step = []
    error = [] 
    with open(filename, 'r') as f:
        array=np.loadtxt(filename) 
        time_step = array[:,1]
        error = array[:, 3]
    return time_step, error    



epi = "epi5"
tol = "1e-06"
Nx = "401"
data_dir = "/home/siw001/hall5/testoutput/ADR_2D/"+epi+"/new_code_july_2025/"

METHOD_krylov=["pmex", "kiops"]

METHOD_embRK= ["BS32",  "M43", "KC32"]

METHOD_EXODE=["EXLRK32", "EXLRK43", "ExLRK4(3)minA5","ExLRK4(3)minA5param0", "ExLRK4(3)minA5param0d5","ExLRK4(3)minA5paramN0D25", "Ralston43"]

plt.figure(figsize=(12, 6))

for method in METHOD_krylov:
    filename = method+"_tol_"+tol+"_Nx_"+Nx+"_repeat_0.txt" 

    time_step,error = read_data(data_dir+filename)
    plt.loglog(time_step, error, linestyle = '--', marker='o', label=method) 

for method in METHOD_embRK:
    filename = method+"_tol_"+tol+"_Nx_"+Nx+"_repeat_0.txt"

    time_step,error = read_data(data_dir+filename)
    plt.loglog(time_step, error, linestyle = '--', marker='s', label=method)

for method in METHOD_EXODE:
    filename = method+"_tol_"+tol+"_Nx_"+Nx+"_repeat_0.txt"

    time_step,error = read_data(data_dir+filename)
    plt.loglog(time_step, error, linestyle = '-', marker='x', label=method)

plt.loglog(time_step, 1e8*time_step**5, linestyle = '--', label = "order 5")

plt.xlabel('time step')
plt.ylabel('error')
plt.title('convergence diagram')
plt.grid(True)

plt.legend(loc='best',bbox_to_anchor=(1.05, 1)) 
plt.tight_layout()
plt.savefig("convergence_"+epi +"_tol_"+tol+".png", dpi = 300)


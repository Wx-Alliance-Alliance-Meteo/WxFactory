import math 
import sys 
sys.path.insert(1, '/home/siw001/gef')
import numpy as np 
#from solvers import exode 
from integrators import Integrator, Epi
from solvers import exode, kiops, matvec_fun
from math import pi
from collections  import deque
from scipy.integrate import solve_ivp
from epi_for_de import Epi_for_DE
from epi_stiff_for_de import EpiStiff_for_DE
import time 

alpha = -10. 
epsil = 1./100. 
gamma = 100.
Nx = 41
x0 = 0
xn = 1
dx = (xn-x0)/(Nx-1)
x = np.linspace(x0,xn,Nx) 
y = np.linspace(x0,xn,Nx)
t0 = 0
tf = 0.1 
#N_step = 100
#dt = (tf-t0)/N_step
#exode_method='RK23' 
controller="PI3040"
epi_method = 6
tol = 1e-16

#initial condition 256(xy(1 −x)(1 −y))^2+0.3
u0 = np.zeros((Nx,Nx))
for i in range(Nx):
    for j in range(Nx):
        u0[i,j] = 256.*((x[i]*y[j]*(1.-x[i])*(1.-y[j]))**2)+0.3
u0 =u0.flatten() 

def boundary(u,bc):
    if bc == "N":
        u_top = np.concatenate(([0],u[1,:],[0])).reshape(1,Nx+2)
        u_bot = np.concatenate(([0],u[-2,:],[0])).reshape(1,Nx+2)
        u_lef = u[:,1].reshape(Nx,1)
        u_rig = u[:,-2].reshape(Nx,1)
        u = np.concatenate((u_lef,u,u_rig), axis=1)
        u = np.concatenate((u_top,u,u_bot),axis = 0)
    else:
        print("boudnary condition not supported")
    
    return u 

def f2D_Dx(u): 
    Dx = 1/(2*dx)*(u[1:-1,2:] - u[1:-1,:-2] + u[2:,1:-1] - u[:-2,1:-1])  
    return Dx

def f2D_D2x(u): 
    D2x = 1/(dx**2)*(u[1:-1,2:] - 2*u[1:-1,1:-1]  + u[1:-1,:-2] + u[2:,1:-1] - 2*u[1:-1,1:-1]+u[:-2,1:-1])
    return D2x 

def rhsfunc(u):
    u = u.reshape(Nx, Nx) 
    RU = gamma*u*(u-0.5)*(1-u)
    
    uN = boundary(u,"N")   
    DU =-alpha*f2D_Dx(uN)
    LU = epsil*f2D_D2x(uN)

    dudt = LU + DU + RU 
    dudt = dudt.flatten() 
    u= u.flatten()
    return dudt

def rhsfunca_solveivp(t,u):
    u = u.reshape(Nx, Nx)
    RU = gamma*u*(u-0.5)*(1-u)

    uN = boundary(u,"N")
    DU =-alpha*f2D_Dx(uN)
    LU = epsil*f2D_D2x(uN)

    dudt = LU + DU + RU
    dudt = dudt.flatten()
    u= u.flatten()
    return dudt

rhs = rhsfunc(u0)
t = t0 
u = u0

def l2norm(array):
    return abs(math.sqrt(sum(np.square(abs(array)))))



# reference solution 
with open ("./2D_ADR_ref_sol_Nx_41_tf_0d1.csv") as ref_file:
    refsol = np.loadtxt(ref_file,delimiter=",")


timing_repeat = 1

for tol in [1e-8, 1e-10]:
    epi_stiff = "False" 
    edit_first_step = "False"
    first_step_controller = 0.9
    for epi_method in [6]:
        # Choose epi or epi_stiff
        if epi_stiff == "True": 
            print("epi_method = epi_stiff", epi_method, " with tolerance = ", tol)
        elif epi_stiff == "False": 
            print("epi_method = epi", epi_method, " with tolerance = ", tol)
        
        # repeat $timing_repeat number of times for timing 
        for repeat in range(timing_repeat): 
          #print("Run number ", repeat) 
          for EXODE_method in ['kiops', 'RK23', 'RK45', 'MERSON4', 'ARK32', 'ERK32']: #,'ERK43']: 
              #['kiops','RK23', 'RK45', 'MERSON4', 'ARK32', 'ERK32','ERK43']:
            
            # file to save testoutput
            if edit_first_step == "False":
                testoutput_file = "./testoutput/epi"+str(epi_method)+"/"+EXODE_method+"_tol_"+str(tol)+"time_step_200_600.txt"
            else:
                testoutput_file = "./testoutput/epi"+str(epi_method)+"/"+EXODE_method+"_tol_"+str(tol)+"_first_step_controller_"+str(first_step_controller)+"time_step_200_600.txt"
            with open(testoutput_file, "w") as outputfile:


                if EXODE_method == "ARK32":
                    exode_method = "ark3(2)4l[2]sa-erk" 
                elif EXODE_method == "ERK32":
                    exode_method = "erk3(2)3l"
                elif EXODE_method == "ERK43":
                    exode_method = "erk4(3)4l"
                else: 
                    exode_method = EXODE_method

                print(exode_method)
                # for convergence
                prev_err = 1.
                prev_dt = 2.
                
                if epi_stiff == "True": 
                    if exode_method == "kiops": 
                        method = EpiStiff_for_DE(epi_method, rhsfunc, tol,'kiops',  exode_method, controller)
                    else:
                        method = EpiStiff_for_DE(epi_method, rhsfunc, tol,'exode',  exode_method, controller)
                elif epi_stiff == "False":
                    if exode_method == "kiops": 
                        method = Epi_for_DE(epi_method, rhsfunc, tol,'kiops',  exode_method, controller)
                    else:
                        method = Epi_for_DE(epi_method, rhsfunc, tol,'exode',  exode_method, controller) 
                
                print("tolerance, dt, N_step, error, order, timing")
                for N_step in [200,300,400,500,600]:
                    dt = (tf-t0)/N_step
                    u = u0 
                    t = t0

                    begin = time.perf_counter()
                    for step in range(N_step): 
                        if edit_first_step == "False":
                            first_step = 1. 
                        else: 
                            if step == 0:
                                first_step = 1.
                            else: 
                                first_step = first_step_controller*stats[4]

                        u,stats = method.step(u, dt, first_step) 
                        t = t+dt 
                    end = time.perf_counter()
                    timing = end-begin
                    #print("Elapsed Time: ", f"{end-begin:0.4f}", "seconds")
            
                    error = l2norm(u-refsol)
                    print(tol, dt, N_step, error, math.log(prev_err/error, prev_dt/dt), timing)
                    outputfile.write(str(tol)+" "+ str(dt) +" "+str(N_step) + " " + str(error)+" "+ str(math.log(prev_err/error, prev_dt/dt))+" " +str(timing) + "\n")
                    
                    prev_err = error
                    prev_dt = dt
                print("\n") 



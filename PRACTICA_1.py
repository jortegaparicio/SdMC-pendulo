#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 16:41:37 2021

@author: Juan Antonio Ortega Aparicio & César Borao Moratinos
"""

# Inverted pendulum analysis

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.signal as ss
from scipy.integrate import odeint
import control


# Non-linear function matrix
def f(xVec, t, params):
    
    # State Variables vector
    # x1 = x, x2 = dX, x3 = theta, x4 = dTheta
    x, dX, theta, dTheta = xVec
    
    # System parameters
    M,m,L,b,u,R,g = params
    
    # x_dot = f1 
    # x_dot_dot = f2
    # theta_dot = f3
    # theta_dot_dot = f4
    
    
    return [dX,
            (((-2*L**2-R**2)*(-b*dX+dTheta**2*L*m*np.sin(theta)+u))/(2*L**2*m*np.cos(theta)**2-m*R**2-M*R**2-2*L**2*m-2*L**2*M) - (2*g*L**2*m*np.cos(theta)*np.sin(theta))/(2*L**2*m*np.cos(theta)**2-m*R**2-M*R**2-2*L**2*m-2*L**2*M)),
            dTheta,
            ((2*L*(-dX*b+dTheta**2*L*m*np.sin(theta)+u)*np.cos(theta))/(2*L**2*m*np.cos(theta)**2-m*R**2-M*R**2-2*L**2*m-2*L**2*M) - (g*L*m*(-2*m-2*M)*np.sin(theta))/(2*L**2*m**2*np.cos(theta)**2-2*L**2*m**2-m**2*R**2-m*M*R**2-2*L**2*m*M))
            ]


if __name__ == "__main__":
    plt.close('all')
    
    # System Parameters 
    u = 1      
    g = 9.81    
    L = 0.3
    m = 0.2
    I = 0.006
    M = 0.5
    b = 0.1
    
    R = np.sqrt(2*I/m)
      
    # Time vector
    t = np.linspace(0,1,500)
    
    xRef = np.array([0.0,   # x_0 
                     np.pi,   # dot_x_0
                     0.0,   # theta_0
                     0.0])  # dot_theta_0
    
    deltaX   = np.array([0.0,   # x_init 
                         0.0,    # dot_x_init
                         0.0,   # theta_init
                         0.0])    # dot_theta_init
    
    #%% Calculating Non linear sistem (EDOs)
    
    xNL = odeint(f, xRef + deltaX, t, args=([M,m,L,b,u,R,g],))
    
    #%% EN DESARROLLO, NO UTILIZAR. Linearized system using Jacobian arrays (Taylor)
   
    # Note: Jx and Ju have been calculated assumming xRef = [0]
    
    # Calculating Jx
    
    df2_x2 = (2*L**2*b+R**2*b)/(-m*R**2-M*R**2-2*L**2*M)
    df2_x3 = (2*g*L**2*m)/(2*L**2*M+m*R**2+M*R**2)
    
    df4_x2 = (-2*L*b)/(-m*R**2-M*R**2-2*L**2*M)
    df4_x3 = (-2*g*L*m-2*g*L*M)/(2*L**2*M+m*R**2+M*R**2)
      
    Jx = np.array([[0,   1,  0,  0],
                   [0, df2_x2, df2_x3, 0],
                   [0,   0,  0,  1],
                   [0, df4_x2, df4_x3, 0]
                    ])
    
    df2_u1 = (2*L**2+R**2)/(2*L**2*M+m*R**2+M*R**2)
    df4_u1 = (-2*L)/(2*L**2*M+m*R**2+M*R**2)
    
    
    # Calculating Ju
    
    Ju = np.array([[0.],
                   [df2_u1],
                   [0.],
                   [df4_u1]])
    
    # C and D Space state arrays
    C = np.array([[1,0,0,0],[0,0,1,0]])
    D = np.array([[0],[0]])
    
    # Creating Space state
    linearizedSystem_Taylor = control.ss(Jx,Ju,C,D)
    
    # Transfer function
    G = control.ss2tf(linearizedSystem_Taylor)
     
    # Removing values near to 0
    den = np.around(G .den,decimals=12)
    num = np.around(G .num,decimals=12)
    G  = control.tf(num,den)

    # Controlability Analysis
    control_matrix_taylor = control.ctrb(Jx, Ju)
    control_matrix_taylor_rank = np.linalg.matrix_rank(control_matrix_taylor)
    n_A_taylor = np.shape(Jx[0])
    
    # Observabilty analysis
    obs_matrix_taylor = control.obsv(Jx, C)
    obs_matrix_taylor_rank = np.linalg.matrix_rank(obs_matrix_taylor)
    
    
    #%% Linearized system using small angles approximation
    # We stabilize the system around theta = pi -> tetha = pi + phi, where phi is de deviation
    # from the equilibrium point theta = pi.
    
    # We will assume that the system stays within a small neighborhood of this equillbrium in theta = pi
    # Where: sin(theta) = sin(pi + phi)~ -phi, cos(theta) = cos(pi + phi) ~ -1 and theta_dot² = pi² * phi_dot² ~ 0
     
    # System constants
    # Where dot_x2 = -b*gamma*x_2 + beta*x_3+gamma*u
    
    gamma = (I+m*L**2)/(I*(M+m)+M*m*L**2)
    beta = (m**2*L**2*g)/(I*(M+m)+M*m*L**2)
    
    # Where dot_x4 = -b*phi*x_2 + alpha*x_3+phi*u
    
    tau = (m*L)/(I*(M+m)+M*m*L**2)
    alpha = (m*g*L*(m+M))/(I*(M+m)+M*m*L**2)

    # Creating Space state
    A = np.array([[0,1,0,0],[0,-b*gamma,beta,0],[0,0,0,1],[0,-b*tau,alpha,0]])
    B = np.array([[0],[gamma],[0],[tau]])
    C = np.array([[1,0,0,0],[0,0,1,0]])
    D = np.array([[0],[0]])
    
    linearizedSystem_angles = control.ss(A,B,C,D)

    # Transfer function (it is an array with 2 tranfer functions)
    H = control.ss2tf(linearizedSystem_angles)
 
    # Removing values near to 0
    den = np.around(H.den,decimals=12)
    num = np.around(H.num,decimals=12)
    H = control.tf(num,den)
    
    # MIRAR COMO LO PODEMOS SEPARAR: We divide the array H in two transfer functions
    #H_x = control.tf(np.real(H(1)),np.real(H(3)))
    #H_phi = control.tf(np.real(H(2)),np.real(H(4)))
       
    # Controlability Analysis
    control_matrix_angles = control.ctrb(A, B)
    control_matrix_angles_rank = np.linalg.matrix_rank(control_matrix_angles)
    n_A_angles = np.shape(A[0])
   
    # Observabilty analysis
    obs_matrix_angles = control.obsv(A, C)
    obs_matrix_angles_rank = np.linalg.matrix_rank(obs_matrix_angles)
     
    #%% System response to step function
     
    #_,yout_taylor = control.step_response(linearizedSystem_Taylor,T=t,X0=xRef)
    _,yout_angles = control.step_response(linearizedSystem_angles,T=t)
    
    # Evolution of x
     
    plt.figure()
    #plt.plot(t,xNL[:,0], label='Non-lineal')
    #plt.plot(t,np.transpose(yout_taylor[0]),label='Linealized by Taylor')
    plt.plot(t,np.transpose(yout_angles[0]),label='Linealized (small angles)')
    plt.title('X variation along time')
    plt.legend(loc='best', shadow=True, framealpha=1)
    plt.grid(alpha=0.3)
    plt.ylabel('x (m)')
    plt.xlabel('time (s)')
    plt.show()
    
    # Evolution of phi

    plt.figure()
    #plt.plot(t,xNL[:,2], label='Non-lineal')
    #plt.plot(t,np.transpose(yout_taylor[1]),label='Linealized by Taylor')
   
    plt.plot(t,np.transpose(yout_angles[1]),label='Linealized (small angles)')
    plt.title('$\\phi$ variation along time')
    plt.legend(loc='best', shadow=True, framealpha=1)
    plt.grid(alpha=0.3)
    plt.ylabel('$\\phi$ (rad)')
    plt.xlabel('time (s)')
    plt.show()
    
    #%% Stability analysis 
    
    # G is the system transfer function using linealized system with Taylor
    # H is the system transfer function using linealized system with small angles
    
    #print(f'Lineal system (Taylor):\n TF is = {G}')
    print(f'\nLineal system (small angles):\n TF is = {H}')
    
    # Poles
    
    #G_poles = G.pole()
    #print(f'Lineal system (Taylor):\n Poles at: {G_poles}')
    
    H_poles = H.pole()
    print(f'\nLineal system ((small angles) Poles at:\n {H_poles}')
   
    #%% Control analysis
    
    """# Controlability in linealized System (Taylor)
    print('Linealized system (Taylor) is:\n')
    if n_A_taylor == control_matrix_taylor_rank:
        print('  Controllable system\n')
    else:
        print('  Non-Controllable system\n')"""
        
    # Controlability in linealized System (small angles)

    print('Linealized system (small angles) is:\n')
    if n_A_angles == control_matrix_angles_rank:
      print('  Controllable system\n')
    else:
        print('  Non-Controllable system\n')
    

    #%% Observability analysis
    
    """# Observability in linealized System (Taylor)
    print('Linealized system (Taylor) is:\n')
    if n_A_taylor == obs_matrix_taylor_rank:
        print('  Fully Observable system\n')
    else:
        print('  Not fully observable system\n')"""
        
    # Observability in linealized System (small angles)

    print('Linealized system (small angles) is:\n')
    if n_A_angles == obs_matrix_angles_rank:
        print('  Fully Observable system\n')
    else:
        print('  Not fully observable system\n')
    
    
    #%% HACIENDO PROBATINAS: State feedback design
    
    # Cambiamos el único polo positivo: -2*polopositivo
    K = control.place(A,B,[-2*H_poles[0],H_poles[1],H_poles[2],H_poles[3]])
    K2 = control.place(A,B,[-8*H_poles[3],H_poles[1]+18j,H_poles[1]-18j,H_poles[2]])
    
    # Creating Space state
    A_1 = A - B@K2

    A = A_1
    B = np.array([[0],[gamma],[0],[tau]])
    C = np.array([[1,0,0,0],[0,0,1,0]])
    D = np.array([[0],[0]])
    
    linearizedSystem_new = control.ss(A,B,C,D)

    # Transfer function (it is an array with 2 tranfer functions)
    H_new = control.ss2tf(linearizedSystem_new)
    
    # System response vs step (with feedback)
    t_new = np.linspace(0,35,50000)
    _,yout_new = control.step_response(linearizedSystem_new,T=t_new)
    
    # Plotting
    plt.figure()
    plt.plot(t_new,np.transpose(yout_new[0]),label='Linealized (small angles)')
    plt.title('X variation along time')
    plt.legend(loc='best', shadow=True, framealpha=1)
    plt.grid(alpha=0.3)
    plt.ylabel('x (m)')
    plt.xlabel('time (s)')
    plt.show()
    
    # Evolution of phi
    plt.figure()
    plt.plot(t_new,np.transpose(yout_new[1]),label='Linealized (small angles)')
    plt.title('$\\phi$ variation along time')

    plt.legend(loc='best', shadow=True, framealpha=1)
    plt.grid(alpha=0.3)
    plt.ylabel('$\\phi$ (rad)')
    plt.xlabel('time (s)')
    plt.show()
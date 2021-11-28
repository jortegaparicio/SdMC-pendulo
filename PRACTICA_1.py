#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 16:41:37 2021

@author: Juan Antonio Ortega Aparicio & César Borao Moratinos

Version History

    v1.0: Basic non-linear system simulation
        v1.1: Added graphical representation
    v2.0: Added linearized symtem simulation
        v2.1: Added transfer function study
        v2.2: Added controlabillity and 
        v2.3: Added feedback control
        
"""

# Inverted pendulum analysis

import numpy as np
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
    
    # System Parameters (parametrized)
    
    u = 1      # Similar to a step reponse
    g = 9.81    
    L = 0.3
    m = 0.2
    I = 0.006
    M = 0.5
    b = 0.1
    
    R = np.sqrt(2*I/m)
      
    # Time vector
    t = np.linspace(0,1,50000)
    
    xRef = np.array([0.0,   # x_0 
                     0.0,   # dot_x_0
                     np.pi,   # theta_0
                     0.0])  # dot_theta_0
    
    deltaX   = np.array([0.0,   # x_init 
                         0.0,    # dot_x_init
                         0.0,   # theta_init
                         0.0])    # dot_theta_init
    
    #%% Calculating Non linear sistem (EDOs)
      
    xNL = odeint(f, xRef + deltaX, t, args=([M,m,L,b,u,R,g],))

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
    
    linearizedSystem = control.ss(A,B,C,D)
     
    #%% Comparison between linealized system and non-lineal system.
    # System response to step function.
    
    # We choose a short temporal interval (1 second)
    # It is enough to analyze the difference
    t1 = np.linspace(0,1,500)
    
    # non-lineal system response to step function
    yout_nonlineal = odeint(f, xRef + deltaX, t1, args=([M,m,L,b,u,R,g],))
    
    # linealized system response to step function
    _, yout_linealized = control.step_response(linearizedSystem,T=t1)
    
    # Evolution of theta: non-lineal system
    plt.figure()
    plt.plot(t1,yout_nonlineal[:,2], label='Non-lineal')
    plt.title('$\\theta$ variation along time')
    plt.legend(loc='best', shadow=True, framealpha=1)
    plt.grid(alpha=0.3)
    plt.ylabel('$\\theta$ (rad)')
    plt.xlabel('time (s)')
    plt.show()
    
    
    # Evolution of x and phi: linealized system
    plt.figure()
    plt.plot(t1,np.transpose(yout_linealized[0]),label='Linealized')
    plt.title('X variation along time in Linealized system')
    plt.legend(loc='best', shadow=True, framealpha=1)
    plt.grid(alpha=0.3)
    plt.ylabel('x (m)')
    plt.xlabel('time (s)')
    plt.show()
    
    plt.figure()
    plt.plot(t1,np.transpose(yout_linealized[0]),label='Linealized')
    plt.title('$\\phi$ variation along time in Linealized system')
    plt.legend(loc='best', shadow=True, framealpha=1)
    plt.grid(alpha=0.3)
    plt.ylabel('$\\phi$ (rad)')
    plt.xlabel('time (s)')
    plt.show()
    
    # Comparison between systems
    
    # Evolution of x. Comparison between systems
    plt.figure()
    plt.plot(t1,yout_nonlineal[:,0], label='Non-lineal')
    plt.plot(t1,np.transpose(yout_linealized[0]),label='Linealized')
    plt.title('X variation along time')
    plt.legend(loc='best', shadow=True, framealpha=1)
    plt.grid(alpha=0.3)
    plt.ylabel('x (m)')
    plt.xlabel('time (s)')
    plt.show()
    
    # Evolution of phi. Comparison between systems
    plt.figure()
    plt.plot(t1,yout_nonlineal[:,2]-np.pi, label='Non-lineal')
    plt.plot(t1,np.transpose(yout_linealized[1]),label='Linealized')
    plt.title('$\\phi$ variation along time')
    plt.legend(loc='best', shadow=True, framealpha=1)
    plt.grid(alpha=0.3)
    plt.ylabel('$\\phi$ (rad)')
    plt.xlabel('time (s)')
    plt.show()
    
    #%%
    """# Evolution of x
    plt.figure()
    plt.plot(t1,xNL1[:,0], label='Non-lineal')
    plt.plot(t1,np.transpose(yout_linealized[0]),label='Linealized (small angles)')
    plt.title('X variation along time')
    plt.legend(loc='best', shadow=True, framealpha=1)
    plt.grid(alpha=0.3)
    plt.ylabel('x (m)')
    plt.xlabel('time (s)')
    plt.show()
    
    # Evolution of phi
    plt.figure()
    plt.plot(t,xNL[:,2]-np.pi, label='Non-lineal')
    plt.plot(t,np.transpose(yout_angles[1]),label='Linealized (small angles)')
    #plt.axhline(0.7,linewidth=0.7, color='r')
    plt.title('$\\phi$ variation along time')
    plt.legend(loc='best', shadow=True, framealpha=1)
    plt.grid(alpha=0.3)
    #plt.ylim(0,0.4)
    #plt.xlim(0.30,0.35)
    plt.ylabel('$\\phi$ (rad)')
    plt.xlabel('time (s)')
    plt.show()"""
    
    #%% Stability analysis in linealized system.
    
    # Transfer function (it is an array with 2 tranfer functions)
    H = control.ss2tf(linearizedSystem)
 
    # Removing values near to 0
    den = np.around(H.den,decimals=12)
    num = np.around(H.num,decimals=12)
    H = control.tf(num,den)
    H_poles = H.pole()
    
    print(f'\nLinealized system transfer function is:\n{H}')
    print(f'\nLinealized system Poles at:\n {H_poles}')
   
    #%% Controllability analysis. Linealized System
    
    # Calculating Pc
    Pc = control.ctrb(A, B)
    Pc_rank = np.linalg.matrix_rank(Pc)
    n = np.shape(A[0])
   
    print('\nLinealized system is a: ')
    if n == Pc_rank:
      print('  Controllable system\n')
    else:
        print('  Non-Controllable system\n')
    
    #%% Observability analysis. Linealized System 
    
    # Calculating Po
    Po = control.obsv(A, C)
    Po_rank = np.linalg.matrix_rank(Po)
    n = np.shape(A[0])
     
    print('Linealized system is a: ')
    if n == Po_rank:
        print('  Fully Observable system\n')
    else:
        print('  Not fully observable system\n')
    
    
    #%% State feedback design in linealized system
    
    # K configurations
    
    #K = control.place(A,B,[-2*H_poles[0],H_poles[1],H_poles[2],H_poles[3]])
    #K = control.place(A,B,[1.2*H_poles[1],H_poles[1],H_poles[2],-1])
    K = control.place(A,B,[H_poles[1]+60j,H_poles[1]-60j,H_poles[2],0])
   
    # Creating Space state
    A_1 = A - B@K

    # Creating new State Space
    A = A_1
    B = np.array([[0],[gamma],[0],[tau]])
    C = np.array([[1,0,0,0],[0,0,1,0]])
    D = np.array([[0],[0]])
    
    linearizedSystem_new = control.ss(A,B,C,D)

    # Transfer function (it is an array with 2 tranfer functions)
    H_new = control.ss2tf(linearizedSystem_new)
    
    # System response vs step (with feedback)
    t_new = np.linspace(0,50,50000)
    _,yout_new = control.impulse_response(linearizedSystem_new,T=t_new)
    
    
    # Plotting
    
    # Evolution of X
    plt.figure()
    plt.plot(t_new,np.transpose(yout_new[0]),label='Linealized System')
    plt.title('X variation along time')
    plt.legend(loc='best', shadow=True, framealpha=1)
    plt.grid(alpha=0.3)
    plt.ylabel('x (m)')
    plt.xlabel('time (s)')
    plt.show()
    
    # Evolution of phi
    plt.figure()
    plt.plot(t_new,np.transpose(yout_new[1]),label='Linealized System')
    plt.title('$\\phi$ variation along time')

    plt.legend(loc='best', shadow=True, framealpha=1)
    plt.grid(alpha=0.3)
    plt.ylabel('$\\phi$ (rad)')
    plt.xlabel('time (s)')
    plt.show()
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 11:25:17 2022
A compliation of 1D analytical solutions and functions that are colled by 
several other scripts in this folder


### NOTE ON IMPLEMENTATION ####
It may be convient to place this script in one folder and have a number of 
different projects in different folders that call these functions. To call
these functions from a different folder at the following to your preamble:
    
from sys import path
path.append('path to files\ADE_analytical_solutions\ADE_analytical_solution_functions')
from analytical_solution_functions import plustwo, ADEwReactions_type1_fun, ...

You only need to import the specific functions that you call in a particular script

@author: Czahasky
"""

from scipy.special import erfc as erfc
from math import pi as pi
# Import the modified bessel function of order 0
from scipy.special import i0

import numpy as np


# function to test calling setup from other files
def plustwo(n):
    out = n + 2
    return out


# Retardation with 1st type inlet BC, infinite length (equation C5 in Van Genuchten and Alves)
# NOTE that the zero order terms have been omitted
# 'u' term identical in equation c5 and c6 (type 3 inlet)
# x, t, v, d, first order attachment, first order detachment, linear adsorption term
def ADEwReactions_type1_fun(x, t, v, D, porosity, ka, kd, kf, gamma, rho, C0, t0, Ci):
    # calculate retardation
    R = 1 + rho*kf/porosity
    # define the lumped first order attachment coefficient
    if kf > 0:
        mu = ka + kd * rho *kf/porosity
    else:
        mu = ka + kd * rho/porosity
        
    # 'u' term identical in equation c5 and c6 (type 3 inlet)
    u = v*(1+(4*mu*D/v**2))**(1/2)
    # Infinite length, type 1 inlet
    # Note that the '\' means continued on the next line
    Atrf = np.exp(-mu*t/R)*(1- (1/2)* \
        erfc((R*x - v*t)/(2*(D*R*t)**(1/2))) - \
        (1/2)*np.exp(v*x/D)*erfc((R*x + v*t)/(2*(D*R*t)**(1/2))))
    # term with B(x, t)
    Btrf = 1/2*np.exp((v-u)*x/(2*D))* \
        erfc((R*x - u*t)/(2*(D*R*t)**(1/2))) \
        + 1/2*np.exp((v+u)*x/(2*D))* erfc((R*x + u*t)/ \
        (2*(D*R*t)**(1/2)))
            
    # if a pulse type injection
    if t0 > 0:
        tt0 = t - t0
        indices_below_zero = tt0 <= 0
        # set values equal to 1 (but this could be anything)
        tt0[indices_below_zero] = 1
    
        Bttrf = 1/2*np.exp((v-u)*x/(2*D))* \
            erfc((R*x - u*tt0)/(2*(D*R*tt0)**(1/2))) \
            + 1/2*np.exp((v+u)*x/(2*D))* erfc((R*x + u*tt0)/ \
            (2*(D*R*tt0)**(1/2)))
        
        # Now set concentration at those negative times equal to 0
        Bttrf[indices_below_zero] = 0
        C_out = Ci*Atrf + C0*Btrf - C0*Bttrf
        
    else: # if a continous injection then ignore the Bttrf term (no superposition)
        C_out = Ci*Atrf + C0*Btrf
    # Return the concentration (C) from this function
    return C_out

# Retardation with 3rd type inlet BC, infinite length (equation C5 in Van Genuchten and Alves)
# NOTE that the zero order terms have been omitted
def ADEwReactions_type3_fun(x, t, v, D, porosity, ka, kd, kf, gamma, rho, C0, t0, Ci):
    # calculate retardation
    R = 1 + rho*kf/porosity
    # define the lumped first order attachment coefficient
    if kf > 0:
        mu = ka + kd * rho *kf/porosity
    else:
        mu = ka + kd * rho/porosity
     
    # 'u' term identical in equation c5 and c6 (type 3 inlet)
    u = v*(1+(4*mu*D/v**2))**(1/2)

    if mu == 0:
        # Infinite length, type 3 inlet  
        Atrv = (1- (1/2)*erfc((R*x - v*t)/(2*(D*R*t)**(1/2))) - \
            (v**2*t/(pi*D*R))**(1/2)*np.exp(-(R*x - v*t)**2/(4*D*R*t)) + \
            (1/2)*(1 + v*x/D + v**2*t/(D*R))*np.exp(v*x/D)* \
            erfc((R*x + v*t)/(2*(D*R*t)**(1/2))))
        # term with B(x, t)
        Btrv = (v/(v+u))*np.exp((v-u)*x/(2*D))* erfc((R*x - u*t)/(2*(D*R*t)**(1/2))) 
        
    else: 
        # Infinite length, type 3 inlet  
        Atrv = np.exp(-mu*t/R)*(1- (1/2)*erfc((R*x - v*t)/(2*(D*R*t)**(1/2))) - \
            (v**2*t/(pi*D*R))**(1/2)*np.exp(-(R*x - v*t)**2/(4*D*R*t)) + \
            (1/2)*(1 + v*x/D + v**2*t/(D*R))*np.exp(v*x/D)* \
            erfc((R*x + v*t)/(2*(D*R*t)**(1/2))))
        # term with B(x, t)
        Btrv = (v/(v+u))*np.exp((v-u)*x/(2*D))* erfc((R*x - u*t)/(2*(D*R*t)**(1/2))) + \
            (v/(v-u))*np.exp((v+u)*x/(2*D))* erfc((R*x + u*t)/(2*(D*R*t)**(1/2))) + \
            (v**2/(2*mu*D))*np.exp((v*x/D) - (mu*t/R))* \
            erfc((R*x + v*t)/(2*(D*R*t)**(1/2)))
    
    # if a pulse type injection
    if t0 > 0:
        tt0 = t - t0
        # 
        indices_below_zero = tt0 <= 0
        
        if np.sum(indices_below_zero) > 0:
            # set values equal to 1 (but this could be anything)
            tt0[indices_below_zero] = 1
        
        if mu == 0:
            Bttrv = (v/(v+u))*np.exp((v-u)*x/(2*D))* erfc((R*x - u*tt0)/(2*(D*R*tt0)**(1/2))) 
        else:
            Bttrv = (v/(v+u))*np.exp((v-u)*x/(2*D))* erfc((R*x - u*tt0)/(2*(D*R*tt0)**(1/2))) + \
                (v/(v-u))*np.exp((v+u)*x/(2*D))* erfc((R*x + u*tt0)/(2*(D*R*tt0)**(1/2))) + \
                (v**2/(2*mu*D))*np.exp((v*x/D) - (mu*tt0/R))* \
                erfc((R*x + v*tt0)/(2*(D*R*tt0)**(1/2)))

        # Now set concentration at those negative times equal to 0
        if np.sum(indices_below_zero) > 0:
            Bttrv[indices_below_zero] = 0

        if gamma > 1E-10:
            C_out = gamma/mu + (Ci-gamma/mu)*Atrv + (C0- gamma/mu)*Btrv - C0*Bttrv
        else:
            C_out = Ci*Atrv + C0*Btrv - C0*Bttrv
            
    else: # if a continous injection then ignore the Bttrf term (no superposition)
        if gamma > 1E-10:
            C_out = gamma/mu + (Ci-gamma/mu)*Atrv + (C0- gamma/mu)*Btrv
        else:
            C_out = Ci*Atrv + C0*Btrv
    # Return the concentration (C) from this function
    return C_out
    


# Retardation with 1st type inlet BC, infinite length (equation C5 in Van Genuchten and Alves)
# NOTE that the zero order terms have been omitted
# 'u' term identical in equation c5 and c6 (type 3 inlet)
# x, t, v, d, first order attachment, first order detachment, linear adsorption term
def ADE_bear1979_type1_fun(x, t, v, D, ka, C0):
    # beta
    b = (v**2/(4*D**2) + ka/D)**(1/2)
    # Infinite length, type 1 inlet
    # Note that the '\' means continued on the next line
    C_out = C0/2*np.exp(x*v/(2*D))*(np.exp(-b*x)* \
        erfc((x- (v**2 + 4*ka*D)**(1/2)*t)/(2*(D*t)**(1/2))) + \
        np.exp(b*x)*erfc((x+ (v**2 + 4*ka*D)**(1/2)*t)/(2*(D*t)**(1/2))))
        
    return C_out



# Solve for the aqueous concentration of multiple species we need to use a general transform approach
# (Sun and Clement, 1999) A decomposition method for solved coupled multispecies
# reactive transport problems. Specifically we implement the 2 species solution
# described in Appendix A
# call with something like this: C = ADE_multispecies_fun(L, t, v, D, kc, kd, 1, 0, t0, 1)
def ADE_multispecies_fun(L, t, v, D, k1, k2, C01, C02, t0, bctype):
    # c = P^-1 a
    P = np.array([[1, 0], [(k1/(k1-k2)), 1]])
    P_inv = np.linalg.inv(P)
    # Inverse matrix check (this should print and identity matrix)
    # print(np.matmul(P, P_inv))
    
    # transformed boundary conditions!!!
    a0 = np.matmul(P, [[C01],[C02]])
    # generic parameters for adsorption variable placeholders
    gamma = 0
    rho_b = 2.6
    porosity = 0.3
    if bctype == 1:
        a1 = ADEwReactions_type1_fun(L, t, v, D, porosity, k1, 0, 0.0, gamma, rho_b, a0[0], t0, 0)
        a2 = ADEwReactions_type1_fun(L, t, v, D, porosity, k2, 0, 0., gamma, rho_b, a0[1], t0, 0)
        
    elif bctype == 3:
        a1 = ADEwReactions_type3_fun(L, t, v, D, porosity, k1, 0, 0.0, gamma, rho_b, a0[0], t0, 0)
        a2 = ADEwReactions_type3_fun(L, t, v, D, porosity, k2, 0, 0., gamma, rho_b, a0[1], t0, 0)
    
    C = np.zeros([2, len(t)])
    for i in range(0, len(t)):
        C[:,i] = np.squeeze(np.matmul(P_inv, [[a1[i]],[a2[i]]]))
        
    return C


# Solution to ade with first-order kinetic attachment (decay) ka AND detachment,
# meaning that the species can return to the aqueous phase at a rate of kd
# Solution is from: Interpreting Deposition Patterns of Microbial Particles in Laboratory-Scale Column Experiments
# 10.1021/es025871i  (Equations 26-28)
def ADE_ka_kd_Lapidus_fun(x, t, v, D, ka, kd, C0, t0):
    # in this formulation kd must be positive
    kd = abs(kd)
    # Equation 28
    d = v**2/(4*D) + ka - kd
    
    # Equation 27 and part of 26    
    def F_function(x, t, v, d, D, ka, kd):
        fi_int = np.zeros(len(t))
        fi_2int = np.zeros(len(t))
        for i in range(0, len(t)):
            tau = t[:i]
            # evaluate the integral in equation 27
            fi_int[i] = np.trapz(i0(2*(ka*kd*tau*(t[i]-tau))**(1/2)) * \
                          x/(2*(pi*D*tau**3)**(1/2)) * np.exp(-x**2/(4*D*tau) - tau*d), tau)
            # evaluate the integral in right term of  equation 26   
            fi_2int[i] = np.trapz(np.exp(-kd*tau)*fi_int[i], tau)
                
        return fi_int, fi_2int
            
    # call the function
    fi_int, fi_2int = F_function(x, t, v, d, D, ka, kd)
    # combine the solution
    atf = np.exp(v*x/(2*D))*(fi_int + kd*fi_2int)
    
    if t0 > 0:
        tt0 = t - t0
        # 
        indices_below_zero = tt0 <= 0
        
        if np.sum(indices_below_zero) > 0:
            # set values equal to 1 (but this could be anything)
            tt0[indices_below_zero] = 1
        # call F_function that evaluates nested integrals
        fi_intt, fi_2intt = F_function(x, tt0, v, d, D, ka, kd)
        
        bttf = np.exp(v*x/(2*D))*(fi_intt + kd*fi_2intt)
        
        # Now set concentration at those negative times equal to 0
        if np.sum(indices_below_zero) > 0:
            bttf[indices_below_zero] = 0
        # pulse injection solution    
        C_out = C0*(atf - bttf)
    else:
        # continous injection solution
        C_out = C0*atf
    # return concentration as a function of time
    return C_out
# Here is some testing/demonstration that might be useful to understand the bessel functions and dummy integration variables
# tt = np.linspace(0,5, 100)
# plot the modified bessel function of zero order
# plt.plot(tt, i0(tt)) 
# Dummy variable integration example
# print(tt[-1]**3/6)
# print(np.trapz(tt*(tt[-1]-tt), tt))
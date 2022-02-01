# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 11:25:17 2022
A compliation of 1D analytical solutions and functions that are colled by 
several other scripts in this folder

@author: Czahasky
"""

from scipy.special import erfc as erfc
import numpy as np
import math


# Retardation with 1st type inlet BC, infinite length (equation C5 in Van Genuchten and Alves)
# NOTE that the zero order terms have been omitted
# 'u' term identical in equation c5 and c6 (type 3 inlet)
# x, t, v, d, first order attachment, first order detachment, linear adsorption term
def ADEwReactions_type1_fun(x, t, v, D, porosity, ka, kd, kf, rho, C0, t0, Ci):
    # calculate retardation
    R = 1 + rho*kf/porosity
    # define the lumped first order attachment coefficient
    mu = ka + kd * rho *kf/porosity
     
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
def ADEwReactions_type3_fun(x, t, v, D, porosity, ka, kd, kf, rho, C0, t0, Ci):
    # calculate retardation
    R = 1 + rho*kf/porosity
    # define the lumped first order attachment coefficient
    mu = ka + kd * rho *kf/porosity
     
    # 'u' term identical in equation c5 and c6 (type 3 inlet)
    u = v*(1+(4*mu*D/v**2))**(1/2)
    
    # Infinite length, type 3 inlet  
    Atrv = np.exp(-mu*t/R)*(1- (1/2)*erfc((R*x - v*t)/(2*(D*R*t)**(1/2))) - \
        (v**2*t/(pi*D*R))**(1/2)*np.exp(-(R*x - v*t)**2/(4*D*R*t)) + \
        (1/2)*(1 + v*x/D + v**2*t/(D*R))*np.exp(v*x/D)* \
        erfc((R*x + v*t)/(2*(D*R*t)**(1/2))))
    
    # term with B(x, t)
    Btrv = (v./(v+u)).*exp((v-u).*x./(2*D)).* erfc((R.*x - u.*t)./(2.*(D.*R.*t).^(1/2))) + ...
        (v./(v-u)).*exp((v+u).*x./(2*D)).* erfc((R.*x + u.*t)./(2.*(D.*R.*t).^(1/2))) + ...
        (v.^2/(2*mu*D)).*exp((v.*x./D) - (mu.*t./R)).* ...
        erfc((R.*x + v.*t)./(2.*(D.*R.*t).^(1/2)));
    
    % if continous injection
    if injection_style == 0
        if mu >0
            C_out = (gamma/mu)+ (Ci- gamma/mu).*Atrv + (C0 - gamma/mu).*Btrv;
        else % if mu = 0 then we would get nans
            C_out = (Ci).*Atrv + (C0).*Btrv ;
        end
        
    % else if pulse injection
    elseif injection_style == 1
        % calculate pulse injection length
        % varargin{1} is equal to t_inj
        tt0 = t - varargin{1};
        
        % If you started the 't' array from zero and then subtracted some time
        % of injection then 'tt0' will have negative numbers. Negative numbers
        % in the 'erfc' function will lead to errors so to correct for
        % this I have added the 'real' function inside the 'erfc' functions.
        % After 'Bttrv' is calculated these imaginary/negative values are set to
        % zero
        Bttrv = (v./(v+u)).*exp((v-u).*x./(2*D)).* erfc(real((R.*x - u.*tt0)./(2.*(D.*R.*tt0).^(1/2)))) + ...
            (v./(v-u)).*exp((v+u).*x./(2*D)).* erfc(real((R.*x + u.*tt0)./(2.*(D.*R.*tt0).^(1/2)))) + ...
            (v.^2/(2*mu*D)).*exp((v.*x./D) - (mu.*tt0./R)).* ...
            erfc(real((R.*x + v.*tt0)./(2.*(D.*R.*tt0).^(1/2))));
        
        % Set imaginary values to 0
        Bttrv(tt0 <= 0) = 0;
        
        % calculation concentration via superposition
        C_out = (gamma/mu)+ (Ci- gamma/mu).*Atrv + (C0 - gamma/mu).*Btrv - C0.*Bttrv;
    end
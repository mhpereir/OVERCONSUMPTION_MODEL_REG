import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16, 'xtick.labelsize':14, 'ytick.labelsize':14})

from astropy import units as u
from astropy.cosmology import Planck15 as cosmo, z_at_value

z_init      = 10
z_final     = 0

M0_range   = np.power(10, np.arange(8.7,11.1,0.1))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def delta_Mhalo(Mhalo,z):
    return 510 *  (Mhalo/(10**12))**(1.10) * ((1+z)/3.2)**(2.2)

def epsilon_in(Mh,z):
    if Mh > 1.5e12 or Mh < 1e11:
        return 0
    else:
        #print(z)
        return 0.7 * (np.minimum(z,2.2)/4.4 + 1/2)
    
#epsilon_in = 0.7        #accretion efficiency
f_b        = 0.18        #baryon fraction
alpha      = 1           #mass loading factor
R          = 0.56        #chambrier return factor
def delta_Mgas(delta_Mh, SFR, Mh, z):
    M_in  = epsilon_in(Mh,z) * f_b * delta_Mh
    M_out = (1 - R + alpha) * SFR 
    
    return M_in - M_out

epsilon_sfr = 0.02
def SFR(M_gas,z):
    t_dyn = 2e7 * ((1+z)/3.2)**(-1.5)
    return epsilon_sfr * M_gas / t_dyn
 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def sSFR_schreiber(logMs, z):       # Schreiber+2015
    Ms  = np.power(10,logMs)
    r   = np.log10(1+z)
    m   = logMs - 9
    
    n   = logMs.shape
    
    m_0 = np.zeros(n) + 0.50
    m_1 = np.zeros(n) + 0.36
    a_0 = np.zeros(n) + 1.50
    a_1 = np.zeros(n) + 0.30
    a_2 = np.zeros(n) + 2.50
            
    lsfr = m - m_0 + a_0 * r - a_1 * np.maximum(np.zeros(n), m-m_1-a_2 * r)**2
    ssfr = 10**(lsfr) / Ms
    return ssfr ## per year

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def sSFR_speagle(logMs, z):
    z_temp     = np.minimum(z,6)
    logMs_temp = np.maximum(logMs, 8.5) 
    
    t   = cosmo.lookback_time(9999).value - cosmo.lookback_time(z_temp).value  ##  WARNING:: TRY cosmo.age(z).value instead!!
    x_1 = 0.840
    x_2 = 0.026
    x_3 = 6.510
    x_4 = 0.110
    
    logSFR = (x_1 - x_2*t) * logMs_temp - (x_3 - x_4*t)
    ssfr   = np.power(10, logSFR) / np.power(10,logMs_temp)
    return  ssfr ## peryear

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

### Stellar mass relationship from Behroozi 2013
def M_star(M_halo,z):  # input in Msol
    a   = 1/(1+z)
    v   = np.exp(-4*a*a)
    
    logEps = -1.777 + (-0.006*(a-1) + (-0.000)*z)*v - 0.119*(a-1)
    eps    = np.power(10, logEps)
    
    logM1 = 11.514 + (-1.793*(a-1) + (-0.251)*z)*v 
    m1    = np.power(10,logM1)
    
    logM_star = np.log10(eps * m1) + f(np.log10(M_halo/m1), z,a,v) - f(0, z,a,v)
    return np.power(10, logM_star)        

def f(x,z,a,v):
    alpha = -1.412 + (0.731*(a-1))*v
    gamma = 0.316 + (1.319*(a-1) + 0.279*z)*v
    delta = 3.508 + (2.608*(a-1)+(-0.043)*z)*v
    return -np.log10(np.power(10,alpha*x) + 1) + delta * np.power(np.log10(1+np.exp(x)), gamma) / (1 + np.exp(np.power(10,-x)))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


fig1,ax1 = plt.subplots(tight_layout=True)
fig2,ax2 = plt.subplots(tight_layout=True)
for M0 in M0_range:
    Mg = 0
    Ms = 0
    inits = np.array([M0, Mg, Ms])
    
    t_init  = cosmo.age(z_init).value
    t_final = cosmo.age(z_final).value
    t       = t_init
    dt      = 0.01
    
    print(np.log10(M0))
    
    while t < t_final:
        z = z_at_value(cosmo.age,t*u.Gyr, zmin=-1e-6)
        
        dMh = delta_Mhalo(inits[0],z) *dt*1e9
        
        print(np.log10(dMh))
        
        sfr = SFR(inits[1], z)        *dt*1e9
        dMg = delta_Mgas(dMh, sfr, inits[0],z) #   *dt*1e9
        
        if t+dt > t_final:
            dt = t_final - t
        else:
            pass
        
        #print(inits, z)
        #print(dMh, dMg, sfr)
        inits = inits + np.array([dMh, dMg, sfr*(1-R)])
        #print('~~~~~~~~~~~~~~')
            
        t += dt
        
    ax1.scatter(np.log10(inits[2]), np.log10(SFR(inits[1], z_final)/inits[2]), color='k')#, label='Regulator Model')
    ax2.scatter(np.log10(inits[0]), inits[2]/inits[0], color='k')
    
logMs_range = np.arange(8.5,11.3,0.02)
ax1.plot(logMs_range, np.log10(sSFR_speagle(logMs_range, z_final)), color='r', label='Speagle')
ax1.plot(logMs_range, np.log10(sSFR_schreiber(logMs_range, z_final)), color='b', label='Schreiber')
fig1.legend(bbox_to_anchor=(0.55,0.45))
ax1.set_title('z = {}'.format(z_final))
ax1.set_ylabel('sSFR [1/yr]')
ax1.set_xlabel('Stellar Mass [$M_*$]')
ax1.set_xlim([8.5,11.3])

logMh_range  = np.arange(8,15,0.02)
Mh_range     = np.power(10, logMh_range)
Mstar_beh    = M_star(np.power(10, logMh_range), z_final)
logMs_beh    = np.log10(Mstar_beh)
ax2.plot(logMh_range, Mstar_beh/Mh_range, color='r', label='Behroozi')
fig2.legend(bbox_to_anchor=(0.65,0.30))
ax2.set_title('z = {}'.format(z_final))
ax2.set_ylabel('$f_*$')
ax2.set_xlabel('Halo Mass [$M_*$]')
ax2.set_xlim([9,15])

plt.show()
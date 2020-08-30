import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
plt.rcParams.update({'font.size': 16, 'xtick.labelsize':14, 'ytick.labelsize':14})

from matplotlib import gridspec
from matplotlib import cm

from scipy.interpolate import interp1d, interp2d
from scipy.optimize import newton_krylov, root_scalar, fsolve
from scipy.integrate import quad
from scipy import stats

from astropy import units as u
from astropy.cosmology import Planck15 as cosmo, z_at_value
#from img_scale import linear

import time

class overconsumption:
    def __init__(self):
        print('Initializing function')
        start = time.time()
        self.f_bar   = 0.17
        self.f_strip = 0
        self.R       = 0.4
        
        self.omega_m = 0.3
        self.omega_b = 0.044
        self.h       = 0.7
        self.sigma_8 = 1.0        
        
        self.M_min   = 9.0
        self.M_max   = 14.0
        
        # Parameters of equation for SSFR from Behroozi 2013
        self.Masses1  = [9, 9.5, 10, 10.5]
        self.A1       = [-1.028,  -0.993,   -1.219,   -1.426]
        self.B1       = [-0.060,  -0.080,   -0.023,   -0.083]
        self.C1       = [3.135e-9, 2.169e-9, 1.873e-9, 1.129e-9]
        
        # Parameters of equation for SMF from Muzzin 2013       
        z_bins = [ 0.35,  0.75,  1.25,  1.75,  2.25,  2.75, 3.5, 10]
        
        M_S_sf    = [10.75, 10.82, 10.82, 10.91, 11.03, 11.14, 11.47, 11.47]
        Phi_s_sf  = np.array([13.58, 10.95,  7.20,  4.49,  2.01,  1.09,  0.09,  0.09])*1e-4
        Alpha_sf  = [-1.3,   -1.3,  -1.3,  -1.3,  -1.3,  -1.3,  -1.3,  -1.3]
        
        M_S_q    = [10.75, 10.84, 10.83, 10.80, 10.79, 10.81, 11.00, 11.00]
        Phi_s_q  = np.array([30.65, 14.38,  7.48,  3.61,  1.14,  0.66,  0.05,  0.05])*1e-4
        Alpha_q  = [-0.4,   -0.4,  -0.4,  -0.4,  -0.4,  -0.4,  -0.4,  -0.4]
        
        M_S_a    = [11.06, 11.04, 10.99, 10.96, 11.00, 11.09, 11.49, 11.49]
        Phi_s_a  = np.array([19.02, 14.48,  9.30,  6.33,  2.94,  1.66,  0.13,  0.13])*1e-4
        Alpha_a  = [-1.2,   -1.2,  -1.2,  -1.2,  -1.2,  -1.2,  -1.2,  -1.2]
        
        self.M_S_sf    = interp1d(z_bins, M_S_sf,   kind='linear', fill_value="extrapolate")
        self.Phi_S_sf  = interp1d(z_bins, Phi_s_sf, kind='linear', fill_value="extrapolate")
        self.Alpha_sf  = interp1d(z_bins, Alpha_sf, kind='linear', fill_value="extrapolate")
        
        self.M_S_q    = interp1d(z_bins, M_S_q,   kind='linear', fill_value="extrapolate")
        self.Phi_S_q  = interp1d(z_bins, Phi_s_q, kind='linear', fill_value="extrapolate")
        self.Alpha_q  = interp1d(z_bins, Alpha_q, kind='linear', fill_value="extrapolate")
        
        self.M_S_a    = interp1d(z_bins, M_S_a,   kind='linear', fill_value="extrapolate")
        self.Phi_S_a  = interp1d(z_bins, Phi_s_a, kind='linear', fill_value="extrapolate")
        self.Alpha_a  = interp1d(z_bins, Alpha_a, kind='linear', fill_value="extrapolate")
        
        print('Ellapsed time: {}'.format(time.time()-start))
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    
    def gen_cluster(self, z_0, z_max, logM0, eta, show_plots):
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('Generating cluster with :')
        print('z_0 = {}; logM0 = {}; eta = {}'.format(z_0, logM0, eta))
        
        delta_z      = 0.05
        self.eta     = eta
        self.z_0     = z_0
        self.logM0   = logM0
        self.z_range = np.arange(self.z_0, z_max, delta_z)
        pool         = mp.Pool(processes=3)
        
        self.delta_m = 0.05
        
        self.logMStar_setup = np.arange(3,12,0.1)
        self.logMHalo_bins  = np.arange(self.M_min,self.M_max,self.delta_m)
        self.logMHalo_eff   = np.array([(self.logMHalo_bins[i]+self.logMHalo_bins[i+1])/2 for i in range(0,len(self.logMHalo_bins)-1)])
        
        self.logMStar_bins  = np.array([np.log10(self.M_star(np.power(10,self.logMHalo_bins),z)) for z in self.z_range])
        self.logMStar_eff   = np.array([[(self.logMStar_bins[i,j]+self.logMStar_bins[i,j+1])/2 for j in range(0,len(self.logMHalo_bins)-1)] for i,z in enumerate(self.z_range)])
        
        self.logMStar_low   = np.array([m for m in self.logMStar_bins[:,:-1]])
        self.logMStar_upp   = np.array([m for m in self.logMStar_bins[:,1:]])
                
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
        phi_init_sf   = self.phi_sf_3(self.logMStar_setup)
        phi_init_q    = np.zeros(len(phi_init_sf))
        
        phi_sf_z_10 = []
        phi_q_z_10  = []
        z_eff       = []
        for i in range(1,len(self.z_range)):
            z       = self.z_range[-i]
            z_eff.append((z + self.z_range[-i-1])/2)
            dt      = cosmo.lookback_time(z).value - cosmo.lookback_time(self.z_range[-i-1]).value
            
            dn_blue = self.dN_blue(self.logMStar_setup, z, phi_init_sf) * dt
            dn_red  = self.dN_red(self.logMStar_setup, z, phi_init_sf, phi_init_q) * dt
                        
            phi_init_sf += dn_blue
            phi_init_q  += dn_red
            
            phi_sf_z_10.append(np.copy(phi_init_sf))
            phi_q_z_10.append(np.copy(phi_init_q))
        
        z_midp                  = np.array(z_eff)
        self.phi_sf_z_10_interp = interp2d(self.logMStar_setup, z_midp[z_midp<=z_max], phi_sf_z_10)
        self.phi_q_z_10_interp  = interp2d(self.logMStar_setup, z_midp[z_midp<=z_max], phi_q_z_10)
        
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
        phi_init_sf = self.phi_sf_3(self.logMStar_setup)
        phi_init_q  = np.zeros(len(phi_init_sf))
        
        phi_sf_z_4  = []
        phi_q_z_4   = []
        for i in range(1,len(self.z_range[self.z_range<=10+delta_z/2])):
            z       = self.z_range[-i]
            dt      = cosmo.lookback_time(z).value - cosmo.lookback_time(self.z_range[-i-1]).value
            
            dn_blue = self.dN_blue(self.logMStar_setup, z, phi_init_sf) * dt
            dn_red  = self.dN_red(self.logMStar_setup, z, phi_init_sf, phi_init_q) * dt
                        
            phi_init_sf += dn_blue
            phi_init_q  += dn_red
            
            phi_sf_z_4.append(np.copy(phi_init_sf))
            phi_q_z_4.append(np.copy(phi_init_q))
        
        
        self.phi_sf_z_4_interp = interp2d(self.logMStar_setup, z_midp[z_midp<=10], phi_sf_z_4)
        self.phi_q_z_4_interp  = interp2d(self.logMStar_setup, z_midp[z_midp<=10], phi_q_z_4)
        
        
        Halo_masses = np.array(pool.map(self.M_main, (z for z in self.z_range) ))
        delta_halos = [np.log10(np.power(10,Halo_masses[i]) - np.power(10,Halo_masses[i+1])) for i in range(0,len(self.z_range)-1)]
        delta_halos.append(Halo_masses[-1])
        delta_halos = np.array(delta_halos)
        
        self.A      = np.power(10,delta_halos) / np.array(pool.map(self.func_A_normal, (z for z in self.z_range)))
        
        
        self.delta_N_sf_infall = np.array(pool.map(self.func_delta_N_sf, ([z,i] for i,z in enumerate(self.z_range))) )
        self.delta_N_q_infall  = np.array(pool.map(self.func_delta_N_q, ([z,i] for i,z in enumerate(self.z_range))) )
        self.delta_N_a_infall  = self.delta_N_q_infall + self.delta_N_sf_infall
        
        self.delta_td          = np.array(pool.map(self.func_delta_td, (m for m in self.logMHalo_bins)) ).T
        
        out                    = np.array(pool.map(self.func_delta_N_sf_final, ([z,self.delta_td[i,:],i] for i,z in enumerate(self.z_range)) ))
        self.delta_N_sf_final, self.delta_N_q_final = out[:,0,:], out[:,1,:]
        self.delta_N_a_final   = self.delta_N_sf_final + self.delta_N_q_final
        
        logMs_final      = np.arange(8,12,0.1)
        
        N_sf_per_mass    = self.group_by_mass(self.delta_N_sf_final, logMs_final)  
        N_q_per_mass     = self.group_by_mass(self.delta_N_q_final, logMs_final)            
        N_a_per_mass     = N_sf_per_mass + N_q_per_mass
        
        
        N_field_q_per_mass = self.phi_q_z_4_interp(logMs_final, self.z_0)
        N_field_a_per_mass = self.phi_q_z_4_interp(logMs_final, self.z_0) + self.phi_sf_z_4_interp(logMs_final, self.z_0)
        
        pool.close()
        
        if show_plots:    
            ### final stellar masses of galaxies
            #fig,ax = plt.subplots()
            #contourf_ = plt.contourf(self.z_range, self.logMHalo_bins, self.logMStar_bins.T)
            #ax.set_xlabel('redshift')
            #ax.set_ylabel('halo mass')
            #ax.set_title('Stellar Mass')
            #cbar = fig.colorbar(contourf_)
            
            ### verification that first derivative is okay!
            #fig,ax = plt.subplots()
            #ax.semilogy(self.logMHalo_bins, self.M_star(np.power(10,self.logMHalo_bins), z=0.35), 'k')
            #ax2 = ax.twinx()
            #ax2.plot(self.logMHalo_bins, self.dMs_dMh(np.power(10,self.logMHalo_bins),z=0.35), linestyle='--')
            #ax.set_ylabel('Stellar Mass [log($M_*/M_\odot$)]')
            #ax.set_xlabel('log Halo Mass')
            #ax2.set_ylabel('$dM_{*}/dMh$')
            
            ## simple plot of phi
            fig,ax = plt.subplots()
            muzz_q    = self.phi_q_2(logMs_final, z=1.2)
            muzz_sf   = self.phi_sf_2(logMs_final, z=1.2) 
            peng_f    = self.phi_q_z_4_interp(logMs_final, 1.2)
            peng_f_sf = self.phi_sf_z_4_interp(logMs_final,1.2)
            #peng_c = self.phi_q_z_10_interp(logMs_final, 1.2)
            #ax.semilogy(logMs_final, self.phi_sf_2(logMs_final, z=1.2), label='Muzz, SF', color='b')
            ax.semilogy(logMs_final, muzz_sf, label='Muzz, SF', color='C0')
            ax.semilogy(logMs_final, muzz_q, label='Muzz, Q', color='r')
            #ax.semilogy(logMs_final, self.phi_sf_z_4_interp(logMs_final, 1.2)*1.5e2, label='Peng, SF', color='b', linestyle='--')
            ax.semilogy(logMs_final, peng_f * np.max(muzz_q)/ np.max(peng_f), label='Peng, Q', color='r', linestyle='--')
            ax.semilogy(logMs_final, peng_f_sf * np.max(muzz_q)/ np.max(peng_f), label='Peng, SF', color='C0', linestyle='--')
            #ax.semilogy(logMs_final, peng_c * np.max(muzz_q)/ np.max(peng_c), label='Peng cls, Q', color='r', alpha=0.3)
            ax.set_xlabel('Stellar Mass [log($M_*/M_\odot$)]')
            ax.set_ylabel('$\Phi_{field}$ [z=1.2]')
            ax.set_xlim([8.4,12])
            ax.set_ylim([1e-6,1e-2])
            ax.legend()
            
            ### Contour plot of delay times vs halo mass and redshift
            #fig,ax    = plt.subplots()
            #contourf_ = ax.contourf(self.z_range, self.logMHalo_bins, self.delta_td.T, np.arange(0,14,2), extend='both', cmap=cm.seismic_r)
            #ax.set_xlabel('Redshift [z]')
            #ax.set_ylabel('Halo Mass [log($M_h$/$M_\odot$)]')
            #ax.set_title('$t_{delay}$ [Gyr]')
            #cbar      = fig.colorbar(contourf_)       
            
            
            ### Contour plot of all galaxies vs halo mass and redshift at INFALL
            #fig,ax    = plt.subplots()
            #contourf_ = ax.contourf(self.z_range, self.logMHalo_eff, np.log10(self.delta_N_a_infall.T))#, np.arange(-15,16,3), extend='both', cmap=cm.seismic_r)
            #ax.set_xlabel('Redshift [z]')
            #ax.set_ylabel('Halo Mass [log($M_h$/$M_\odot$)]')
            #ax.set_title('Delta N [Infall]')
            #cbar      = fig.colorbar(contourf_)
            
            
            ### Contour plot of star forming galaxies vs halo mass and redshift
            #fig,ax    = plt.subplots()
            #contourf_ = ax.contourf(self.z_range, self.logMHalo_eff, np.log10(self.delta_N_sf_final.T))#, np.arange(-15,16,3), extend='both', cmap=cm.seismic_r)
            #ax.set_xlabel('Redshift [z]')
            #ax.set_ylabel('Halo Mass [log($M_h$/$M_\odot$)]')
            #ax.set_title('Delta N [SF]')
            #cbar      = fig.colorbar(contourf_)
            
            ### Contour plot of quiescent vs stellar mass and redshift
            #fig,ax    = plt.subplots()
            #contourf_ = ax.contourf(self.z_range, self.logMHalo_eff, np.log10(self.delta_N_q_final.T))#, np.arange(-15,16,3), extend='both', cmap=cm.seismic_r)
            #ax.set_xlabel('Redshift [z]')
            #ax.set_ylabel('Halo Mass [log($M_h$/$M_\odot$)]')
            #ax.set_title('Delta N [Q]')
            #cbar      = fig.colorbar(contourf_)
            
            ### Plot of peng sSFR
            #fig,ax    = plt.subplots()
            #ax.plot(self.z_range, self.SSFR(np.array(10),self.z_range))
            #ax.set_xlabel('Redshift [z]')
            #ax.set_ylabel('sSFR yr$^{-1}$')
            #ax.set_title('[Peng+2010]')
            #plt.show()
                        
            ### Plot of Quenched Fraction per mass
            #fig,ax = plt.subplots()#figsize=(5,4), tight_layout=True, sharey=True)
            #ax.plot(logMs_final, N_q_per_mass/N_a_per_mass, label='Cluster')
            #ax.plot(logMs_final, N_field_q_per_mass/N_field_a_per_mass, label='Field')
            #ax.set_xlabel('Stellar Mass [log($M_*$/$M_\odot$)]')
            #ax.set_ylabel('Quenched Fraction')
            
            #ax.set_xlim([9,12])
            #ax.set_ylim([0,1])
            
            #ax.legend(bbox_to_anchor=(1,0.9))
            
            
            ### Time slice of delay times
            #fig,ax    = plt.subplots()
            
            #ii   = 20
            #temp = cosmo.lookback_time(self.z_range[ii]).value - cosmo.lookback_time(self.z_0).value
            
            #ax.plot(self.logMStar_bins[ii,:], self.delta_td[ii,:])
            #ax.plot([8,12], [temp,temp])
            #ax.set_xlabel('Stellar Mass [$M_*$]')
            #ax.set_ylabel('Delay Times [Gyr]')
            #ax.set_title('z = {:.1f}'.format(self.z_range[ii]))
            #ax.set_xlim([8,12])
            #ax.set_ylim([0,12])
            #plt.show()
            
            
            plt.show()
            plt.close('all')
        
        #FIX TDELAY.....
        
        return (logMs_final, N_q_per_mass,  N_a_per_mass, N_field_q_per_mass, N_field_a_per_mass)
        
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    
    # Stuff to calculate T_DELAY
    
    def M_star(self,M_halo,z):  # input in Msol
        a   = self.a(z)
        v   = self.v(a)
        eps = self.epsilon(z,a,v)
        m1  = self.M1(z,a,v)
        
        logM_star = np.log10(eps * m1) + self.f(np.log10(M_halo/m1), z,a,v) - self.f(0, z,a,v)
        
        return np.power(10, logM_star)
    
    def a(self,z):
        return 1/(1+z)
    
    def v(self,a):
        return np.exp(-4*a*a)
    
    def epsilon(self,z,a,v):
        logEps = -1.777 + (-0.006*(a-1) + (-0.000)*z)*v - 0.119*(a-1)
        return np.power(10, logEps)
    
    def M1(self,z,a,v):
        logM1 = 11.514 + (-1.793*(a-1) + (-0.251)*z)*v 
        return np.power(10,logM1)
    
    def alpha(self,a,v):
        return -1.412 + (0.731*(a-1))*v
    
    def delta(self,z,a,v):
        return 3.508 + (2.608*(a-1)+(-0.043)*z)*v
    
    def gamma(self,z,a,v):
        return 0.316 + (1.319*(a-1) + 0.279*z)*v
    
    def f(self,x,z,a,v):
        alpha = self.alpha(a,v)
        gamma = self.gamma(z,a,v)
        delta = self.delta(z,a,v)
        return -np.log10(np.power(10,alpha*x) + 1) + delta * np.power(np.log10(1+np.exp(x)), gamma) / (1 + np.exp(np.power(10,-x)))
    
   
    def dMs_dMh(self,Mh,z):  #1st order derivative: central limit theorem
        h = 100 # Mh is in Msol h = 1  means dx of 1 solar mass
        return ( self.M_star(Mh + h,z) - self.M_star(Mh - h,z) )/ (2*h) 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    
#     def SSFR(self, logMs,z):      
#         a = np.interp(logMs, self.Masses1, self.A1)
#         b = np.interp(logMs, self.Masses1, self.B1)
#         c = np.interp(logMs, self.Masses1, self.C1)
#         z_1 = 1.0
#         return c / (np.power(10, a*(z-z_1)) + np.power(10, b*(z-z_1)))

#     def SSFR(self,logMs,z):
#         alpha = 0.7 - 0.13*z
#         beta  = 0.38 + 1.14*z - 0.19*z*z
#         lsSFR = alpha * (logMs - 10.5) + beta - logMs
#         return np.power(10,lsSFR)

#     def SSFR(self,logMs,z):
#         z_temp = np.minimum(z,np.zeros(z.shape)+2)
#         #z_temp = z
#         t    = cosmo.lookback_time(9999).value - cosmo.lookback_time(z_temp).value  ##
#         Ms   = np.power(10,logMs)
#         beta = -0
        
#         ssfr = 2.5e-9 * (Ms/(10e10))**(beta) * (t/3.5)**(-2.2)
        
#         return ssfr ## per year
    
    def SSFR(self, logMs, z):
        Ms = np.power(10,logMs)
        r = np.log10(1+z)
        m = logMs - 9
        m_0 = np.zeros(m.shape) + 0.5 # pm 0.07
        m_1 = np.zeros(m.shape) + 0.36 # pm 0.3
        a_0 = np.zeros(m.shape) + 1.5 # pm 0.15
        a_1 = np.zeros(m.shape) + 0.3 # pm 0.08
        a_2 = np.zeros(m.shape) + 2.5 # pm 0.6
                
        lsfr = m - m_0 + a_0 * r - a_1 * np.maximum(np.zeros(m.shape), m-m_1-a_2 * r)**2
        ssfr = 10**(lsfr) / Ms 
        
        return ssfr ## per year
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    
#     def t_delay(self, M_halo, z):
#         M_star    = self.M_star(M_halo,z)
#         logM_star = np.log10(M_star)
        
#         f_star    = M_star/M_halo
#         #f_cold    = 0.1*(1+np.minimum(z,np.zeros(len(z))+2))**2 * f_star
        
#         f_cold    = self.M_cold(logM_star, z)/M_halo
        
#         num       = self.f_bar - f_cold - f_star * (1 + self.eta*(1+self.R)) - self.f_strip
#         den       = f_star * (1 - self.R + self.eta) * self.SSFR(logM_star, z)
#         return (num/den)/(1e9)
    
#     def t_delay(self, M_halo, z):
#         return 3.8 * ((1+z)/(1+0.5))**(-3/2)
      
    def t_delay(self, M_halo, z):
        return np.zeros(np.shape(z)) + 30
    
    
    def M_cold(self,logMs, z):
        A_mu = 0.07 #pm 0.15
        B_mu = -3.8 #pm 0.4
        F_mu = 0.63 #pm 0.1
        C_mu = 0.53 #pm 0.03
        D_mu = -0.33 #pm 0.03
        
        beta = 2.
        
        sSFR_fcold = self.sSFR_fcold(logMs, z)
        sSFR       = self.SSFR(logMs, z)*1e9
        
        M_cold_Ms = A_mu + B_mu*(np.log10(1+z) - F_mu)**beta + C_mu*np.log10(sSFR / sSFR_fcold) + \
                    D_mu*(logMs - 10.7)
        
        return M_cold_Ms * np.power(10,logMs)
    
    def sSFR_fcold(self,logMs, z):
        t_c = 1.143 - 1.026*np.log10(1+z) - 0.599*np.log10(1+z)**2. + 0.528*np.log10(1+z)**3.
        log_sSFR_per_gyr = (-0.16 - 0.026*t_c) * (logMs+0.025)-(6.51 - 0.11*t_c) + 9
        
        return np.power(10, log_sSFR_per_gyr)
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    
    # STELLAR MASS GALAXY FUNCTION
    
    def phi_sf(self,Mh,z):  #SF Muzzin 2013  #TAKES Mh IN Msol
        logMs   = np.log10(self.M_star(Mh, z))
        z       = 0.35
        Density = np.log([10]) * self.Phi_S_sf(z) * np.power(10, (logMs-self.M_S_sf(z))*(1+self.Alpha_sf(z))) * np.exp(-np.power(10, (logMs - self.M_S_sf(z) )))
        return Density
     
    def phi_q(self,Mh,z):   #Quiescent Muzzin 2013 #TAKES Mh IN Msol
        logMs   = np.log10(self.M_star(Mh, z))
        z       = 0.35
        Density = np.log([10]) * self.Phi_S_q(z) * np.power(10, (logMs-self.M_S_q(z))*(1+self.Alpha_q(z))) * np.exp(-np.power(10, (logMs - self.M_S_q(z) )))
        return Density
    
    def phi_a(self,Mh,z):   #All Muzzin 2013  #TAKES Mh IN Msol
        density_total = self.phi_sf(Mh,z) + self.phi_q(Mh,z)
        return density_total
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    
    def phi_sf_2(self,logMs,z):  #SF Muzzin 2013  #TAKES Mh IN Msol
        Density = np.log([10]) * self.Phi_S_sf(z) * np.power(10, (logMs-self.M_S_sf(z))*(1+self.Alpha_sf(z))) * np.exp(-np.power(10, (logMs - self.M_S_sf(z) )))
        return Density
     
    def phi_q_2(self,logMs,z):   #Quiescent Muzzin 2013 #TAKES Mh IN Msol
        Density = np.log([10]) * self.Phi_S_q(z) * np.power(10, (logMs-self.M_S_q(z))*(1+self.Alpha_q(z))) * np.exp(-np.power(10, (logMs - self.M_S_q(z) )))
        return Density
    
    def phi_a_2(self,logMs,z):   #All Muzzin 2013  #TAKES Mh IN Msol
        density_total = self.phi_sf_2(logMs,z) + self.phi_q_2(logMs,z)
        return density_total
    
    def phi_sf_3(self,logMs):  #SF Muzzin 2013  #TAKES Mh IN Msol
        Density = 1e-5 * np.power(10, (logMs-10.6)*(1-1.4)) * np.exp(-np.power(10, (logMs - 10.6 )))
        return Density
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    
    def phi_q_remko(self,logMs):
        Density = np.power(10, (logMs-10.77)*(1-0.35)) * np.exp(-np.power(10, (logMs - 10.77 )))
        return Density
    
    def phi_a_M_int(self,logMh,z):  # Integrand of SF Muzin 2013 for calculating expectation mass #TAKES Mh IN logMsol
        Mh = np.power(10, logMh)
        return self.phi_a(Mh, z) * Mh * self.dMs_dMh(Mh,z)
    
    def phi_sf_M_int(self,logMh,z):  # Integrand of SF Muzin 2013 for calculating expectation mass #TAKES Mh IN logMsol
        Mh = np.power(10, logMh)
        return self.phi_sf(Mh, z) * Mh * self.dMs_dMh(Mh,z)
    
    
    
    def phi_sf_int(self,logMh,z):  # Integrand of SF Muzin 2013 for calculating expectation mass #TAKES Mh logMsol
        Mh = np.power(10, logMh)
        return self.phi_sf(Mh, z)# * self.dMs_dMh(Mh,z)
    
    def phi_q_int(self,logMh,z):  # Integrand of SF Muzin 2013 for calculating expectation mass #TAKES Mh logMsol
        Mh = np.power(10, logMh)
        return self.phi_q(Mh, z)# * self.dMs_dMh(Mh,z)
    
    def phi_a_int(self,logMh,z):  # Integrand of SF Muzin 2013 for calculating expectation mass #TAKES Mh logMsol
        Mh = np.power(10, logMh)
        return self.phi_a(Mh, z)# * self.dMs_dMh(Mh,z)
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    
    # MAIN PROGENITOR HALO MASS
    
    def M_main(self, z): #Main progenitor halo mass from Neistein 2006
        M_0 = np.power(10, self.logM0)
        gamma = self.omega_m * self.h * np.exp(-self.omega_b*(1+np.sqrt(2*self.h)/self.omega_m ))
        return np.log10(self.omega_m/(gamma**3) * newton_krylov(self.F_inv1(z,M_0,gamma), 0.01))
    
    def F_inv1(self, z, M_0, gamma):
        def F_inv2(x):
            return self.g(32*gamma)*(self.w(z) - self.w(self.z_0))/self.sigma_8 + self.F(gamma**3 * M_0 / self.omega_m) - self.F(x)
        return F_inv2
    
    def g(self, x):
        return 64.087 * np.power(1 + 1.074*x**(0.3) - 1.581*x**(0.4) + 0.954*x**(0.5) - 0.185*x**(0.6), -10)
    
    def F(self, u):
        return -6.92e-5*np.log(u)**4 + 5.0e-3*np.log(u)**3 + 8.64e-2*np.log(u)**2 - 12.66*np.log(u) + 110.8
    
    def w(self, z):
        return 1.260*(1 + z + 0.09/(1+z) + 0.24*np.exp(-1.16*z))
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # Parallel processing functions
    
    def func_A_normal(self,z):
        Mstar_from_halo = self.M_star(np.power(10,self.logMHalo_bins),z=z) ## Msol (NOT log!)
        logMs           = np.log10(Mstar_from_halo)
        phi_a_interp    = self.phi_sf_z_10_interp(logMs,z) + self.phi_q_z_10_interp(logMs,z)
        x               = np.power(10,self.logMHalo_bins)
        y               = phi_a_interp * np.power(10,self.logMHalo_bins) * \
                            self.dMs_dMh(np.power(10,self.logMHalo_bins),z)
        
        integral = np.trapz(x=x,y=y, dx=np.diff(x))
        return integral
    
    def func_delta_N_sf(self,item):
        z,j              = item
        logMs            = np.log10(self.M_star(np.power(10,self.logMHalo_bins), z))
        
        x                = self.phi_sf_z_10_interp(logMs,z)
        ys               = [(x[i]+x[i+1])/2 for i in range(len(x)-1)]
        areas            = np.diff(logMs)*ys
        
        return areas*self.A[j]
    
    def func_delta_N_q(self,item):
        z,j              = item
        logMs            = np.log10(self.M_star(np.power(10,self.logMHalo_bins), z))
        
        x                = self.phi_q_z_10_interp(logMs,z)
        ys               = [(x[i]+x[i+1])/2 for i in range(len(x)-1)]
        areas            = np.diff(logMs)*ys
        
        return areas*self.A[j]
    
    def func_delta_NMs(self,inpt):
        i,m = inpt
        return self.delta_N_a[:,i]*self.M_star(np.power(10,m),self.z_range)
    
    def func_delta_td(self,logMh):
        Mh = np.power(10,logMh)
        return self.t_delay(Mh, self.z_range)# - (cosmo.lookback_time(self.z_range).value - cosmo.lookback_time(self.z_0).value)
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    
    def func_delta_N_sf_final(self,item):
        z, delays, j = item        
        logMs        = np.log10(self.M_star(np.power(10,self.logMHalo_eff), z))
        phi_sf       = self.phi_sf_z_10_interp(logMs,z)
        phi_q        = self.phi_q_z_10_interp(logMs,z)
        
        t_end        = cosmo.lookback_time(self.z_0).value
        
        for i in range(0,len(logMs)):
            m  = logMs[i]
            td = delays[i]
            dt = 0.5 #Gyr
            t  = cosmo.lookback_time(z).value
            t_init = np.copy(t) 
            end_condition = True
            while t > np.maximum(t_end,t_init-td) and end_condition:
                #print(t)
                dn_b = self.dN_blue(m,z, phi_sf[i]) * phi_sf[i] * dt
                dn_r = self.dN_red(m,z, phi_sf[i], phi_q[i]) * phi_sf[i] * dt
                
                #print(dn_b,phi_sf[i])
                
                phi_sf[i] += dn_b
                phi_q[i]  += dn_r
                
                if t - dt <= np.maximum(t_end,t_init-td):
                    dt = t - np.maximum(t_end,t_init-td)
                    end_condition= False
                
                t -= dt
            
            if t_end <= t_init-td:
                phi_q[i]  += phi_sf[i]
                phi_sf[i]  = 0
                
        return [phi_sf, phi_q]
    
    def dN_blue(self, logMs, z, N_b):
        alpha    = -1.3
        beta     = -0
        lambda_m = (np.power(10,logMs)/np.power(10,10.6)) * self.SSFR(logMs,z)*1e9 
        return N_b * (-(1 + alpha + beta - (np.power(10,logMs)/np.power(10,10.6)))*self.SSFR(logMs, z)*1e9 - lambda_m) # - 0.027/4*(1+z)**(1.2))
    
    def dN_red(self, logMs, z, N_b, N_r):
        lambda_m = (np.power(10,logMs)/np.power(10,10.6)) * self.SSFR(logMs,z)*1e9 
        return N_b * lambda_m #- 0.027/4*(1+z)**(1.2)*N_r
    
    
    def group_by_mass(self, N_array, mass_vector):
        output_vector = []
        
        #print(N_array.shape, self.logMStar_upp.shape, self.logMStar_low.shape)
        
        for mass in mass_vector:
            
            valid = N_array[(mass <  self.logMStar_upp) & 
                            (mass >= self.logMStar_low)]
            
            output_vector.append(np.sum(valid))
            
        return np.array(output_vector)
    
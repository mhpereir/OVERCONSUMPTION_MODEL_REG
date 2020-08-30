import sys
import numpy as np

from astropy import units as u
from astropy.cosmology import Planck15 as cosmo, z_at_value
from scipy.interpolate import interp1d, interp2d
from scipy.optimize import newton_krylov, brentq
from scipy.integrate import quad
from multiprocessing import Pool
from time import time


from utils import integration_utils

#import matplotlib.pyplot as plt

class PENG_model:
    def __init__(self, params, z_i, z_f): 
        self.logM_min  = params['model_setup']['logM_min']    # minimum sampled mass from init SF population
        self.logM_max  = params['model_setup']['logM_max']    # maximum //
        self.logM_std  = params['model_setup']['logM_std']    # std of MCMC - Metropolis-hastings sampler
        self.f_s_limit = params['model_setup']['f_s_limit']   # lower limit in the stellar mass fraction of galaxy halo
        
        self.x_bins    = np.arange(self.logM_min, 15.01, 0.1)
        self.x_midp    = (self.x_bins[1:] + self.x_bins[:-1])/2
        
        self.sSFR_key  = params['model_setup']['sSFR']
        
        #Cosmology params
        self.omega_m = params['cosmology']['omega_m']
        self.omega_b = params['cosmology']['omega_b']
        self.h       = params['cosmology']['h']
        self.sigma_8 = params['cosmology']['sigma_8']
        
        self.f_bar   = 0.18
        self.f_strip = 0
        self.R       = 0.56
        
        self.step    = 0.01 #Gyr
        
        #Model params
        self.z_init  = z_i
        self.z_final = z_f
        
        # Global variables
        self.mass_history = []
    
    ###  Schechter Stuff ###
    def schechter_SMF_prob(self, logMs):
        if logMs > self.logM_min and logMs <= self.logM_max:
            Density = 1e-5 * np.power(10, (logMs-10.6)*(1-1.4)) * np.exp(-np.power(10, (logMs-10.6 )))
        else:
            Density = 0
        return Density
    
    def open_HMF_file(self,file_name='./HMF_watson.txt'):
        file_content = np.genfromtxt(file_name, skip_header=12)
        
        logMh    = np.log10(file_content[:,0])
        dn_dm    = file_content[:,5]
        
        Density = interp1d(logMh, dn_dm, fill_value=0, bounds_error=False)
        
        return Density
    
    def schechter_SMF_func(self, logMs):
        Density = 1e-5 * np.power(10, (logMs-10.6)*(1-1.4)) * np.exp(-np.power(10, (logMs-10.6 )))
        return Density
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    
    ### PENG Monte Carlo Model ###
    def gen_galaxies(self,N):
        start_time = time()
        list_masses = []
        x_0 = 9 #np.random.rand()*(self.logM_max-self.logM_min) + self.logM_min
        n   = 0
        
        self.phi_hm_interp = self.open_HMF_file()
        
        while n < N:
            x_1 = np.random.normal(x_0, self.logM_std)
            
            p0 = self.phi_hm_interp(x_0)
            p1 = self.phi_hm_interp(x_1)
            
            if np.random.rand() < p1/p0:
                x_0 = x_1
                list_masses.append(x_0)
                n  += 1
            else:
                pass
        
        print('Time to generate population: {:.2f} m.'.format( (time() - start_time)/60  ))
        
        self.hmass_init   = np.array(list_masses, copy=True)
        self.hmass_init   = self.hmass_init[self.hmass_init > (self.logM_min + self.logM_std)]
        self.hmass_init   = self.hmass_init[self.hmass_init < (self.logM_max - self.logM_std)]
        
        #self.hmass_init = np.arange(8,11.1,0.01)
        
        list_masses       = None
    
    def setup_evolve(self):
        t_init       = cosmo.lookback_time(self.z_init).value #gyr
        self.t_final = cosmo.lookback_time(self.z_final).value #gyr
        
        self.t  = t_init
        self.z  = self.z_init

        # self.integ     = integration_utils(self.DE, hmax=0.1, hmin=1e-2, htol=1e0)
        self.condition = True                # Always run at least once
        # self.force     = False

        self.hmass    = np.ma.power(10, np.array(self.hmass_init, copy=True))      #halo mass
        self.gmass    = np.ma.zeros(self.hmass.shape)                              #gas mass
        self.smass    = np.ma.zeros(self.hmass.shape)                              #stellar mass
        
    def evolve_field(self):
        vals      = np.array([self.hmass, self.gmass, self.smass])
        #vals_2    = self.integ.RK45(vals, self.t, self.force)
        
        vals_2    = vals + self.DE(vals, self.t)# * self.step
        
        self.hmass = vals_2[0,:]
        self.gmass = vals_2[1,:]
        self.smass = vals_2[2,:]
        
        if (self.t - self.step) > self.t_final:
            pass
        else:
            self.step = self.t - self.t_final
            # self.force      = True
            self.condition  = False
        
        hist_hm ,_          = np.histogram( np.log10(self.hmass) , bins=self.x_bins)
               
        self.phi_hm_interp  = interp1d(self.x_midp, hist_hm , bounds_error=True)
            
    def gen_cluster(self, cluster_mass, n_clusters, oc_flag, oc_eta):
        '''
        Generates the cluster. Samples N galaxies from the star forming galaxy array
        '''
        self.logM0     = cluster_mass
        self.z_0       = self.z_final
        self.n_cluster = n_clusters
        self.oc_flag   = oc_flag  #toggle for applying OC in the model
        self.oc_eta    = oc_eta
        
        self.progenitor_mass = np.power(10, self.M_main(self.z_init))
        
        A_norm               = self.progenitor_mass / self.A_denom(self.z_init) * self.n_cluster
        
        Mh_min      = np.ma.min(self.hmass)
        Mh_max      = np.ma.max(self.hmass)
        
        phi_sf_at_z = lambda x: self.phi_hm_interp(np.log10(x))
        N           = A_norm * quad(phi_sf_at_z, Mh_min, Mh_max, epsrel=1e-2)[0]
        N_floor     = int(np.floor(N))
        N_rest      = N - N_floor
        
        
        if N_floor != 0:
            ii = np.random.randint(low=0, high=len(self.hmass), size=N_floor)
            
        if np.random.uniform(low=0.0, high=1.0, size=1) <= N_rest:
            iii       = np.random.randint(low=0, high=len(self.hmass), size=1)
            N         = int(N_floor + 1)
        else:
            N         = int(N_floor)
        
        if N == 0:
            self.cluster_hmass  = None 
            self.cluster_gmass  = None
            self.cluster_smass  = None
            
            self.infall_Ms      = None
            self.infall_Mh      = None
            self.infall_z       = None
            
        else:            
            if N_floor > 0:
                self.cluster_hmass              = np.ma.zeros(N)
                self.cluster_gmass              = np.ma.zeros(N)
                self.cluster_smass              = np.ma.zeros(N)
                
                self.cluster_hmass[0:N_floor]   = np.ma.copy(self.hmass[ii])
                self.cluster_gmass[0:N_floor]   = np.ma.copy(self.gmass[ii])
                self.cluster_smass[0:N_floor]   = np.ma.copy(self.smass[ii])
                
            if N == (N_floor+1) and N_floor >0:
                self.cluster_hmass[-1]          = np.ma.copy(self.hmass[iii])
                self.cluster_gmass[-1]          = np.ma.copy(self.gmass[iii])
                self.cluster_smass[-1]          = np.ma.copy(self.smass[iii])
            elif N == (N_floor+1) and N_floor == 0:
                self.cluster_hmass              = np.ma.copy(self.hmass[iii])
                self.cluster_gmass              = np.ma.copy(self.gmass[iii])
                self.cluster_smass              = np.ma.copy(self.smass[iii])
            
            self.infall_Mh                     = np.ma.copy(self.cluster_hmass)
            self.infall_Ms                     = np.ma.copy(self.cluster_smass)
            
            self.infall_z                      = np.ma.zeros(N) + self.z_init
            
            self.mass_history.append(self.cluster_smass)
        
    def grow_cluster(self):
        new_progenitor_mass  = np.power(10, self.M_main(self.z))
        mass_increase        = new_progenitor_mass - self.progenitor_mass
        A_norm               = mass_increase / self.A_denom(self.z) * self.n_cluster
               
        self.progenitor_mass = new_progenitor_mass
        
        Mh_min               = np.ma.min(self.hmass)
        Mh_max               = np.ma.max(self.hmass)
        
        phi_at_z = lambda x: self.phi_hm_interp(np.log10(x))
        N           = A_norm * quad(phi_at_z, Mh_min, Mh_max, epsrel=1e-2)[0]
        N_floor     = int(np.floor(N))
        N_rest      = N - N_floor
        
        if N_floor != 0:
            ii = np.random.randint(low=0, high=len(self.hmass), size=N_floor)
            
        if np.random.uniform(low=0.0, high=1.0, size=1) <= N_rest:
            iii       = np.random.randint(low=0, high=len(self.hmass), size=1)
            N         = int(N_floor + 1)
        else:
            N         = int(N_floor)
        
        print(self.t, self.step, A_norm, N)
        
        if N == 0:
            pass
        else:
            if self.cluster_hmass is None:
                if N_floor > 0:            
                    self.cluster_hmass              = np.ma.zeros(N)
                    self.cluster_gmass              = np.ma.zeros(N)
                    self.cluster_smass              = np.ma.zeros(N)
                    
                    self.cluster_hmass[0:N_floor]   = np.ma.copy(self.hmass[ii])
                    self.cluster_gmass[0:N_floor]   = np.ma.copy(self.gmass[ii])
                    self.cluster_smass[0:N_floor]   = np.ma.copy(self.smass[ii])
                    
                if N == (N_floor+1) and N_floor >0:
                    self.cluster_hmass[-1]          = np.ma.copy(self.hmass[iii])
                    self.cluster_gmass[-1]          = np.ma.copy(self.gmass[iii])
                    self.cluster_smass[-1]          = np.ma.copy(self.smass[iii])
                    
                elif N == (N_floor+1):
                    self.cluster_hmass              = np.ma.copy(self.hmass[iii])
                    self.cluster_gmass              = np.ma.copy(self.gmass[iii])
                    self.cluster_smass              = np.ma.copy(self.smass[iii])
                    
                self.infall_Mh                     = np.ma.copy(self.cluster_hmass)
                self.infall_Ms                     = np.ma.copy(self.cluster_smass)
            
                self.infall_z                      = np.ma.zeros(N) + self.z_init
                
                self.mass_history.append(self.cluster_hmass)
                
            else:
                n                             = len(self.cluster_hmass)
                
                new_arr_hmass                 = np.ma.zeros(n + N)
                new_arr_gmass                 = np.ma.zeros(n + N)
                new_arr_smass                 = np.ma.zeros(n + N)
                
                new_arr_Ms                    = np.ma.zeros(n + N)
                new_arr_Mh                    = np.ma.zeros(n + N)
                
                new_arr_z                     = np.ma.zeros(n + N)
                
                new_arr_hmass[0:n]            = self.cluster_hmass[:]
                new_arr_gmass[0:n]            = self.cluster_gmass[:]
                new_arr_smass[0:n]            = self.cluster_smass[:]
                
                new_arr_Ms[0:n]               = self.infall_Ms[:]
                new_arr_Mh[0:n]               = self.infall_Mh[:]
                new_arr_z[0:n]                = self.infall_z[:]
                
                if N_floor > 0:               
                    new_arr_hmass[n:n+N_floor]  = np.ma.copy(self.hmass[ii])
                    new_arr_gmass[n:n+N_floor]  = np.ma.copy(self.gmass[ii])
                    new_arr_smass[n:n+N_floor]  = np.ma.copy(self.smass[ii])
                    
                if N == (N_floor+1):
                    new_arr_hmass[n:n+N_floor]  = np.ma.copy(self.hmass[iii])
                    new_arr_gmass[n:n+N_floor]  = np.ma.copy(self.gmass[iii])
                    new_arr_smass[n:n+N_floor]  = np.ma.copy(self.smass[iii])
                
                new_arr_z[n:n+N]                = np.ma.zeros(N) + self.z
                new_arr_Ms[n:n+N]               = np.ma.copy(new_arr_smass[n:n+N])
                new_arr_Mh[n:n+N]               = np.ma.copy(new_arr_hmass[n:n+N])
                               
                self.cluster_hmass = new_arr_hmass
                self.cluster_gmass = new_arr_gmass
                self.cluster_smass = new_arr_smass
                
                
                self.infall_z       = new_arr_z
                self.infall_Ms      = new_arr_Ms
                self.infall_Mh      = new_arr_Mh
    
                self.mass_history.append(self.cluster_hmass)
    
    def evolve_cluster(self):        
        if self.cluster_hmass is None:
            pass
        else:
            vals      = np.array([self.cluster_hmass, self.cluster_gmass, self.cluster_smass])
            
            #mass_array = self.integ.RK45(inits, self.t, force=True)
            vals_2    = vals + self.DE2(vals, self.t)# * self.step
            
            self.cluster_hmass = vals_2[0,:]
            self.cluster_gmass = vals_2[1,:]
            self.cluster_smass = vals_2[2,:]
        
    def update_step(self):
        self.t -= self.step
        self.z  = z_at_value(cosmo.age,(cosmo.age(0).value - self.t)*u.Gyr, zmin=-1e-6)
    
    def parse_masked_mass_field(self):
        ssfr = self.SFR(self.gmass, self.z_final) / self.smass
                
        self.final_mass_field_SF = np.log10(self.smass[ssfr >= 5e-10])
        self.final_mass_field_Q  = np.log10(self.smass[ssfr <  5e-10])
    
    def parse_masked_mass_cluster(self):
        ssfr = self.SFR(self.cluster_gmass, self.z_final) / self.cluster_smass
                
        self.final_mass_cluster_SF = np.log10(self.cluster_smass[ssfr >= 5e-10])
        self.final_mass_cluster_Q  = np.log10(self.cluster_smass[ssfr <  5e-10])
    
    def DE(self, item, t):
        Mhalo = item[0]
        Mgas  = item[1]
        # Mstar = item[2]
        z     = z_at_value(cosmo.age,(cosmo.age(0).value - self.t)*u.Gyr, zmin=-1e-6)
        
        dMh = self.delta_Mhalo(Mhalo, z) * self.step*1e9
        sfr = self.SFR(Mgas, z) * self.step*1e9
        dMg = self.delta_Mgas(dMh, sfr, Mhalo, z)
        dMs = sfr * (1-self.R)
                
        return np.array([dMh, dMg, dMs])
    
    
    def DE2(self, item, t):
        Mhalo = item[0]
        Mgas  = item[1]
        # Mstar = item[2]
        z     = z_at_value(cosmo.age,(cosmo.age(0).value - self.t)*u.Gyr, zmin=-1e-6)
        
        dMh = np.zeros(len(Mhalo))
        sfr = self.SFR(Mgas, z) * self.step*1e9
        dMg = self.delta_Mgas(dMh, sfr, Mhalo, z)
        dMs = sfr * (1-self.R)
                
        return np.array([dMh, dMg, dMs])
    
    def delta_Mhalo(self, Mhalo, z):
        return 510 *  (Mhalo/(10**12))**(1.10) * ((1+z)/3.2)**(2.2)

    def epsilon_in(self, Mhalo, z):
        output = np.zeros(len(Mhalo))
        
        output[(Mhalo > 1.5e12)  & (Mhalo < 1e11)]  = 0
        output[(Mhalo <= 1.5e12) & (Mhalo >= 1e11)] = 0.7 * (np.minimum(z,2.2)/4.4 + 1/2)
        
        return output
        
    def delta_Mgas(self, delta_Mh, SFR, Mh, z):
        M_in  = self.epsilon_in(Mh,z) * self.f_bar * delta_Mh
        M_out = (1 - self.R + self.oc_eta) * SFR 
        
        return M_in - M_out
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    
    ### sSFR functions ###
    def sSFR_peng(self, logMs, z, *ssfr_params):          #PENG sSFR
        z_temp = np.minimum(z,2)
        #z_temp = z
        t    = cosmo.lookback_time(9999).value - cosmo.lookback_time(z_temp).value  ##
        Ms   = np.power(10,logMs)
        beta = -0
        
        ssfr = 2.5e-9 * (Ms/(10e10))**(beta) * (t/3.5)**(-2.2)
        
        return ssfr ## per year
    
    def sSFR_schreiber_old(self, logMs, z):       # Schreiber+2015
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
    
    def sSFR_speagle_old(self, logMs, z):
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
    
    def SFR(self, Mgas,z):
        epsilon_sfr = 0.02
        t_dyn = 2e7 * ((1+z)/3.2)**(-1.5)
        return epsilon_sfr * Mgas / t_dyn

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    
    ### Overconsumption Delay times ###
    def t_delay(self, item):
        M_star, ssfr_p, M_halo, z = item[0],item[1],item[2],item[3]
        
        #delay time params
        
        logMs   = np.ma.log10(M_star)
        
        f_star    = M_star/M_halo
        #f_cold    = 0.1*(1+np.minimum(z,2))**2 * f_star
        f_cold    = self.M_cold(logMs, z, ssfr_p)/M_halo
        
        num       = self.f_bar - f_cold - f_star * (1 + self.oc_eta*(1+self.R)) - self.f_strip
        den       = f_star * (1 - self.R + self.oc_eta) * self.sSFR(logMs, z, ssfr_p)
        return (num/den)/(1e9) #gyr
        
    def t_delay_2(self, M_halo, z):
        
        #delay time params
        
        M_star     = self.M_star(M_halo,z)
        logMs      = np.log10(M_star)
        
        f_star     = M_star/M_halo
        f_cold    = 0.1*(1+np.minimum(z,2))**2 * f_star
        #f_cold    = self.M_cold(logMs, z)/M_halo
        
        num       = self.f_bar - f_cold - f_star * (1 + self.oc_eta*(1+self.R)) - self.f_strip
        den       = f_star * (1 - self.R + self.oc_eta) * self.sSFR(logMs, z)
        return (num/den)/(1e9) #gyr
        
    
    
    def M_cold(self,logMs, z, ssfr_params):          #cold gas mass fraction 
        A_mu = 0.07 #pm 0.15
        B_mu = -3.8 #pm 0.4
        F_mu = 0.63 #pm 0.1
        C_mu = 0.53 #pm 0.03
        D_mu = -0.33 #pm 0.03
        
        beta = 2.
        
        sSFR_fcold = self.sSFR_fcold(logMs, z)
        sSFR       = self.sSFR(logMs, z, ssfr_params)*1e9
        
        M_cold_Ms = A_mu + B_mu*(np.log10(1+z) - F_mu)**beta + C_mu*np.log10(sSFR / sSFR_fcold) + \
                    D_mu*(logMs - 10.7)
        
        return M_cold_Ms * np.power(10,logMs)
    
    def sSFR_fcold(self,logMs, z):
        t_c = 1.143 - 1.026*np.log10(1+z) - 0.599*np.log10(1+z)**2. + 0.528*np.log10(1+z)**3.
        log_sSFR_per_gyr = (-0.16 - 0.026*t_c) * (logMs+0.025)-(6.51 - 0.11*t_c) + 9
        
        return np.power(10, log_sSFR_per_gyr)
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    
    ### Main progenitor mass growth ###
    def M_main(self, z): #Main progenitor halo mass from Neistein 2006
        M_0 = np.power(10, self.logM0)
        gamma = self.omega_m * self.h * np.exp(-self.omega_b*(1+np.sqrt(2*self.h)/self.omega_m ))
        return np.log10(self.omega_m/(gamma**3) * newton_krylov(self.F_inv1(z,self.z_0,M_0,gamma), 0.01))
    
    def F_inv1(self, z, z_0, M_0, gamma):
        def F_inv2(x):
            return self.g(32*gamma)*(self.w(z) - self.w(self.z_0))/self.sigma_8 + self.F(gamma**3 * M_0 / self.omega_m) - self.F(x)
        return F_inv2
    
    def g(self, x):
        return 64.087 * np.power(1 + 1.074*x**(0.3) - 1.581*x**(0.4) + 0.954*x**(0.5) - 0.185*x**(0.6), -10)
    
    def F(self, u):
        return -6.92e-5*np.log(u)**4 + 5.0e-3*np.log(u)**3 + 8.64e-2*np.log(u)**2 - 12.66*np.log(u) + 110.8
    
    def w(self, z):
        return 1.260*(1 + z + 0.09/(1+z) + 0.24*np.exp(-1.16*z))
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    ### Stellar mass relationship from Behroozi 2013
    def M_star(self,M_halo,z):  # input in Msol
        a   = 1/(1+z)
        v   = np.exp(-4*a*a)
        
        logEps = -1.777 + (-0.006*(a-1) + (-0.000)*z)*v - 0.119*(a-1)
        eps    = np.power(10, logEps)
        
        logM1 = 11.514 + (-1.793*(a-1) + (-0.251)*z)*v 
        m1    = np.power(10,logM1)
        
        logM_star = np.log10(eps * m1) + self.f(np.log10(M_halo/m1), z,a,v) - self.f(0, z,a,v)
        return np.power(10, logM_star)        
    
    def f(self,x,z,a,v):
        alpha = -1.412 + (0.731*(a-1))*v
        gamma = 0.316 + (1.319*(a-1) + 0.279*z)*v
        delta = 3.508 + (2.608*(a-1)+(-0.043)*z)*v
        return -np.log10(np.power(10,alpha*x) + 1) + delta * np.power(np.log10(1+np.exp(x)), gamma) / (1 + np.exp(np.power(10,-x)))
    
    def dMs_dMh(self,Mh,z):  #1st order derivative: central limit theorem
        h = 100 # Mh is in Msol h = 1  means dx of 1 solar mass
        return ( self.M_star(Mh + h,z) - self.M_star(Mh - h,z) )/ (2*h) 
    
    def M_star_inv(self, logMs, z):
        Ms = np.power(10, logMs)
        def Ms_inv(x):
            return (self.M_star(np.power(10,x), z) - Ms)/Ms
        return Ms_inv
    
    def find_M_halo(self,Ms, z):
        logMs      = np.ma.log10(Ms)
        logMs.mask = np.ma.nomask
        #tt = time()
        logMh     = np.array([self.func_mpp(logms,z) for logms in logMs])
        
        M_halo    = np.ma.power(10, logMh)
        M_halo[self.f_s_limit > Ms/M_halo] = Ms[self.f_s_limit > Ms/M_halo] / (self.f_s_limit)
        
        temp = np.log10(np.ma.max(M_halo))
        if temp >=16:
            sys.exit('WARNING: maximum halo_mass exceeded')
        #print('find_M_Halo : {}'.format(time() - tt))
        
        return M_halo

    def func_mpp(self,logms, z):
        #logms, z = item[0], item[1]
        return brentq(self.M_star_inv(logms,z), 5, 100)
    
    def find_logM_halo(self,logMs,z):
        Ms        = np.power(10,logMs)
        logMh     = brentq(self.M_star_inv(logMs,z), 5, 100)
        M_halo    = np.ma.power(10, logMh)
        
        M_halo[self.f_s_limit > Ms/M_halo] = Ms[self.f_s_limit > Ms/M_halo]/(self.f_s_limit)
        logMh                    = np.log10(M_halo)
        
        if logMh >=16:
            sys.exit('WARNING: maximum halo_mass exceeded')
        
        return logMh

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~# 
    
    ### Other functions ###
    def A_denom(self,z):
        '''
        Denominator of normalization constant A
        '''
        
        logMh_min = np.log10(np.ma.min(self.hmass))           ###z 
        logMh_max = np.log10(np.ma.max(self.hmass))           ### z
        
        logMhalo_range  = np.arange(logMh_min, logMh_max, 0.0005)
        
        phi_a_interp    = self.phi_hm_interp(logMhalo_range)
        x               = np.power(10, logMhalo_range)
        y               = phi_a_interp*np.power(10,logMhalo_range) ## ## *self.dMs_dMh(np.power(10,logMhalo_range),z)
        
        integral = np.trapz(x=x,y=y, dx=np.diff(x))
        return integral
        
    
    
    
    
    
    
    
        #def evolve(self, z_i, z_f):
        #t_init  = cosmo.lookback_time(z_i).value #gyr
        #t_final = cosmo.lookback_time(z_f).value #gyr
        #dt      = 0.5 # Gyr
        
        #mass_array = np.power(10, np.array(self.sf_masses, copy=True))
        
        #t   = t_init
        #z_1 = z_i
        
        #t  -= dt
        #z_2 = z_at_value(cosmo.age,(cosmo.age(0).value - t)*u.Gyr) 
        #end_condition = True
        #while t >= t_final and end_condition:            
            #if z_1 <= 3:
                #k_minus = 0.027/4/np.sqrt(1+(z_1 + z_2)/2 )/1e9     #merging is turned-off at z>3, PENG+2010
            #else:
                #k_minus = 0
            
            #death_func  = np.minimum( (self.eta_m(np.log10(mass_array), (z_1+z_2)/2 )) * dt*1e9, np.ones(mass_array.shape))
            #prob_array  = np.random.uniform(size=mass_array.shape)
            
            #mass_array  = np.ma.masked_where(death_func > prob_array, mass_array)
            #mass_array += mass_array * (self.sSFR(np.log10(mass_array), z_1)+self.sSFR(np.log10(mass_array), z_2))/2 * dt*1e9 # gyr to yr
            
            #if t - dt < t_final:
                #dt = t - t_final
                #end_condition = False
            #else:
                #pass
            
            #t -= dt
            #z_1  = z_2
            #z_2  = z_at_value(cosmo.age,(cosmo.age(0).value - t)*u.Gyr, zmin=-1e-6) 
            
        #temp       = np.ma.copy(mass_array)
        
        #mask               = mass_array.mask
        #temp.mask          = np.ma.nomask
        #self.final_mass_SF = np.log10(temp[np.logical_not(mask)])
        #self.final_mass_Q  = np.log10(temp[mask])
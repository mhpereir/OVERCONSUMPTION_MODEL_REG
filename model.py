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
        
        self.x_bins    = np.arange(self.logM_min, 12.01, 0.1)
        self.x_midp    = (self.x_bins[1:] + self.x_bins[:-1])/2
        
        self.sSFR_key       = params['model_setup']['sSFR']
        
        #Cosmology params
        self.omega_m = params['cosmology']['omega_m']
        self.omega_b = params['cosmology']['omega_b']
        self.h       = params['cosmology']['h']
        self.sigma_8 = params['cosmology']['sigma_8']
        
        #Model params
        self.z_init  = z_i
        self.z_final = z_f
        
        # Global variables
        self.mass_history = []
        
        if self.sSFR_key == 'Peng' or self.sSFR_key == 'PENG':
            self.sSFR            = self.sSFR_peng
            self.sSFR_an         = self.sSFR_peng
            self.gen_ssfr_params = self.gen_ssfr_peng_params
        
        elif self.sSFR_key == 'Schreiber' or self.sSFR_key == 'SCHREIBER':
            self.sSFR            = self.sSFR_schreiber
            self.sSFR_an         = self.sSFR_schreiber_old
            self.gen_ssfr_params = self.gen_ssfr_schreiber_params
        
        elif self.sSFR_key == 'Speagle' or self.sSFR_key == 'SPEAGLE':
            self.sSFR            = self.sSFR_speagle
            self.sSFR_an         = self.sSFR_speagle_old
            self.gen_ssfr_params = self.gen_ssfr_speagle_params
        
        else:
            print('Unrecognized sSFR name. Try: Peng; Schreiber or Speagle.')
            sys.exit()
    
    ###  Schechter Stuff ###
    def schechter_SMF_prob(self, logMs):
        if logMs > self.logM_min and logMs <= self.logM_max:
            Density = 1e-5 * np.power(10, (logMs-10.6)*(1-1.4)) * np.exp(-np.power(10, (logMs-10.6 )))
        else:
            Density = 0
        return Density
    
    def schechter_SMF_func(self, logMs):
        Density = 1e-5 * np.power(10, (logMs-10.6)*(1-1.4)) * np.exp(-np.power(10, (logMs-10.6 )))
        return Density
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    
    ### PENG Monte Carlo Model ###
    def gen_galaxies(self,N):
        start_time = time()
        list_masses = []
        x_0 = np.random.rand()*(self.logM_max-self.logM_min) + self.logM_min
        n   = 0
        
        while n < N:
            x_1 = np.random.normal(x_0, self.logM_std)
            
            p0 = self.schechter_SMF_prob(x_0)
            p1 = self.schechter_SMF_prob(x_1)
            
            if np.random.rand() < p1/p0:
                x_0 = x_1
                list_masses.append(x_0)
                n  += 1
            else:
                pass
        
        print('Time to generate population: {:.2f} m.'.format( (time() - start_time)/60  ))
        
        self.sf_masses   = np.array(list_masses, copy=True)
        self.sf_masses   = self.sf_masses[self.sf_masses > (self.logM_min + self.logM_std)]
        self.sf_masses   = self.sf_masses[self.sf_masses < (self.logM_max - self.logM_std)]
        list_masses      = None
    
    def setup_evolve(self):
        t_init       = cosmo.lookback_time(self.z_init).value #gyr
        self.t_final = cosmo.lookback_time(self.z_final).value #gyr
        
        self.t  = t_init
        self.z  = self.z_init

        self.integ     = integration_utils(self.DE, hmax=1, hmin=1e-5, htol=1e1)
        self.condition = True                # Always run at least once
        self.force     = False

        self.mass_array = np.ma.power(10, np.array(self.sf_masses, copy=True))
        
        self.ssfr_params = self.gen_ssfr_params(len(self.sf_masses))
    
        self.phi_sf_interp = self.schechter_SMF_func
    def phi_q_interp(self,logMs):
        return np.zeros(logMs.shape)
    
    def evolve_field(self, p):        
        if self.z <= 3:
            k_minus = 0.027/4/np.sqrt(1+self.z)/1e9     #merging is turned-off at z>3, PENG+2010
        else:
            k_minus = 0
        
        mass_mask = np.logical_not(self.mass_array.mask)
        if len(self.mass_array[mass_mask]) == 0:
            pass
        else:
            inits                       = (self.mass_array[mass_mask], self.ssfr_params[:,mass_mask])
            self.mass_array[mass_mask]  = self.integ.RK45(p, inits, self.t, self.force, mpp=False, analytic=False)
            
            if (self.t - self.integ.step) > self.t_final:
                pass
            else:
                self.integ.step = self.t - self.t_final
                self.force      = True
                self.condition  = False
            
            death_func  = np.minimum( self.eta_m(np.log10(self.mass_array), self.z, self.ssfr_params) * self.integ.step *1e9, np.ones(self.mass_array.shape))
            prob_array  = np.ones(self.mass_array.shape)
            prob_array[mass_mask]  = np.random.uniform(size=self.mass_array.count())
            self.mass_array        = np.ma.masked_where(death_func > prob_array, self.mass_array)
            
            self.parse_masked_mass_field()
            
            hist_sf,_          = np.histogram( self.final_mass_field_SF, bins=self.x_bins)
            hist_q ,_          = np.histogram( self.final_mass_field_Q , bins=self.x_bins)
            
            self.phi_sf_interp = interp1d(self.x_midp, hist_sf, fill_value="extrapolate")
            self.phi_q_interp  = interp1d(self.x_midp, hist_q , fill_value="extrapolate")
            
    def gen_cluster(self, cluster_mass, n_clusters, oc_flag, oc_eta):
        '''
        Generates the cluster. Samples N galaxies from the star forming galaxy array
        '''
        self.logM0     = cluster_mass
        self.z_0       = self.z_final
        self.n_cluster = n_clusters
        self.oc_flag   = oc_flag  #toggle for applying OC in the model
        self.oc_eta    = oc_eta
        
        self.MQ_flags  = []
        self.OC_flags  = []
        
        self.progenitor_mass = np.power(10, self.M_main(self.z_init))
        
        A_norm               = self.progenitor_mass / self.A_denom(self.z_init) * self.n_cluster
        
        temp_mass_array      = np.ma.copy(self.mass_array)
        temp_mass_array.mask = np.ma.nomask 
        
        Ms_min      = np.ma.min(temp_mass_array)
        Ms_max      = np.ma.max(temp_mass_array)
        
        phi_sf_at_z = lambda x: self.phi_sf_interp(np.log10(x))
        N           = A_norm * quad(phi_sf_at_z, Ms_min, Ms_max, epsrel=1e-2)[0]
        N_floor     = int(np.floor(N))
        N_rest      = N - N_floor
        
        
        if N_floor != 0:
            ii = np.random.randint(low=0, high=len(self.mass_array), size=N_floor)
            
        if np.random.uniform(low=0.0, high=1.0, size=1) <= N_rest:
            iii       = np.random.randint(low=0, high=len(self.mass_array), size=1)
            N         = int(N_floor + 1)
        else:
            N         = int(N_floor)
        
        if N == 0:
            self.cluster_masses = None
            self.cluster_ssfr_p = None
            self.infall_Ms      = None
            self.infall_Mh      = None
            self.infall_z       = None
            
        else:
            
            m = len(self.ssfr_params[:,0])
            
            if N_floor > 0:
                self.cluster_masses              = np.ma.zeros(N)
                self.cluster_ssfr_p              = np.ma.zeros([m,N])
                self.cluster_masses[0:N_floor]   = np.ma.copy(self.mass_array[ii])
                self.cluster_ssfr_p[:,0:N_floor] = np.ma.copy(self.ssfr_params[:,ii])
                
            if N == (N_floor+1) and N_floor >0:
                self.cluster_masses[-1]        = np.ma.copy(self.mass_array[iii])
                self.cluster_ssfr_p[:,-1]      = np.ma.copy(self.ssfr_params[:,iii]).flatten()
            elif N == (N_floor+1) and N_floor == 0:
                self.cluster_masses            = np.ma.copy(self.mass_array[iii])
                self.cluster_ssfr_p            = np.ma.copy(self.ssfr_params[:,iii]).flatten()
            
            self.infall_Ms                     = np.ma.copy(self.cluster_masses)
            self.infall_Mh                     = self.find_M_halo(self.infall_Ms,self.z_init)       ###### z_init
            
            self.infall_z                      = np.ma.zeros(N) + self.z_init
            
            self.mass_history.append(self.cluster_masses)
        
        if self.oc_flag and N!=0:
            td              = np.ma.array(self.t_delay([self.cluster_masses, self.cluster_ssfr_p, self.infall_Mh, self.z_init]))
            self.death_date = cosmo.lookback_time(self.z_init).value - td
        
    def grow_cluster(self):
        new_progenitor_mass  = np.power(10, self.M_main(self.z))
        mass_increase        = new_progenitor_mass - self.progenitor_mass
        A_norm               = mass_increase / self.A_denom(self.z) * self.n_cluster
        
        self.progenitor_mass = new_progenitor_mass
        
        temp_mass_array      = np.ma.copy(self.mass_array) # identify integration range
        temp_mass_array.mask = np.ma.nomask 
        Ms_min               = np.ma.min(temp_mass_array)
        Ms_max               = np.ma.max(temp_mass_array)
        temp_mass_array      = None
        
        phi_sf_at_z = lambda x: self.phi_sf_interp(np.log10(x))
        N           = A_norm * quad(phi_sf_at_z, Ms_min, Ms_max, epsrel=1e-2)[0]
        N_floor     = int(np.floor(N))
        N_rest      = N - N_floor
        
        if N_floor != 0:
            ii = np.random.randint(low=0, high=len(self.mass_array), size=N_floor)
            
        if np.random.uniform(low=0.0, high=1.0, size=1) <= N_rest:
            iii       = np.random.randint(low=0, high=len(self.mass_array), size=1)
            N         = int(N_floor + 1)
        else:
            N         = int(N_floor)
        
        print(self.t, self.integ.step, A_norm, N)
        
        if N == 0:
            pass
        else:
            if self.cluster_masses is None:
                if N_floor > 0:            
                    
                    m = len(self.ssfr_params[:,0])
                    
                    self.cluster_masses              = np.ma.zeros(N)
                    self.cluster_ssfr_p              = np.ma.zeros([m, N])
                    self.cluster_masses[0:N_floor]   = np.ma.copy(self.mass_array[ii])
                    self.cluster_ssfr_p[:,0:N_floor] = np.ma.copy(self.ssfr_params[:,ii])
                    
                if N == (N_floor+1) and N_floor >0:
                    self.cluster_masses[-1]        = np.ma.copy(self.mass_array[iii])
                    self.cluster_ssfr_p[:,-1]      = np.ma.copy(self.ssfr_params[:,iii]).flatten()
                elif N == (N_floor+1):
                    self.cluster_masses            = np.ma.copy(self.mass_array[iii])
                    self.cluster_ssfr_p            = np.ma.copy(self.ssfr_params[:,iii]).flatten()
                    
                self.infall_z                      = np.ma.zeros(N) + self.z
                self.infall_Ms                     = np.ma.copy(self.cluster_masses)
                self.infall_Mh                     = self.find_M_halo(self.infall_Ms, self.z)      ##### self.z
                #self.infall_Mh                    = self.p.map(self.find_M_halo, ([ms,self.z] for ms in self.infall_Ms))
                
                if self.oc_flag and N!=0:
                    td              = np.ma.array(self.t_delay([self.cluster_masses, self.cluster_ssfr_p, self.infall_Mh, self.z]))
                    self.death_date = cosmo.lookback_time(self.z).value - td
                
                self.mass_history.append(self.cluster_masses)
                
            else:
                n                             = len(self.cluster_masses)
                m                             = len(self.ssfr_params[:,0])
                
                new_arr_mass                  = np.ma.zeros(n + N)
                new_arr_ssfr_p                = np.ma.zeros([m, n + N])
                new_arr_z                     = np.ma.zeros(n + N)
                new_arr_Ms                    = np.ma.zeros(n + N)
                new_arr_Mh                    = np.ma.zeros(n + N)
                
                new_arr_flag_MQ               = np.ma.zeros(n + N)
                
                new_arr_mass[0:n]             = self.cluster_masses[:]
                new_arr_ssfr_p[:,0:n]         = self.cluster_ssfr_p[:,:]
                new_arr_z[0:n]                = self.infall_z[:]
                new_arr_Ms[0:n]               = self.infall_Ms[:]
                new_arr_Mh[0:n]               = self.infall_Mh[:]
                new_arr_flag_MQ[0:n]          = self.MQ_flags[:]
                
                if N_floor > 0:               
                    new_arr_mass[n:n+N_floor]     = np.ma.copy(self.mass_array[ii])
                    new_arr_ssfr_p[:,n:n+N_floor] = np.ma.reshape(self.ssfr_params[:,ii], (m,-1))#, copy=True)
                    
                if N == (N_floor+1):
                    new_arr_mass[-1]            = np.ma.copy(self.mass_array[iii])
                    new_arr_ssfr_p[:,-1]        = np.ma.copy(self.ssfr_params[:,iii]).flatten()
                
                if self.oc_flag:
                    new_arr_flag_OC             = np.ma.zeros(n + N)
                    new_arr_flag_OC[0:n]        = self.OC_flags[:]
                    self.OC_flags               = new_arr_flag_OC
                    
                
                new_arr_z[n:n+N]                = np.ma.zeros(N) + self.z
                new_arr_Ms[n:n+N]               = np.ma.copy(new_arr_mass[n:n+N])
                new_arr_Mh[n:n+N]               = self.find_M_halo(new_arr_mass[n:n+N], self.z) ###self.z 
                #new_arr_Mh[n:n+N]              = p.map(self.find_M_halo, ([ms,self.z] for ms in new_arr_mass[n:n+N]))
                               
                self.cluster_masses = new_arr_mass
                self.cluster_ssfr_p = new_arr_ssfr_p
                self.infall_z       = new_arr_z
                self.infall_Ms      = new_arr_Ms
                self.infall_Mh      = new_arr_Mh
    
                self.MQ_flags       = new_arr_flag_MQ
    
                self.mass_history.append(self.cluster_masses)
    
    def evolve_cluster(self, p):
        if self.z <= 3:
            k_minus = 0.027/4/np.sqrt(1+self.z)/1e9     #merging is turned-off at z>3, PENG+2010
        else:
            k_minus = 0
        
        mass_array  = np.ma.copy(self.cluster_masses)
        ssfr_params = np.ma.copy(self.cluster_ssfr_p)
        
        mass_mask = np.logical_not(mass_array.mask)
        
        if self.cluster_masses is None or len(mass_array[mass_mask]) == 0:
            pass
        else:
            inits                 = (mass_array[mass_mask], ssfr_params[:,mass_mask])
            mass_array[mass_mask] = self.integ.RK45(p, inits, self.t, force=True, mpp=False, analytic=False)
            
            death_func    = np.minimum( self.eta_m(np.ma.log10(mass_array), self.z, ssfr_params) * self.integ.step *1e9, np.ones(mass_array.shape))
            prob_array    = np.ones(mass_array.shape)
            prob_array[mass_mask]  = np.random.uniform(size=mass_array.count())
            mass_array    = np.ma.masked_where(death_func > prob_array, mass_array)               
            
            self.MQ_flags = [death_func > prob_array][0]
            
            if self.oc_flag:
                self.update_death_date(mass_array, ssfr_params)
                mass_array    = np.ma.masked_where(self.death_date > self.t, mass_array)
                self.OC_flags = [self.death_date > self.t][0]
                print(len(self.death_date[self.death_date > self.t]), len(self.death_date))
        
            self.cluster_masses = mass_array
    
    def update_death_date(self, mass_array, ssfr_params):
        tt = time()
        
        N = mass_array.shape[0]
        n = self.death_date.shape[0]
                
        dn = int(N - n)
    
        if dn == 0:
            death_date      = self.death_date
        else:
            death_date      = np.ma.zeros(N)
            death_date[0:n] = self.death_date[:]
            
            infall_times    = cosmo.lookback_time(self.infall_z[n:N]).value
            delay_times     = self.t_delay([mass_array[n:N], ssfr_params[:,n:N], self.infall_Mh[n:N], self.infall_z[n:N]])
            #delay_times     = pool.map( self.t_delay, ([ms,mh,z] for (ms,mh,z) in zip(mass_array[n:N], self.infall_Mh[n:N], self.infall_z[n:N])))
            death_date[n:N] = infall_times - delay_times
        
        not_masked   = np.logical_not(mass_array.mask)
        
        
        
        if len(mass_array[not_masked]) == 0:
            pass
        else:
            #print(mass_array[not_masked], len(mass_array[not_masked]))
            
            infall_times = cosmo.lookback_time(self.infall_z[not_masked]).value
            delay_times  = self.t_delay([mass_array[not_masked], ssfr_params[:,not_masked], self.infall_Mh[not_masked], self.infall_z[not_masked]])
            
            #delay_times = pool.map( self.t_delay, ([ms,mh,z] for (ms,mh,z) in zip(mass_array[not_masked], self.infall_Mh[not_masked], self.infall_z[not_masked])))
            
            death_date[not_masked] = infall_times - delay_times
        
        self.death_date        = death_date
            
        print('update death date: {}'.format(time() - tt))
    
    def update_step(self):
        self.t -= self.integ.step
        self.z  = z_at_value(cosmo.age,(cosmo.age(0).value - self.t)*u.Gyr, zmin=-1e-6)
    
    def parse_masked_mass_field(self):
        temp       = np.ma.copy(self.mass_array)
        
        mask                     = self.mass_array.mask
        temp.mask                = np.ma.nomask
        self.final_mass_field_SF = np.log10(temp[np.logical_not(mask)])
        self.final_mass_field_Q  = np.log10(temp[mask])
    
    def parse_masked_mass_cluster(self):
        temp       = np.ma.copy(self.cluster_masses)
        
        mask                       = self.cluster_masses.mask
        temp.mask                  = np.ma.nomask
        self.final_mass_cluster_SF = np.log10(temp[np.logical_not(mask)])
        self.infall_z_SF           = self.infall_z[np.logical_not(mask)]
        
        self.final_mass_cluster_Q  = np.log10(temp[mask])
        self.infall_z_Q            = self.infall_z[mask]
    
    def DE(self, item, t):
        mass, ssfr_params = item[0], item[1]
        z = z_at_value(cosmo.age,(cosmo.age(0).value - t)*u.Gyr, zmin=-1e-6) 
        return mass*(self.sSFR(np.log10(mass), z, ssfr_params)) *1e9
    
    def eta_m(self, logMs, z, ssfr_params):      # mass-dependent death function from PENG+2010
        M_ref = np.power(10,10.6)
        return self.sSFR(logMs,z, ssfr_params) * np.power(10, logMs) / M_ref
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    
    ### PENG Analytic Model ###
    def gen_field_analytic(self,p):
        self.logMStar_setup = np.arange(self.logM_min, 12, 0.01)
        
        phi_init_sf = self.schechter_SMF_func(self.logMStar_setup)
        phi_init_q  = np.zeros(len(phi_init_sf))
        
        t_init      = cosmo.lookback_time(self.z_init).value #gyr
        t_final     = cosmo.lookback_time(self.z_final).value #gyr
        
        t  = t_init
        z  = self.z_init

        phi_sf      = [phi_init_sf]
        phi_q       = [phi_init_q]
        z_range     = [z]

        integ_an  = integration_utils(self.DE_2, hmax=0.1, hmin=1e-5, htol=1e-9)        #_an == analytic
        condition = True                                                              # Always run at least once
        force     = False
        
        while t >= t_final and condition:
            inits                   = np.array([phi_init_sf,phi_init_q], copy=True)
            output                  = integ_an.RK45(p, inits, t, force, mpp=False, analytic=True)
            phi_init_sf, phi_init_q = output[0,:], output[1,:]
                        
            phi_sf.append(np.copy(phi_init_sf))
            phi_q.append(np.copy(phi_init_q))
            
            if (t - integ_an.step) > t_final:
                pass
            else:
                integ_an.step = t - t_final
                force         = True
                condition     = False
            
            
            print(t, integ_an.step)
            
            t -= integ_an.step
            z  = z_at_value(cosmo.age,(cosmo.age(0).value - t)*u.Gyr, zmin=-1e-6)
            
            z_range.append(z)
        
        self.phi_sf_interp_an = interp2d(self.logMStar_setup, z_range, phi_sf)
        self.phi_q_interp_an  = interp2d(self.logMStar_setup, z_range, phi_q)
    
    def DE_2(self, inits, t):
        phi_sf, phi_q = inits[0,:], inits[1,:]
        z = z_at_value(cosmo.age,(cosmo.age(0).value - t)*u.Gyr, zmin=-1e-6) 
        dn_blue = self.dN_blue(self.logMStar_setup, z, phi_sf)
        dn_red  = self.dN_red(self.logMStar_setup, z, phi_sf)
        return np.array([dn_blue, dn_red])
    
    def dN_blue(self, logMs, z, N_b):
        alpha    = -1.3
        beta     = -0
        lambda_m = (np.power(10,logMs)/np.power(10,10.6)) * self.sSFR_an(logMs,z)*1e9 
        return N_b * (-(1 + alpha + beta - (np.power(10,logMs)/np.power(10,10.6)))*self.sSFR_an(logMs, z)*1e9 - lambda_m) # - 0.027/4*(1+z)**(1.2))
    
    def dN_red(self, logMs, z, N_b):
        lambda_m = (np.power(10,logMs)/np.power(10,10.6)) * self.sSFR_an(logMs,z)*1e9 
        return N_b * lambda_m #- 0.027/4*(1+z)**(1.2)*N_r
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
    
    def gen_ssfr_peng_params(self, n):
        return np.zeros([1,n])

    def sSFR_schreiber(self, logMs, z, ssfr_params):       # Schreiber+2015
        Ms  = np.power(10,logMs)
        r   = np.log10(1+z)
        m   = logMs - 9
        try:
            m_0 = ssfr_params[0,:]
            m_1 = ssfr_params[1,:]
            a_0 = ssfr_params[2,:]
            a_1 = ssfr_params[3,:]
            a_2 = ssfr_params[4,:]
        except:
            m_0 = ssfr_params[0]
            m_1 = ssfr_params[1]
            a_0 = ssfr_params[2]
            a_1 = ssfr_params[3]
            a_2 = ssfr_params[4]
                
        lsfr = m - m_0 + a_0 * r - a_1 * np.maximum(np.zeros(m.shape), m-m_1-a_2 * r)**2
        ssfr = 10**(lsfr) / Ms 
        
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
    
    def gen_ssfr_schreiber_params(self, n):
        m_0 = np.random.normal(loc=0.50, scale=0.07, size=n)
        m_1 = np.random.normal(loc=0.36, scale=0.30, size=n)
        a_0 = np.random.normal(loc=1.50, scale=0.15, size=n)
        a_1 = np.random.normal(loc=0.30, scale=0.08, size=n)
        a_2 = np.random.normal(loc=2.50, scale=0.60, size=n)
        
        return np.array([m_0, m_1, a_0, a_1, a_2])
        
    def sSFR_speagle(self, logMs, z, ssfr_params):
        z_temp     = np.minimum(z,6)
        logMs_temp = np.maximum(logMs, 8.5) 
        
        t   = cosmo.lookback_time(9999).value - cosmo.lookback_time(z_temp).value  ##  WARNING:: TRY cosmo.age(z).value instead!!
        
        try:
            x_1 = ssfr_params[0,:]
            x_2 = ssfr_params[1,:]
            x_3 = ssfr_params[2,:]
            x_4 = ssfr_params[3,:]
        except:
            x_1 = ssfr_params[0]
            x_2 = ssfr_params[1]
            x_3 = ssfr_params[2]
            x_4 = ssfr_params[3]
        
        logSFR = (x_1 - x_2*t) * logMs_temp - (x_3 - x_4*t)
        ssfr   = np.power(10, logSFR) / np.power(10,logMs_temp)
        
        return  ssfr ## peryear
    
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
    
    def gen_ssfr_speagle_params(self, n):
        x_1 = np.random.normal(loc=0.840, scale=0.020, size=n)
        x_2 = np.random.normal(loc=0.026, scale=0.003, size=n)
        x_3 = np.random.normal(loc=6.510, scale=0.240, size=n)
        x_4 = np.random.normal(loc=0.110, scale=0.030, size=n)

        return np.array([x_1, x_2, x_3, x_4])
        
    def update_ssfr_params(self):
        try:
            delta_t    = self.t_old - self.t
        except:
            self.t_old = cosmo.lookback_time(self.z_init).value
            delta_t    = self.t_old - self.t
        
        print(delta_t)
        
        if delta_t >= 0.1: #Gyr
            n                  = len(self.sf_masses)
            self.ssfr_params   = self.gen_ssfr_params(n)
            
            m                   = len(self.cluster_masses)
            self.cluster_ssfr_p = self.gen_ssfr_params(m)
            
            self.t_old          = self.t
            print('*new ssfr params*')
        else:
            pass

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    
    ### Overconsumption Delay times ###
    def t_delay(self, item):
        M_star, ssfr_p, M_halo, z = item[0],item[1],item[2],item[3]
        
        #delay time params
        f_bar   = 0.17
        f_strip = 0
        R       = 0.4
        
        logMs   = np.ma.log10(M_star)
        
        f_star    = M_star/M_halo
        #f_cold    = 0.1*(1+np.minimum(z,2))**2 * f_star
        f_cold    = self.M_cold(logMs, z, ssfr_p)/M_halo
        
        num       = f_bar - f_cold - f_star * (1 + self.oc_eta*(1+R)) - f_strip
        den       = f_star * (1 - R + self.oc_eta) * self.sSFR(logMs, z, ssfr_p)
        return (num/den)/(1e9) #gyr
        
    def t_delay_2(self, M_halo, z):
        
        #delay time params
        f_bar   = 0.17
        f_strip = 0
        R       = 0.4
        
        M_star     = self.M_star(M_halo,z)
        logMs      = np.log10(M_star)
        
        f_star     = M_star/M_halo
        f_cold    = 0.1*(1+np.minimum(z,2))**2 * f_star
        #f_cold    = self.M_cold(logMs, z)/M_halo
        
        num       = f_bar - f_cold - f_star * (1 + self.oc_eta*(1+R)) - f_strip
        den       = f_star * (1 - R + self.oc_eta) * self.sSFR(logMs, z)
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
        
        temp_mass_array      = np.ma.copy(self.mass_array)
        temp_mass_array.mask = np.ma.nomask
        temp_mass_array      = np.ma.log10(temp_mass_array)
                
        logMh_min = self.find_logM_halo(np.ma.min(temp_mass_array),z)           ###z 
        logMh_max = self.find_logM_halo(np.ma.max(temp_mass_array),z)           ### z
        
        logMhalo_range  = np.arange(logMh_min, logMh_max, 0.0005)
        
        Mstar_from_halo = self.M_star(np.power(10, logMhalo_range),z=z) ## Msol (NOT log!)
        logMs           = np.log10(Mstar_from_halo)
        phi_a_interp    = self.phi_sf_interp(logMs) + self.phi_q_interp(logMs)
        x               = np.power(10, logMhalo_range)
        y               = phi_a_interp*np.power(10,logMhalo_range)*self.dMs_dMh(np.power(10,logMhalo_range),z)
        
        integral = np.trapz(x=x,y=y, dx=np.diff(x))
        return integral
        
    
    def process_manager(self, func, arr):
        pass
    
    
    
    
    
    
    
    
    
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
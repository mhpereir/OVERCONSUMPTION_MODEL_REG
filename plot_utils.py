import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16, 'xtick.labelsize':14, 'ytick.labelsize':14})

from astropy.cosmology import Planck15 as cosmo, z_at_value

class plot_results:
    def __init__(self, params, model_c, model_f): 
        
        plot_flag    = params['model_setup']['show_plots']
        
        if plot_flag == "True":
            plotting_flag = True
        else:
            plotting_flag = False
        
        if plotting_flag:
            
            self.init_pop(model_c)
            self.final_pop(model_c, model_f)
            
            self.QFs_QE(model_c, model_f)
            
            #self.delay_times(model_c)
            
            #self.history_tracks(model_c)
            

    def init_pop(self, model):
        '''
        Initial star forming field
        '''
        x_range_init = np.arange(model.logM_min,model.logM_max+0.05,0.1)
        
        init_phi_hm  = model.open_HMF_file()
        
        fig,ax = plt.subplots(tight_layout=True)
        y_init, x_init,_ = ax.hist(model.hmass_init, log=True, bins=x_range_init)
        ax.semilogy(x_range_init, init_phi_hm(x_range_init)*y_init[abs(x_init[:-1] - 9) < 0.001]/init_phi_hm(9) )
        ax.set_xlabel('Halo Mass [log($M_h/M_\odot$)]')
        ax.set_ylabel('$\Phi_{Field}$')
        fig.savefig('./images/SMF_init.png', dpi=220)
        
    def final_pop(self, model_c, model_f):
        '''
        Parameters
        ----------
        model : class
            Contains the evolved cluster and field data.

        Returns
        -------
        figures: 
        '''
        x_range = np.arange(8,12,0.1)
        
        fig,ax = plt.subplots(tight_layout=True)
        y_sf,x_sf = np.histogram(model_f.final_mass_field_SF, bins=np.arange(6,14,0.1))
        y_q,x_q   = np.histogram(model_f.final_mass_field_Q, bins=np.arange(6,14,0.1))
        x_midp    = (x_sf[1:] + x_sf[:-1])/2
        
        ax.scatter(x_midp, y_sf, c='b', marker='x')
        ax.semilogy(x_midp, y_sf, color='b', alpha=0.5)
        ax.scatter(x_midp, y_q, c='r', marker='x')
        ax.semilogy(x_midp, y_q, color='r', alpha=0.5)
        
        #ax.semilogy(x_range, model_f.phi_sf_interp_an(x_range,z_final) * y_sf[abs(x_midp[:] - 10) < 0.1]/model_f.phi_sf_interp_an(10,z_final) )
        #ax.semilogy(x_range, model_f.phi_q_interp_an(x_range,z_final)  * y_sf[abs(x_midp[:] - 10) < 0.1]/model_f.phi_sf_interp_an(10,z_final) )
        ax.set_xlabel('Stellar Mass [log($M_*/M_\odot$)]')
        ax.set_ylabel('$\Phi_{Field}$')
        ax.set_xlim([8.5,12])
        fig.savefig('./images/SMF_field.png', dpi=220)
        
        fig,ax = plt.subplots(tight_layout=True)
        y_sf,_  = np.histogram(model_c.final_mass_cluster_SF, bins=np.arange(6,14,0.1))
        y_q,_   = np.histogram(model_c.final_mass_cluster_Q, bins=np.arange(6,14,0.1))
        
        
        ax.scatter(x_midp, y_sf, c='b', marker='x')
        ax.semilogy(x_midp, y_sf, color='b', alpha=0.5)
        ax.scatter(x_midp, y_q, c='r', marker='x')
        ax.semilogy(x_midp, y_q, color='r', alpha=0.5)
        
        # ax.semilogy(x_range, model_c.phi_sf_interp_an(x_range,z_final) * y_sf[abs(x_midp[:] - 10) < 0.1]/model_c.phi_sf_interp_an(10,z_final) )
        # ax.semilogy(x_range, model_c.phi_q_interp_an(x_range,z_final)  * y_sf[abs(x_midp[:] - 10) < 0.1]/model_c.phi_sf_interp_an(10,z_final) )
        ax.set_xlabel('Stellar Mass [log($M_*/M_\odot$)]')
        ax.set_ylabel('$\Phi_{Cluster}$')
        ax.set_xlim([8.5,12])
        fig.savefig('./images/SMF_cluster.png', dpi=220)
        
    def QFs_QE(self, model_c, model_f):
        hist_sf_field, bins   = np.histogram(model_f.final_mass_field_SF, bins=np.arange(6,14,0.1))
        hist_q_field, bins    = np.histogram(model_f.final_mass_field_Q, bins=np.arange(6,14,0.1))
        
        hist_sf_cluster, bins = np.histogram(model_c.final_mass_cluster_SF, bins=np.arange(6,14,0.1))
        hist_q_cluster, bins  = np.histogram(model_c.final_mass_cluster_Q, bins=np.arange(6,14,0.1))
        
        bins_midp = (bins[:-1] + bins[1:])/2
        QF_field  = hist_q_field / (hist_q_field + hist_sf_field)
        QF_cluster  = hist_q_cluster / (hist_q_cluster + hist_sf_cluster)
        
        QE = (QF_cluster - QF_field)/(1 - QF_field)
        
        fig,ax = plt.subplots()
        ax.plot(bins_midp, QF_field)
        ax.plot(bins_midp, QF_cluster)
        ax.set_xlabel('Stellar Mass [log($M_*/M_\odot$)]')
        ax.set_ylabel('Quenched Fraction')
        ax.set_xlim([9,11.5])
        fig.savefig('./images/QFs.png', dpi=220)
        
        fig,ax = plt.subplots()
        ax.plot(bins_midp, QE)
        ax.set_xlim([9,11.5])
        ax.set_xlabel('Stellar Mass [log($M_*/M_\odot$)]')
        ax.set_ylabel('Quenching Efficiency')
        fig.savefig('./images/QE.png', dpi=220)
        
    def delay_times(self, model):
        z_init      = model.z_init
        z_final     = model.z_final
        z_range     = np.arange(z_final, z_init, 0.1)
        
        logMh_range = np.arange(7,15,0.1)
        Mh_range    = np.power(10, logMh_range)
        
        xx   = []
        xx_2 = []
        yy   = []
        #zz   = []
        
        for z in z_range:
            xx.append(np.zeros(len(Mh_range)) + z)
            xx_2.append(np.zeros(len(Mh_range)) + cosmo.lookback_time(z).value)
            yy.append(np.log10(model.M_star(Mh_range, z)))
            #zz.append(cosmo.lookback_time(z).value - model.t_delay_2(Mh_range, z))
        
        xx   = np.array(xx).flatten()
        xx_2 = np.array(xx_2).flatten()
        yy   = np.array(yy).flatten()
        #zz   = np.array(zz).flatten()
        
        fig,ax = plt.subplots(tight_layout=True)
        #contourf_ = ax.tricontourf(xx,yy,zz, np.arange(0,13,0.1), extend='both')
    
        ax.scatter(model.infall_z_Q, model.final_mass_cluster_Q, s=0.1, c='r')
        ax.scatter(model.infall_z_SF, model.final_mass_cluster_SF, s=0.1, c='b')
        
        ax.set_xlabel('Redshift [z]')
        ax.set_ylabel('Stellar Mass [log($M_*$/$M_\odot$)]')
        #ax.set_ylim([8,12])
        #cbar      = fig.colorbar(contourf_, label='Quenched due to OC')
        fig.savefig('./images/cluster_final_m_z.png', dpi=220)
        
        
        
        fig,ax = plt.subplots(tight_layout=True)
        #contourf_ = ax.tricontourf(xx_2,yy,zz, np.arange(0,13,0.1), extend='both')
        try:
            ax.scatter(cosmo.lookback_time(model.infall_z_Q).value, model.final_mass_cluster_Q, s=0.1, c='r')
        except:
            pass
        
        try:
            ax.scatter(cosmo.lookback_time(model.infall_z_SF).value, model.final_mass_cluster_SF, s=0.1, c='b')
        except:
            pass
        
        ax.set_xlabel('Lookback Time [Gyr]')
        ax.set_ylabel('Stellar Mass [log($M_*$/$M_\odot$)]')
        #ax.set_ylim([8,12])
        #cbar      = fig.colorbar(contourf_, label='Quenched due to OC')
        fig.savefig('./images/cluster_final_m.png', dpi=220)
        
        
        
        fig,ax = plt.subplots(tight_layout=True)
        #contourf_ = ax.tricontourf(xx_2,yy,zz, np.arange(0,13,0.1), extend='both')
        
        temp_z      = np.ma.copy(model.infall_z)
        temp_z.mask = np.ma.nomask
        temp_m      = np.ma.copy(model.infall_Ms)
        temp_m.mask = np.ma.nomask
        temp_m      = np.ma.log10(temp_m)
        
        ax.scatter(cosmo.lookback_time(temp_z).value, temp_m, s=0.1, c='r')
        ax.set_xlabel('Lookback Time [Gyr]')
        ax.set_ylabel('Stellar Mass [log($M_*$/$M_\odot$)]')
        #ax.set_ylim([8,12])
        #cbar      = fig.colorbar(contourf_, label='Quenched due to OC')
        fig.savefig('./images/cluster_infall_m.png', dpi=220)
        
    def history_tracks(self, model):
        n  = len(model.mass_history)
        m  = len(model.mass_history[-1])
        
        number_of_lines = 10000
        prob_threshold  = number_of_lines/m
        
        tt = cosmo.lookback_time(np.unique( model.infall_z) )
    
        mass_history = np.ma.zeros([n,m])
        for i,mass_arr in enumerate(model.mass_history):
            mass_history[i,0:int(mass_arr.shape[0])] = mass_arr[:]
        
        mass_history      = np.ma.log10(mass_history[::-1,:])
        
        fig,ax = plt.subplots()
        
        count_mq = 0
        count_oc = 0
        count_sf = 0
        
        print('Expected total: {}'.format(number_of_lines))
        
        for i in range(0,m):
            
            p = np.random.uniform()
            
            if p < prob_threshold:
                
                ax.plot(tt, mass_history[:,i], color='k', alpha=0.01)
                
                # if model.oc_flag:
                #     if OC_flags[i]:
                #         ax.plot(tt, mass_history[:,i], color='C1', alpha=0.01)
                
                # if MQ_flags[i] == 0:
                #     #ax.plot(tt, mass_history[:,i], color='b', alpha=0.01)
                #     pass
                # else:
                #     ax.plot(tt, mass_history[:,i], color='r', alpha=0.01)
                #     pass
                
            
                # final_masses = mass_history[mass_history[:,i].mask == False, i]
                
                # try:
                #     if final_masses[0] > 8.9 and final_masses[0] < 9.1:
                #         temp = True
                #         if model.oc_flag:
                #             if OC_flags[i]:
                #                 ax.plot(tt, mass_history[:,i], color='purple', alpha=0.5)
                #                 count_oc += 1
                #                 temp = False
                                
                        
                #         if MQ_flags[i] == 0 and temp:
                #             ax.plot(tt, mass_history[:,i], color='C0', alpha=0.5)
                #             count_sf += 1
                #         elif temp:
                #             ax.plot(tt, mass_history[:,i], color='r', alpha=0.5)
                #             count_mq += 1
                #             pass
                # except:
                #     pass
            
            
            
                # final_masses = mass_history[mass_history[:,i].mask == False, i]
                
                # try:  #some galaxies infall quenched. those will fail this step
                #     if final_masses[0] > 8.9 and final_masses[0] < 9.1:
                #         ax.plot(tt, mass_history[:,i], color='C0', alpha=0.1)
                # except:
                #     pass
        
        print('Mass Quenched: {}'.format(count_mq))
        print('Over Consumed: {}'.format(count_oc))
        print('Star Forming: {}'.format(count_sf))
        
        ax.set_xlabel('Lookback Time [Gyr]')
        ax.set_ylabel('Stellar Mass [log($M_*/M_\odot$)]')
        fig.savefig('./images/history_path.png', dpi=220)
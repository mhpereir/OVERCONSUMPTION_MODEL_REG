import numpy as np
import sys

from time import time

class integration_utils():
    def __init__(self, DE, hmax, hmin, htol):
        self.hmax = hmax     #1 Gyr
        self.hmin = hmin     #1 Myr
        self.htol = htol
        
        self.DE   = DE
        self.step = self.hmin
        
    def RK45(self, inits,r,force): # analytic tells whether the Rk45 is being used for the gen_analytic_field or not
        f          = self.DE
        #ignore_min = False
        start_ = time()
        
        for i in range(2):
            
            if (r - self.step) < 0 and not force:
                self.step = r
                        
            k1 = self.step*f(inits                                                              , r                 )
            k2 = self.step*f(inits + 1/4.*k1                                                    , r-1/4.*self.step  )
            k3 = self.step*f(inits + 3/32.*k1 + 9/32.*k2                                        , r-3/8.*self.step  )
            k4 = self.step*f(inits + 1932/2197.*k1 - 7200/2197.*k2 + 7296/2197.*k3              , r-12/13.*self.step)
            k5 = self.step*f(inits + 439/216.*k1 - 8*k2 + 3680/513.*k3 - 845/4104.*k4           , r-self.step       )
            k6 = self.step*f(inits - 8/27.*k1 + 2*k2 - 3544/2565.*k3 + 1859/4104.*k4 - 11/40.*k5, r-1/2.*self.step  )
        
            y1 = inits + 25/216.*k1 + 1408/2565.*k3 + 2197/4104.*k4 - 1/5.*k5
            z1 = inits + 16/135.*k1 + 6656/12825.*k3 + 28561/56430.*k4 - 9/50.*k5 + 2/55.*k6        
            
            if i == 0 and not force:
                s = np.ma.min( (self.htol /(2*abs(z1[0,:]-y1[0,:])))**(1/4.)  )         
                
                if abs(s) == np.inf:
                    s = 1
                elif np.isnan(s):
                    s = 0.5
                
                self.step *= s              # optimal step is step*s
                
                #### Keep adaptive step in check. Not too small, not too big. force first step to be hmin
                if self.step > self.hmax:
                    self.step = self.hmax
                elif self.step < self.hmin:# and not ignore_min:
                    self.step = self.hmin
            
            elif force:
                break
        
        print('Ellapsed time: {} for arr size: {}'.format(time() - start_, y1.shape))
        
        return y1


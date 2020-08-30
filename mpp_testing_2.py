# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 10:09:36 2020

@author: Matthew Wilson
"""



from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
from time import time,sleep
plt.rcParams.update({'font.size': 16, 'xtick.labelsize':14, 'ytick.labelsize':14})

def f(arr):
    return np.sqrt( np.power(10, np.log10(np.exp(np.log(arr)))) ** 2)



def main(number_of_task, arr,p):
    number_of_task = int(number_of_task)
    _start = time()
        
    dn = np.floor(n/number_of_task)
    list_of_arr = [arr[int(dn*i):int(dn*(i+1))] for i in range(number_of_task)]
    
    out = np.array(p.map(f, (arr for arr in list_of_arr)))
    
    print('Total time: {:.2f}. {}'.format(time() - _start, number_of_task))
    
    return time() - _start
    p.close()
    p.join()

if __name__ == '__main__':
    p        = Pool(4)
    p_range  = range(1,5)
    sleep(2)
    
    n_range = np.power(10, range(4,8))
    fig,ax   = plt.subplots()
    fig1,ax1 = plt.subplots()
    ax2 = ax.twinx()
    for n in n_range:
        
        ax.set_title('N = {}'.format(n))
        
        arr_size = n/np.array(p_range)
        arr      = np.arange(1,n)
        
        times = [main(i,arr,p) for i in p_range]
        
        ax.plot(p_range, times)
        ax2.plot(p_range, arr_size)
        
        ax1.semilogy(times, arr_size )
        ax1.scatter(times, arr_size, s=10)
    
    
    
    
    ax.set_ylabel('Time [s]')
    ax2.set_ylabel('Arr Size')
    
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Size of Batch p/ Core')
    
    plt.show()
    
    p.close()
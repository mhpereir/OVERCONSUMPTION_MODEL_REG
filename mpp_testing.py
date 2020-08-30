# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 09:32:20 2020

@author: Matthew Wilson
"""

from multiprocessing import Process, Queue

import numpy as np
import time
import sys

def reader_proc(queue):
    ## Read from the queue; this will be spawned as a separate Process
    while True:
        msg = queue.get()         # Read from the queue and do nothing
        if (msg == 'DONE'):
            break

def f(x):
    arr = np.arange(0,x)
    return np.sin(arr)

def writer(queue):
    ## Write to the queue
    n = 1e8
    for count in [n,n,n,n]:
        start = time.time()
        queue.put(f(count))             # Write 'count' numbers into the queue
        print(time.time() - start)
    queue.put('DONE')

if __name__=='__main__':
    pqueue = Queue() # writer() writes to pqueue from _this_ process
    __start = time.time()  
          
    writer(pqueue)    # Send a lot of stuff to reader()
    
    print('Time to write: ', time.time() - __start)
    ### reader_proc() reads from pqueue as a separate process
    reader_p = Process(target=reader_proc, args=((pqueue),))
    reader_p.daemon = True
    reader_p.start()        # Launch reader_proc() as a separate python process

    _start = time.time()
    reader_p.join()         # Wait for the reader to finish
    print("Sending numbers to Queue() took {} seconds".format((time.time() - _start)))
        
    print('Total time: ', time.time() - __start)
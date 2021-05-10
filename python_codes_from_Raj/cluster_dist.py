import wrapper as ts
import glob
import array as ar
import numpy as np

icount=0
n=150
cl_array=[]
for i in range(n):
    cl_array.append(0)
for file in glob.glob("timestep_0000[0-3][0-9].vtu"):
    vesicle=ts.parseDump(file)
    cstlist=ts.init_cluster_list()
    ts.clusterize_vesicle(vesicle, cstlist)    
    for j in range(cstlist.contents.n):
        size=cstlist.contents.cluster[j].contents.nvtx
        cl_array[size-1]+=1
    icount+=1
for i in range(n):
    f=open('cluster_distribution_c_100_E_0_F_pt_8.dat','a')
    f.write('{}, {} \n'.format(i+1,cl_array[i]/icount))
                        

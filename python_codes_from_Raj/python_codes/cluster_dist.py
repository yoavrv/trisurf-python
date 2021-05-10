import wrapper as ts
import glob
import array as ar
import numpy as np

icount=0
n=350
cl_array=[]
for i in range(n):
    cl_array.append(0.000)
for file in glob.glob("../../E_1_pt_5_2_pt_5/f19/timestep_000[0-1][0-9][0-9].vtu"):
#for file in glob.glob("../../c_300/E_pt_001/timestep_0001[0-9][0-9].vtu"):
    vesicle=ts.parseDump(file)
    cstlist=ts.init_cluster_list()
    ts.clusterize_vesicle(vesicle, cstlist)    
    for j in range(cstlist.contents.n):
        size=cstlist.contents.cluster[j].contents.nvtx
        cl_array[size-1]+=1.0000
    icount+=1
for i in range(n):
    f=open('cluster_dist_passive_E_2_pt_5_c_350.dat','a')
    f.write('{}, {} \n'.format(i+1,cl_array[i]/icount))
                        

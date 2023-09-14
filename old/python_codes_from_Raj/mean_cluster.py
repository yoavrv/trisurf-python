import wrapper as ts
import glob
import array as ar
import numpy as np

icount=0
sum=0.000000
sum_sq=0.00000
rho=90.0/1447.0
for file in glob.glob("timestep_000[0][0-9][0-9].vtu"):
    vesicle=ts.parseDump(file)
    cstlist=ts.init_cluster_list()
    ts.clusterize_vesicle(vesicle, cstlist)    
    for j in range(cstlist.contents.n):
        size=cstlist.contents.cluster[j].contents.nvtx
        sum+=size
        sum_sq+=size*size
        icount+=1
error=(sum_sq/icount - (sum/icount)**2)**0.5
f=open('rho_vs_mean_cluster_E_pt_3.dat','a')
f.write(' {}, {}, {} \n'.format(rho,sum/icount,error))
print(sum/icount,error)                        

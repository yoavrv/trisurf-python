import wrapper as ts
import glob
import array as ar
import numpy as np

icount=0
avg=0.000000
avg2=0.0000000
Nc=400
rho=Nc/1447.0
f=open('rho_vs_mean_of_largest_cluster_E_pt_50_F_0.dat','a')
for file in glob.glob("../../E_pt_20_50/f20/timestep_000[1][0-9][0-9].vtu"):
#for file in glob.glob("../../second_pancake_transition/F_3_pt_50_4/f20/timestep_000[0-1][0-9][0-9].vtu"):
    vesicle=ts.parseDump(file)
    cstlist=ts.init_cluster_list()
    ts.clusterize_vesicle(vesicle, cstlist)  
    max_size=0
    for j in range(cstlist.contents.n):
        size=cstlist.contents.cluster[j].contents.nvtx
        if (size>max_size):
            max_size=size
    avg+=max_size
    avg2+=max_size**2
    icount+=1
error=(avg2/icount - (avg/icount)**2)**0.5
f.write(' {}, {}, {} \n'.format(rho,avg/icount,error))
print(avg/icount,error)                        

import wrapper as ts
import glob
import array as ar
import numpy as np

nd=100
dd=(nd/3.14)+0.0001
ar.r_theta=[]
ar.c=[]
icount=0
a=0
for i in range(nd+1):
    ar.r_theta.append(0)
    ar.c.append(0)
for file in glob.glob("../../E_pt_20_50/f18/timestep_000[0-1][0-9][0-9].vtu"):
#for file in glob.glob("../../more_data_close_c/E_1_pt_5_2_2_pt_5/f12/timestep_000[0-1][0-9][0-9].vtu"):
    vesicle=ts.parseDump(file)

    cmx=0.0000
    cmy=0.0000
    cmz=0.0000
    for i in range(vesicle.contents.vlist.contents.n):
        x=vesicle.contents.vlist.contents.vtx[i].contents.x
        y=vesicle.contents.vlist.contents.vtx[i].contents.y
        z=vesicle.contents.vlist.contents.vtx[i].contents.z
        cmx+=x
        cmy+=y
        cmz+=z
    cmx/=vesicle.contents.vlist.contents.n
    cmy/=vesicle.contents.vlist.contents.n
    cmz/=vesicle.contents.vlist.contents.n

    for i in range(vesicle.contents.vlist.contents.n):
        c=vesicle.contents.vlist.contents.vtx[i].contents.c
        if (c>0):
            x=vesicle.contents.vlist.contents.vtx[i].contents.x - cmx
            y=vesicle.contents.vlist.contents.vtx[i].contents.y - cmy
            z=vesicle.contents.vlist.contents.vtx[i].contents.z - cmz
            s=(x**2+y**2)**(0.5)
            theta=np.arctan(z/s)
            k=int(theta*dd)
            ar.r_theta[k]+=1.00
            ar.c[k]+=1.00

    icount+=1          
for i in range(nd+1):
    ttheta=(i/dd)
    if(ttheta>1.57):
        ttheta-=3.14        
    f=open('angular_cluster_distribution_E_pt_50_c_300.dat','a')
    f.write('{}, {}, {}, {} \n'.format(i,ttheta,ar.c[i]/icount,ar.r_theta[i]/icount))
#f=open('angular_cluster_distribution_E_pt_50_c_300.dat','a')
#f.write('{}\n'.format(a))


import wrapper as ts
import glob
import array as ar
import numpy as np

icount=0
avg=0.00000000000
avg_sq=0.000000000
Nc=225
rho=Nc/1447.0
theta_min=-1.57
theta_max=-0.0091918
#for file in glob.glob("../../E_pt_20_50/f17/timestep_000[0-1][0-9][0-9].vtu"):
for file in glob.glob("../../more_data_close_c/E_pt_50_pt_75_1/f4/timestep_000[0-1][0-9][0-9].vtu"):
    vesicle=ts.parseDump(file)
    rc=0.000000000000

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
            if ((theta < theta_max) and (theta > theta_min)):
                rc+=1.00000000
    
    avg+=rc/Nc
    avg_sq+=(rc/Nc)**2  
    icount+=1

avg/=icount
avg_sq/=icount
error=(avg_sq - avg**2)**0.5
                  
f=open('rho_vs_rho_c_E_pt_50_passive_c0_1.dat','a')
f.write('{}, {}, {} \n'.format(rho,avg,error))


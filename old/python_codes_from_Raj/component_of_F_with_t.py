import wrapper as ts
import glob
import array as ar
import numpy as np

icount=0
for file in glob.glob("timestep_000[0][0-9][0-9].vtu"):
    name=int(file[10:15]) 
    vesicle=ts.parseDump(file)
    fx=0.000000000000
    fy=0.000000000000
    fz=0.000000000000
    for i in range(vesicle.contents.vlist.contents.n):
        if(vesicle.contents.vlist.contents.vtx[i].contents.c > 0.00000000000001):
            xnorm=0.00000000000
            ynorm=0.00000000000
            znorm=0.00000000000
            for j in range (vesicle.contents.vlist.contents.vtx[i].contents.tristar_no):
                xnorm+=vesicle.contents.vlist.contents.vtx[i].contents.tristar[j].contents.xnorm
                ynorm+=vesicle.contents.vlist.contents.vtx[i].contents.tristar[j].contents.ynorm
                znorm+=vesicle.contents.vlist.contents.vtx[i].contents.tristar[j].contents.znorm
            normal=(xnorm**2 + ynorm**2 + znorm**2)**0.5
            xnorm/=normal
            ynorm/=normal
            znorm/=normal
            fx+=xnorm
            fy+=ynorm
            fz+=znorm
    f=open('component_of_F_with_t_E_pt_75_c_50.dat','a')
    f.write('{}, {}, {}, {} \n'.format(name,fx,fy,fz))

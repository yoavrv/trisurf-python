import wrapper as ts
import glob
import array as ar
import numpy as np

icount=0
kappa=20.0
F=4.0
E=0.50
W_ad=0.000000000
W_b=0.0000000000
W_d=0.0000000000
W_a=0.0000000000
n=150
rho=n/1447.0

for file in glob.glob("timestep_0000[0-9][0-9].vtu"):
    vesicle=ts.parseDump(file)
    cstlist=ts.init_cluster_list()
    ts.clusterize_vesicle(vesicle, cstlist)  
    ts.mean_curvature_and_energy(vesicle)	
    z_min=-10.724916

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
        z=vesicle.contents.vlist.contents.vtx[i].contents.z
        if((z-z_min)<1.0):
            W_ad-=E

        W_b+=vesicle.contents.vlist.contents.vtx[i].contents.energy

        if (vesicle.contents.vlist.contents.vtx[i].contents.c > 0.00000000000001):
            for j in range (vesicle.contents.vlist.contents.vtx[i].contents.neigh_no):
                if (vesicle.contents.vlist.contents.vtx[i].contents.neigh[j].contents.c > 0.00000000000001):
                    W_d-=1.0
                    
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
            x=vesicle.contents.vlist.contents.vtx[i].contents.x - cmx
            y=vesicle.contents.vlist.contents.vtx[i].contents.y - cmy
            z=vesicle.contents.vlist.contents.vtx[i].contents.z - cmz
            W_a-=F*(xnorm*x + ynorm*y + znorm*z)
    icount+=1
if (icount>0):
    W_ad/=icount
    W_b/=icount
    W_d/=icount
    W_a/=icount
f=open('rho_vs_Wad_Wb_Wd_Wa_E_0_F_0.dat','a')
f.write('{}, {}, {}, {}, {} \n'.format(rho,W_ad,W_b,W_d,W_a))
                        

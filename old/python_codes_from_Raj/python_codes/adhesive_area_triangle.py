import wrapper as ts
import glob
import array as ar
import numpy as np

icount=0
only_area=0.00000000000
avg=0.000000000000
sq_avg=0.000000000
E=1.75
rho=400/1447.0
for file in glob.glob("../../E_1_pt_25_1_pt_75/f20/timestep_000[0-1][0-9][0-9].vtu"):
#for file in glob.glob("../../more_data_close_c/E_1_pt_25_1_pt_75_2_pt_25/f14/timestep_000[0-1][0-9][0-9].vtu"):
    vesicle=ts.parseDump(file)
    ts.vesicle_area(vesicle)
    z_min=-10.724916
    total_area=vesicle.contents.area
    area=0
#    for i in range(vesicle.contents.vlist.contents.n):
#        z=vesicle.contents.vlist.contents.vtx[i].contents.z
#        if (z<z_min):
#            z_min=z
    zcount=0
    z_av=0.000000000
    for i in range(vesicle.contents.vlist.contents.n):
        z=vesicle.contents.vlist.contents.vtx[i].contents.z
        z_av+=z
        zcount+=1
    z_av/=zcount

    da=(z_av-z_min)/2
    if(da>1):
        da=1
    for i in range(vesicle.contents.tlist.contents.n):
	vcount=0
        z1=vesicle.contents.tlist.contents.tria[i].contents.vertex[0].contents.z
        z2=vesicle.contents.tlist.contents.tria[i].contents.vertex[1].contents.z
        z3=vesicle.contents.tlist.contents.tria[i].contents.vertex[2].contents.z
	if((z1-z_min)<da):
	    vcount+=1
	if((z2-z_min)<da):
	    vcount+=2
	if((z3-z_min)<da):
	    vcount+=4

	if(vcount==7):
	    area+=vesicle.contents.tlist.contents.tria[i].contents.area

	if(vcount==3):
            x1=vesicle.contents.tlist.contents.tria[i].contents.vertex[0].contents.x
            y1=vesicle.contents.tlist.contents.tria[i].contents.vertex[0].contents.y
            z1=vesicle.contents.tlist.contents.tria[i].contents.vertex[0].contents.z
            x2=vesicle.contents.tlist.contents.tria[i].contents.vertex[1].contents.x
            y2=vesicle.contents.tlist.contents.tria[i].contents.vertex[1].contents.y
            z2=vesicle.contents.tlist.contents.tria[i].contents.vertex[1].contents.z
            x3=vesicle.contents.tlist.contents.tria[i].contents.vertex[2].contents.x
            y3=vesicle.contents.tlist.contents.tria[i].contents.vertex[2].contents.y
            z3=vesicle.contents.tlist.contents.tria[i].contents.vertex[2].contents.z

	    x=((x3-x1)*(z_min+da) - (z1*x3-z3*x1))/(z3-z1)
	    y=((y3-y1)*(z_min+da) - (z1*y3-z3*y1))/(z3-z1)
	    z=z_min+da

	    x0=((x2-x3)*(z_min+da) - (z3*x2-z2*x3))/(z2-z3)
	    y0=((y2-y3)*(z_min+da) - (z3*y2-z2*y3))/(z2-z3)
	    z0=z_min+da

	    a=((x1-x)**2 + (y1-y)**2 + (z1-z)**2)**(0.5)
	    b=((x2-x)**2 + (y2-y)**2 + (z2-z)**2)**(0.5)
	    c=((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)**(0.5)
            s=(a+b+c)/2
	    area1=(s*(s-a)*(s-b)*(s-c))**(0.5)

	    a=((x2-x)**2 + (y2-y)**2 + (z2-z)**2)**(0.5)
	    b=((x2-x0)**2 + (y2-y0)**2 + (z2-z0)**2)**(0.5)
	    c=((x-x0)**2 + (y-y0)**2 + (z-z0)**2)**(0.5)
            s=(a+b+c)/2
	    area2=(s*(s-a)*(s-b)*(s-c))**(0.5)

	    area=area+area1+area2

	if(vcount==6):
            x1=vesicle.contents.tlist.contents.tria[i].contents.vertex[0].contents.x
            y1=vesicle.contents.tlist.contents.tria[i].contents.vertex[0].contents.y
            z1=vesicle.contents.tlist.contents.tria[i].contents.vertex[0].contents.z
            x2=vesicle.contents.tlist.contents.tria[i].contents.vertex[1].contents.x
            y2=vesicle.contents.tlist.contents.tria[i].contents.vertex[1].contents.y
            z2=vesicle.contents.tlist.contents.tria[i].contents.vertex[1].contents.z
            x3=vesicle.contents.tlist.contents.tria[i].contents.vertex[2].contents.x
            y3=vesicle.contents.tlist.contents.tria[i].contents.vertex[2].contents.y
            z3=vesicle.contents.tlist.contents.tria[i].contents.vertex[2].contents.z

	    x=((x2-x1)*(z_min+da) - (z1*x2-z2*x1))/(z2-z1)
	    y=((y2-y1)*(z_min+da) - (z1*y2-z2*y1))/(z2-z1)
	    z=z_min+da

	    x0=((x1-x3)*(z_min+da) - (z3*x1-z1*x3))/(z1-z3)
	    y0=((y1-y3)*(z_min+da) - (z3*y1-z1*y3))/(z1-z3)
	    z0=z_min+da

	    a=((x2-x)**2 + (y2-y)**2 + (z2-z)**2)**(0.5)
	    b=((x3-x)**2 + (y3-y)**2 + (z3-z)**2)**(0.5)
	    c=((x2-x3)**2 + (y2-y3)**2 + (z2-z3)**2)**(0.5)
            s=(a+b+c)/2
	    area1=(s*(s-a)*(s-b)*(s-c))**(0.5)

	    a=((x3-x)**2 + (y3-y)**2 + (z3-z)**2)**(0.5)
	    b=((x3-x0)**2 + (y3-y0)**2 + (z3-z0)**2)**(0.5)
	    c=((x-x0)**2 + (y-y0)**2 + (z-z0)**2)**(0.5)
            s=(a+b+c)/2
	    area2=(s*(s-a)*(s-b)*(s-c))**(0.5)

	    area=area+area1+area2

	if(vcount==5):
            x1=vesicle.contents.tlist.contents.tria[i].contents.vertex[0].contents.x
            y1=vesicle.contents.tlist.contents.tria[i].contents.vertex[0].contents.y
            z1=vesicle.contents.tlist.contents.tria[i].contents.vertex[0].contents.z
            x2=vesicle.contents.tlist.contents.tria[i].contents.vertex[1].contents.x
            y2=vesicle.contents.tlist.contents.tria[i].contents.vertex[1].contents.y
            z2=vesicle.contents.tlist.contents.tria[i].contents.vertex[1].contents.z
            x3=vesicle.contents.tlist.contents.tria[i].contents.vertex[2].contents.x
            y3=vesicle.contents.tlist.contents.tria[i].contents.vertex[2].contents.y
            z3=vesicle.contents.tlist.contents.tria[i].contents.vertex[2].contents.z

	    x=((x2-x1)*(z_min+da) - (z1*x2-z2*x1))/(z2-z1)
	    y=((y2-y1)*(z_min+da) - (z1*y2-z2*y1))/(z2-z1)
	    z=z_min+da

	    x0=((x2-x3)*(z_min+da) - (z3*x2-z2*x3))/(z2-z3)
	    y0=((y2-y3)*(z_min+da) - (z3*y2-z2*y3))/(z2-z3)
	    z0=z_min+da

	    a=((x1-x)**2 + (y1-y)**2 + (z1-z)**2)**(0.5)
	    b=((x3-x)**2 + (y3-y)**2 + (z3-z)**2)**(0.5)
	    c=((x3-x1)**2 + (y3-y1)**2 + (z3-z1)**2)**(0.5)
            s=(a+b+c)/2
	    area1=(s*(s-a)*(s-b)*(s-c))**(0.5)

	    a=((x3-x)**2 + (y3-y)**2 + (z3-z)**2)**(0.5)
	    b=((x3-x0)**2 + (y3-y0)**2 + (z3-z0)**2)**(0.5)
	    c=((x-x0)**2 + (y-y0)**2 + (z-z0)**2)**(0.5)
            s=(a+b+c)/2
	    area2=(s*(s-a)*(s-b)*(s-c))**(0.5)

	    area=area+area1+area2

	if(vcount==1):
            x1=vesicle.contents.tlist.contents.tria[i].contents.vertex[0].contents.x
            y1=vesicle.contents.tlist.contents.tria[i].contents.vertex[0].contents.y
            z1=vesicle.contents.tlist.contents.tria[i].contents.vertex[0].contents.z
            x2=vesicle.contents.tlist.contents.tria[i].contents.vertex[1].contents.x
            y2=vesicle.contents.tlist.contents.tria[i].contents.vertex[1].contents.y
            z2=vesicle.contents.tlist.contents.tria[i].contents.vertex[1].contents.z
            x3=vesicle.contents.tlist.contents.tria[i].contents.vertex[2].contents.x
            y3=vesicle.contents.tlist.contents.tria[i].contents.vertex[2].contents.y
            z3=vesicle.contents.tlist.contents.tria[i].contents.vertex[2].contents.z

	    x=((x3-x1)*(z_min+da) - (z1*x3-z3*x1))/(z3-z1)
	    y=((y3-y1)*(z_min+da) - (z1*y3-z3*y1))/(z3-z1)
	    z=z_min+da

	    x0=((x2-x1)*(z_min+da) - (z1*x2-z2*x1))/(z2-z1)
	    y0=((y2-y1)*(z_min+da) - (z1*y2-z2*y1))/(z2-z1)
	    z0=z_min+da

	    a=((x1-x)**2 + (y1-y)**2 + (z1-z)**2)**(0.5)
	    b=((x1-x0)**2 + (y1-y0)**2 + (z1-z0)**2)**(0.5)
	    c=((x-x0)**2 + (y-y0)**2 + (z-z0)**2)**(0.5)
            s=(a+b+c)/2

	    area+=(s*(s-a)*(s-b)*(s-c))**(0.5)

	if(vcount==2):
            x1=vesicle.contents.tlist.contents.tria[i].contents.vertex[0].contents.x
            y1=vesicle.contents.tlist.contents.tria[i].contents.vertex[0].contents.y
            z1=vesicle.contents.tlist.contents.tria[i].contents.vertex[0].contents.z
            x2=vesicle.contents.tlist.contents.tria[i].contents.vertex[1].contents.x
            y2=vesicle.contents.tlist.contents.tria[i].contents.vertex[1].contents.y
            z2=vesicle.contents.tlist.contents.tria[i].contents.vertex[1].contents.z
            x3=vesicle.contents.tlist.contents.tria[i].contents.vertex[2].contents.x
            y3=vesicle.contents.tlist.contents.tria[i].contents.vertex[2].contents.y
            z3=vesicle.contents.tlist.contents.tria[i].contents.vertex[2].contents.z

	    x=((x2-x1)*(z_min+da) - (z1*x2-z2*x1))/(z2-z1)
	    y=((y2-y1)*(z_min+da) - (z1*y2-z2*y1))/(z2-z1)
	    z=z_min+da

	    x0=((x2-x3)*(z_min+da) - (z3*x2-z2*x3))/(z2-z3)
	    y0=((y2-y3)*(z_min+da) - (z3*y2-z2*y3))/(z2-z3)
	    z0=z_min+da

	    a=((x2-x)**2 + (y2-y)**2 + (z2-z)**2)**(0.5)
	    b=((x2-x0)**2 + (y2-y0)**2 + (z2-z0)**2)**(0.5)
	    c=((x-x0)**2 + (y-y0)**2 + (z-z0)**2)**(0.5)
            s=(a+b+c)/2

	    area+=(s*(s-a)*(s-b)*(s-c))**(0.5)

	if(vcount==4):
            x1=vesicle.contents.tlist.contents.tria[i].contents.vertex[0].contents.x
            y1=vesicle.contents.tlist.contents.tria[i].contents.vertex[0].contents.y
            z1=vesicle.contents.tlist.contents.tria[i].contents.vertex[0].contents.z
            x2=vesicle.contents.tlist.contents.tria[i].contents.vertex[1].contents.x
            y2=vesicle.contents.tlist.contents.tria[i].contents.vertex[1].contents.y
            z2=vesicle.contents.tlist.contents.tria[i].contents.vertex[1].contents.z
            x3=vesicle.contents.tlist.contents.tria[i].contents.vertex[2].contents.x
            y3=vesicle.contents.tlist.contents.tria[i].contents.vertex[2].contents.y
            z3=vesicle.contents.tlist.contents.tria[i].contents.vertex[2].contents.z

	    x=((x3-x1)*(z_min+da) - (z1*x3-z3*x1))/(z3-z1)
	    y=((y3-y1)*(z_min+da) - (z1*y3-z3*y1))/(z3-z1)
	    z=z_min+da

	    x0=((x2-x3)*(z_min+da) - (z3*x2-z2*x3))/(z2-z3)
	    y0=((y2-y3)*(z_min+da) - (z3*y2-z2*y3))/(z2-z3)
	    z0=z_min+da

	    a=((x3-x)**2 + (y3-y)**2 + (z3-z)**2)**(0.5)
	    b=((x3-x0)**2 + (y3-y0)**2 + (z3-z0)**2)**(0.5)
	    c=((x-x0)**2 + (y-y0)**2 + (z-z0)**2)**(0.5)
            s=(a+b+c)/2

	    area+=(s*(s-a)*(s-b)*(s-c))**(0.5)

    only_area+=area
    avg+=area/total_area
    sq_avg+=(area/total_area)*(area/total_area)
    icount+=1
var=(sq_avg/icount) - (avg/icount)*(avg/icount)
error=var**0.5
f=open('E_rho_frac_area_passive_c0_1_close_c_E.dat','a')
f.write('{}, {}, {}, {} \n'.format(E,rho,avg/icount,error))

f2=open('E_rho_area_only_passice_c0_1_close_c_E.dat','a')
f2.write('{}, {}, {} \n'.format(E,rho,only_area/icount))

                        

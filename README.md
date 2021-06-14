# trisurf-python
python scripts and functions to use with trisurf. Depends on things in a standard anaconda (numpy, matplotlib, pandas, numba)

### MAIN IMPORTANT FUNCTIONS:
ts_vtu_to_python.py  
> *module*  
>mainly, contains functions to parse a .vtu file outputted by trisurf to produce a python geometry
>* vtu_get_geometry: takes .vtu, returns vertex positions list, bond list, and triangle list
>* vtu_get_tape: takes .vtu, returns the tape
>* vtu_get_vertex_data: takes .vtu, returns spontaneus curvature and bending energy (numpy arrays)
>* cluster_dist_from_vtu: takes .vtu, returns distribution of cluster size  

statistics_from_vtu.py  
>*command line function* -python version of tspoststat
> accumulate statistics of the main order parameters for .vtu files into single file
> volume,
> optionally writes cluster size histogram files
> example usage:
> ```bash
> $python trisurf_python/statistics_from_vtu.py folder_of_vtus -o all_statistics -w -v 
> ```
> will create `all_statistics.csv` file with No,Volume,Area,lamdba1,lambda2,lambda3,Nbw/Nb,hbar,mean_cluster_size,std_cluster_size,line_length  
> and write a `histogram_\*.csv` for each file
Everything else is mostly lousy/predecessor scripts, for plotting, adding spontaneus curvature to vtu, or messing about

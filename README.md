# trisurf-python
python scripts and functions to use with trisurf. Depends purely on things in a standard anaconda (numpy, matplotlib, pandas, numba)  

Command line programs:  
* get_HDF_from_vtu  
    Extract statistics of trisurf .vtu files into an hdf5 file.  
* xmloctomy  
    Remove unnecessary data from trisurf vtu files.  

Package modules:  
* vtu  
    Handle a trisurf .vtu as a python object  
* for_paraview  
    Helpful functions for ParaView to view and understand a trisurf .vtu  
* numba_numeric  
    Uses numba to process a .vtu file in similar way to the c program  
* extractors and dfget  
    extract imformation from hdf5 file generated by get_HDF_from_vtu  
* ts_auto_wrapper   
    Creates a wrapper for trisurf  
* small_functions  
    utilities  

Notebooks:  
* ts_auto_wrapper_test.ipynb  
    Testing the wrapper and examples of use  
* test_versions.ipynb  
    Testing newly compiled trisurf versions (work in progress)
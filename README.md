# cavity_genesis4

combining Genesis4 and recirculation with crystals/lens to run on s3df


This code is divided into two parts
1.  Genesis4 wrapper (`cavity_genesis4/genesis4`). It includes

        lume_genesis_JT
        run_genesis4.py
        scan_param_20keV.py
    1. `lume_genesis_JT `is modified from https://github.com/slaclab/lume-genesis version 4 to make it work seamlessly on s3df.
Especailly a function `make_lattice2.py `is added to make undulator tapering period by period. `lume_genesis_JT.Genesis4.submit_batch` is used to submit batch jobs to s3df rather than interactive modes, which is convenient for scanning paramters.

    2. `run_genesis4.py` is an interface to conveniently modifiy beam, lattice, seed files etc in Genesis4. It will generate initial files, submitting batch jobs to s3df and check if the job is finished. This function is directly called in the main cavity codes
 
    3. `scan_param_20keV.py` is an interface for single pass run of genesis. It is convenient for submitting multiple jobs at the same time to scan parameters like undulator tapering.


2. Cavity recirculation codes (`cavity_genesis4/recircualtion`). Main functions are

        Bragg_mirror.py Bragg_mirror3.py
        dfl_cbxfel_new_config.py
        merge_files_mpi.py
   1.  `Bragg_mirror.py` and `Brag_mirror3.py` are functions generating 2D (photon energy vs kx) Bragg response of crystals given *Shvyd’Ko, Yuri, and Ryan Lindberg. "Spatiotemporal response of crystals in x-ray Bragg diffraction." Physical Review Special Topics-Accelerators and Beams 15.10 (2012): 100702.*

   2. `dfl_cbxfel_new_config.py` reads the output field files from Genesis4, do padding in time and x, make Fourier transform, propagate it through elements in a cavity (drift, Bragg mirrors, lens) and back the start of the undulator. There is option to recirculate the field for multiple roundtrips. It write recirculated seed files to Genesis4 format for next Genesis run. 
The field (in frequency space) will be divided into multiple blocks to be processed on mutiple cores with MPI.  It therefore provides more memory and  allows more paddings which is required when the Bragg mirror has narrow bandwidth.

   3. `merge_files_mpi.py` It will merge the recirculated and transmitted field files written by different cores from the last step into a single file.


`cavity_genesis4/run_cavity.py` is the main control file for the whole cavity simulation. It will call genesis4, recirculation and merge file functions one by one in a loop. Electron beam, crystal, cavity configuration can be easily tuned here.


# How to Run

## For cavity simulations
1. modify ebeam, undulator and cavity parameters in `cavity_genesis4/run_cavity.py`. 
2. modify the path in `submit_cavity.sh`
3. submit by `sbatch submit_cavity.sh`

## For single pass run
1. modify the parameters in `cavity_genesis4/genesis4/scan_param_20keV.py`
2. modify the path in `submit_scan.sh`
3. submit by `sbatch submit_scan.sh`
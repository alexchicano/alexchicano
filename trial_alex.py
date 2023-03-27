"""
export PATH=/home/alex/fsl/bin${PATH:+:${PATH}}
export FSLDIR=/home/alex/fsl
. ${FSLDIR}/etc/fslconf/fsl.sh
"""

from utils_fsl_tutorial import *
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import os
import scipy.io as sio
from pathlib import Path
import time


counter = 0

cwd = os.getcwd()

# Since this script is being run in "/home/{username}/src", the fsl directory will always be:
fsldir = str(Path(cwd).parent) + '/fsl'
workdir_root = str(Path(cwd).parent) + '/data/datasets/ImaGenoma/T1_b'
mni_path =  str(Path(cwd).parent) + '/data/datasets/_MNI_template/mni_icbm152_nlin_asym_09c_nifti/mni_icbm152_nlin_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c.nii'

# Define function to read non-hidden directories in a given path:
def  list_nonhid_dirs(path):
    nonhid_dirs = [ f for f in os.listdir(path) if not f.startswith('.') ]
    return nonhid_dirs

def image_registration(mni_path, workdir_root, id, fsldir):
    global counter
    counter += 1
    
    print('Current count: ', counter)
    id_dir = workdir_root +'/'+ id
    im_path = id_dir +'/3D_T1WI_MPRAGE.nii'
    transform_path = id_dir + '/fsl_pairreg.mat'

    # Define workdir and create it if it doesn't exist:
    workdir = workdir_root + '/' + id + '/temporary_files'
    if not os.path.exists(workdir):
        os.makedirs(workdir)

    print('Just getting started with Pairreg!')

    run_fsl_pairreg(
        ref_filepath = mni_path, 
        mov_filepath = im_path, 
        transform_filepath = transform_path, 
        workdir=workdir,
        fsldir=fsldir
        )

    print('Now finishing with Transform ;)')
    apply_fsl_transform(
        filepath_in = im_path, 
        filepath_ref = mni_path, 
        filepath_out = workdir + '/transformed.nii.gz', 
        filepath_transform = transform_path, 
        fsldir=fsldir
    )

"""def segment_anat_fast_first(args):
    global counter
    counter += 1
    segment_anat_fast_first(**args)
"""
if __name__ == '__main__':

    # Obtain a time stamp for the start of the script:
    start_time = time.time()

    # Read patients list and their images:
    #id_dirs = list_nonhid_dirs(workdir_root)
    id_dirs = ['000018-01']
    

    args_dict = []
    for id in id_dirs:
        args_dict.append({
        'mni_path': mni_path,
        'workdir_root': workdir_root,
        'id': id,
        'fsldir': fsldir
        })
    
    
    parallel_run(
        func=image_registration, 
        args=args_dict, 
        num_threads=70
    )

    # Obtain an intermediate time stamp:
    print('Time elapsed: ', time.time() - start_time)

    print('Just getting started with Segment_anat_fast_first!')
    
    args_dict = []
    for id in id_dirs:
        id_dir = workdir_root + '/' + id
        
        fp_fast = [
                id_dir + '/fast_0.nii.gz',
                id_dir + '/fast_1.nii.gz',
                id_dir + '/fast_2.nii.gz',
                id_dir + '/fast_3.nii.gz',
            ]
        fp_first = id_dir + '/first.nii.gz'

        
        args_dict.append({
            'fpath_in': id_dir +'/temporary_files/transformed.nii.gz',
            'is_bias_corrected': True,
            'fpaths_fast': fp_fast,
            'fpath_first': fp_first,
            'fp_skull_stripped_out': id_dir + '/skull_stripped.nii.gz',
            'decimals': 5, # Rounding to 4 decimals saves hard drive space 
            'remove_anat_dir': False
        })
    
    parallel_run(
        func=segment_anat_fast_first, 
        args=args_dict, 
        num_threads=50
    )

    print('Done! :D')
    
    print('Time elapsed: ', time.time() - start_time)

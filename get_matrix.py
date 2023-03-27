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

cwd = os.getcwd()

# Since this script is being run in "/home/{username}/src", the fsl directory will always be:
fsldir = str(Path(cwd).parent) + '/fsl'
workdir_root = str(Path(cwd).parent) + '/data/datasets/ImaGenoma/T1_b'
mni_path =  str(Path(cwd).parent) + '/data/datasets/_MNI_template/mni_icbm152_nlin_asym_09c_nifti/mni_icbm152_nlin_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c.nii'

# Define function to read non-hidden directories in a given path:
def  list_nonhid_dirs(path):
    nonhid_dirs = [ f for f in os.listdir(path) if not f.startswith('.') ]
    return nonhid_dirs


#Define the function to find the transformation matrix and save it as .mat file using FLIRT
def find_transform_and_save_mat(ref_nii, mov_nii):
    """
    Finds the transformation matrix between two NIfTI images and saves it as 'fsl_pairreg.mat' file in the same folder as moving images.
    
    Args:
    ref_nii (str): Path to the reference template NIfTI image.
    mov_nii (str): Path to the each moving NIfTI image.
    
    Returns:
    out_mat: The tranformation matrix as .mat file
    """
    # Find the paths and file names of the input files
    ref_dir, ref_name = os.path.split(ref_nii)
    mov_dir, mov_name = os.path.split(mov_nii)

    # Define the output path and file name
    out_mat = os.path.join(mov_dir, "fsl_pairreg.mat")

    # Run the FLIRT command to find the transformation matrix
    cmd = ["flirt", "-in", mov_nii, "-ref", ref_nii, "-omat", out_mat]
    subprocess.run(cmd, check=True)

    print(f"Transformation matrix saved to {out_mat}")

    return out_mat


if __name__ == '__main__':
    
    # Read patients list and their images:
    id_dirs = list_nonhid_dirs(workdir_root)

    args_dict = []
    for id in id_dirs:
        args_dict.append({
        'ref_nii': mni_path,
        'mov_nii': workdir_root +'/'+ id +'/3D_T1WI_MPRAGE.nii',
        })
    
    parallel_run(
        func=find_transform_and_save_mat, 
        args=args_dict, 
        num_threads=70
    )

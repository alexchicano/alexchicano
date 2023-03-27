from fsl_utils import *
import os
from pathlib import Path

# Variable used to keep track of the number of patients processed:
patients_counter = 0

# Get current working directory:
cwd = os.getcwd()

# Since this script is being run in "/home/{username}/src", the fsl directory will always be:
fsldir = str(Path(cwd).parent) + '/fsl'

# Load the path to the root directory of the dataset and the path to the MNI template:
workdir_root = str(Path(cwd).parent) + '/data/datasets/ImaGenoma/T1_b'
mni_path =  str(Path(cwd).parent) + '/data/datasets/_MNI_template/mni_icbm152_nlin_asym_09c_nifti/mni_icbm152_nlin_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c.nii'


# Define function to read non-hidden directories in a given path:
def  list_nonhid_dirs(path):
    nonhid_dirs = [ f for f in os.listdir(path) if not f.startswith('.') ]
    return nonhid_dirs

# Define function to perform image registration (pairreg and transform):
def image_registration(mni_path, workdir_root, id, fsldir):
    global patients_counter
    patients_counter += 1
    
    print('Now registering patient number ', patients_counter, ' out of ', len(id_dirs), ' patients.')
    
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

# Define function to perform segmentation (fast and first):
def segment(**args):
    
    global patients_counter
    patients_counter += 1
    print('Now segmenting patient number ', patients_counter, ' out of ', len(id_dirs), ' patients.')
    segment_anat_fast_first(**args)



if __name__ == '__main__':

    number_threads = 70
    
    # Read patients list and their images:
    id_dirs = list_nonhid_dirs(workdir_root)
    
    
    print('Just getting started with image registration!')
    # Arguments for parallelization of image registration:
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
        num_threads=number_threads
    )
    
    patients_counter = 0
    print('Just getting started with Segment_anat_fast_first!')
    
    # Arguments for parallelization of Segmentation_anat_fast_first:
    args_dict = []
    for id in id_dirs:
        id_dir = workdir_root + '/' + id
        
        fp_fast = [
                id_dir + '/background.nii.gz',
                id_dir + '/csf.nii.gz',
                id_dir + '/gm.nii.gz',
                id_dir + '/wm.nii.gz',
            ]
        fp_first = id_dir + '/subc.nii.gz'

        
        args_dict.append({
            'fpath_in': id_dir + '/temporary_files/transformed.nii.gz',
            'is_bias_corrected': True,
            'fpaths_fast': fp_fast,
            'fpath_first': fp_first,
            'fpath_anat_to_mni_out': id_dir + '/T1_to_MNI_nonlin.nii.gz',
            'fp_skull_stripped_out': id_dir + '/rigid_skull_stripped.nii.gz',
            'fp_nonl_skull_stripped_out': id_dir + '/nonl_brain_mask.nii.gz',
            'decimals': 4, # Rounding to 4 decimals saves hard drive space 
            'remove_anat_dir': True
        })
    
    
    parallel_run(
        func=segment,
        args=args_dict, 
        num_threads=number_threads
    )

    print('Done! :D')

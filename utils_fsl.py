import tempfile
import shutil
import subprocess
import os
import nibabel as nib
import numpy as np

def remove_ext(filepath):
    """Removes all extensions of the filename pointed by filepath.

    :Example:

    >>> remove_ext('home/user/t1_image.nii.gz')
    'home/user/t1_image'
    """
    paths = filepath.split('/')
    return filepath if '.' not in paths[-1] else os.sep.join(paths[:-1] + [paths[-1].split('.')[0]])


def run_fsl_bet(
    t1_fp, 
    brain_out_fp=None, 
    brain_mask_out_fp=None, 
    skull_out_fp=None, 
    betopts='-B', 
    fsldir='/usr/local/fsl'
):
    """
    useful betopts (mutually exclusive options, so just one of these at a time)
      -R          robust brain centre estimation (iterates BET several times)
      -S          eye & optic nerve cleanup (can be useful in SIENA - disables -o option)
      -B          bias field & neck cleanup (can be useful in SIENA)
    
    check outputs to see which option is best
    """

    with tempfile.TemporaryDirectory() as tempdir:
        t1_fp_in = os.path.join(tempdir, 't1.nii.gz')
        brain_fp = os.path.join(tempdir, 't1_bet_brain.nii.gz')
        brain_mask_fp = os.path.join(tempdir, 't1_bet_brain_mask.nii.gz')
        skull_fp = os.path.join(tempdir, 't1_bet_brain_skull.nii.gz')

        shutil.copy(t1_fp, t1_fp_in)
        subprocess.check_output(['bash', '-c', f'{fsldir}/bin/bet {t1_fp_in} t1_bet_brain -s {betopts}'], cwd=tempdir)

        if brain_out_fp is not None:
            shutil.copy(brain_fp, brain_out_fp)

        if brain_mask_out_fp is not None:
            shutil.copy(brain_mask_fp, brain_mask_out_fp)

        if skull_out_fp is not None:
            shutil.copy(skull_fp, skull_out_fp)


def run_fsl_pairreg(
    ref_filepath, 
    mov_filepath, 
    transform_filepath, 
    workdir=None, 
    fsldir='/usr/local/fsl'
):
    """
    workdir: can be used to specify a folder where the intermediate results are stored, otherwise they are
        stored in a temporary directory which is erased at the end
    """
    tempdir = None
    if workdir is None:
        tempdir = tempfile.TemporaryDirectory()
        workdir = tempdir.name

    try:
        ref_brain_fp = os.path.join(workdir, 'ref_bet_brain.nii.gz')
        ref_skull_fp = os.path.join(workdir, 'ref_bet_brain_skull.nii.gz')
        run_fsl_bet(ref_filepath, brain_out_fp=ref_brain_fp, skull_out_fp=ref_skull_fp, betopts='-R -S')

        mov_brain_fp = os.path.join(workdir, 'mov_bet_brain.nii.gz')
        mov_skull_fp = os.path.join(workdir, 'mov_bet_brain_skull.nii.gz')
        run_fsl_bet(mov_filepath, brain_out_fp=mov_brain_fp, skull_out_fp=mov_skull_fp, betopts='-R -S')

        subprocess.check_output(
            ['bash', '-c', f'{fsldir}/bin/pairreg {ref_brain_fp} {mov_brain_fp} {ref_skull_fp} {mov_skull_fp} {transform_filepath}'])
    finally:
        if tempdir is not None:
            tempdir.cleanup()

def apply_fsl_transform(
    filepath_in, 
    filepath_ref, 
    filepath_out, 
    filepath_transform, 
    fsldir='/usr/local/fsl'
):
    tx_cmd = '{}/bin/flirt -out {} -applyxfm -init {} -ref {} -in {}'.format(
        fsldir, filepath_out, filepath_transform, filepath_ref, filepath_in)
    subprocess.check_output(['bash', '-c', tx_cmd])


def invert_fsl_transform(
    filepath_transform, 
    filepath_inverse, 
    fsldir='/usr/local/fsl'
):
    subprocess.check_output(
        ['bash', '-c', f'{fsldir}/bin/convert_xfm -inverse -omat {filepath_inverse} {filepath_transform}'])

def segment_anat_fast_first(
        fpath_in,
        is_bias_corrected,
        fpaths_fast,
        fpath_first,
        decimals=4, # Rounding to 4 decimals saves hard drive space 
        remove_anat_dir=True
):
    """Runs fsl_anat pipeline doing skull-stripping, tissue segmentation (FAST) and subcortical structures (FIRST)

    fpaths_fast is a 4 element list, with the filepaths for the background, CSF, GM and WM output probability maps
    fpath_first is a single string
    """
    assert os.path.isfile(fpath_in), fpath_in

    anat_dir = remove_ext(fpath_in) + '.anat'
    fpaths_anat_fast = [os.path.join(anat_dir, 'T1_fast_pve_{}.nii.gz'.format(i)) for i in range(3)]
    fpath_anat_first = os.path.join(anat_dir, 'T1_subcort_seg.nii.gz')

    # Build and run fsl_anat command
    bias_opt = '--nobias' if is_bias_corrected else '--weakbias'
    subprocess.check_output(
        ['bash', '-c', f'fsl_anat --clobber --nocrop --noreorient {bias_opt} -i {fpath_in}'])

    # Load FAST segmentations, add background prob channel
    fast_nifti = nib.load(fpaths_anat_fast[0])
    fast_pves = [nib.load(fp).get_data() for fp in fpaths_anat_fast]
    fast_pves.insert(0, 1.0 - np.sum(np.stack(fast_pves, axis=0), axis=0))  # Background probability

    # Save fast segmentation by itself
    for fast_pve, fpath_tgt_fast in zip(fast_pves, fpaths_fast):
        fast_pve_arr = np.round(fast_pve, decimals=decimals)
        nib.Nifti1Image(fast_pve_arr, fast_nifti.affine, fast_nifti.header).to_filename(fpath_tgt_fast)

    # Load first segmentation and store
    first_nifti = nib.load(fpath_anat_first)
    first_seg = first_nifti.get_fdata().astype(int)
    nib.Nifti1Image(first_seg, fast_nifti.affine, fast_nifti.header).to_filename(fpath_first)
    
    if remove_anat_dir:
        shutil.rmtree(anat_dir)



if __name__ == '__main__':
    print('Registering...')

    run_fsl_pairreg(
        ref_filepath='/home/albert/Desktop/probes/regv2/VH_timepoints/xnat_vicorob_S01395/20190507220159/t1.nii.gz',
        mov_filepath='/home/albert/Desktop/probes/regv2/VH_timepoints/xnat_vicorob_S01395/20200504204521/t1.nii.gz',
        transform_filepath='/home/albert/Desktop/fsl_pairreg.mat',
        workdir='/home/albert/Desktop/pairreg_test/'
    )

    apply_fsl_transform(
        filepath_in='/home/albert/Desktop/probes/regv2/VH_timepoints/xnat_vicorob_S01395/20200504204521/t1.nii.gz',
        filepath_ref='/home/albert/Desktop/probes/regv2/VH_timepoints/xnat_vicorob_S01395/20190507220159/t1.nii.gz',
        filepath_out='/home/albert/Desktop/fsl_pairreg.nii.gz',
        filepath_transform='/home/albert/Desktop/fsl_pairreg.mat'
    )

    print('fsl_anat...')

    segment_anat_fast_first(
        fpath_in='/home/albert/Desktop/probes/regv2/VH_timepoints/xnat_vicorob_S01395/20190507220159/t1.nii.gz',
        is_bias_corrected=False,
        fpaths_fast=[
            '/home/albert/Desktop/pairreg_test/fast_0.nii.gz',
            '/home/albert/Desktop/pairreg_test/fast_1.nii.gz',
            '/home/albert/Desktop/pairreg_test/fast_2.nii.gz',
            '/home/albert/Desktop/pairreg_test/fast_3.nii.gz'
        ],
        fpath_first='/home/albert/Desktop/pairreg_test/first.nii.gz',
        decimals=4, # Rounding to 4 decimals saves hard drive space 
        remove_anat_dir=True
    )

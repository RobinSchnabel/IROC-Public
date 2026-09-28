#!/usr/bin/env python3

'''
Compute distortion between two images using non-linear ANTs registration (rigid + SyN).

Registration setup:
  - Fixed image: phantom (3T MRI reference)
  - Moving image: ULF (Ultra Low Field image to analyze)

Distortion metric:
  - The distortion is computed as the magnitude of the inverse deformation field.
  - This field represents the nonlinear warping required to transform phantom (reference space)
    into ULF (moving space).
  - The metric is evaluated in the reference (phantom) space and quantifies image artifacts/warping
    as a vector magnitude: sqrt(dx² + dy² + dz²) per voxel.

Output files follow convention: {src}_2_{dst}_{type}.{ext}
Example: phantom_2_ulf_warp-field-inv.nii.gz

Example Call:
python script_distortion_computation.py
-p ~/data/ground_truth/phantom_MRI04_iso1mm_N4.nii.gz
-u 20cm/260605-124820-TSE3D_rv5--pha333_at20cm_ro_y_reg--recon_IFFT.nii.gz # input here the reconstruction of intrest
-l ~/data/ground_truth/LABEL_SEGMENTATION_ALL.nii.gz
'''

import argparse
import csv
import subprocess
import sys
import os
from pathlib import Path
import nibabel as nib
import numpy as np
from skimage import filters


def strip_nii_suffix(path: str) -> str:
    name = Path(path).name
    return name[:-7] if name.endswith('.nii.gz') else Path(path).stem


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Distortion computation using non-linear ANTs registration (Phantom vs ULF)"
    )
    parser.add_argument("--phantom", "-p", type=str, required=True, help="Reference 3T MRI phantom image")
    parser.add_argument("--ulf", "-u", type=str, required=True, help="Ultra Low Field (ULF) image to analyze")
    parser.add_argument("--labels", "-l", type=str, required=True, help="Segmentation label map, aligned to the phantom image")
    parser.add_argument("--output","-o",type=str,required=True,help="Directory where results should be saved")
    args = parser.parse_args()

    # Validate input files
    if not os.path.exists(args.phantom):
        print(f"✗ Error: Phantom image not found: {args.phantom}")
        sys.exit(1)
    if not os.path.exists(args.ulf):
        print(f"✗ Error: ULF image not found: {args.ulf}")
        sys.exit(1)
    if not os.path.exists(args.labels):
        print(f"✗ Error: Label map not found: {args.labels}")
        sys.exit(1)

    # Extract prefixes correctly (handling .nii.gz files)
    phantom_prefix = strip_nii_suffix(args.phantom)
    ulf_prefix = strip_nii_suffix(args.ulf)

    # Auto-generate output directory
    #output_dir = Path(f'results_{phantom_prefix}_vs_{ulf_prefix}')
    output_dir = Path(args.output) / Path(f'{ulf_prefix}')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define standardized output file names (constant across all runs)
    phantom_thresholded = output_dir / 'phantom_threshold-otsu.nii.gz'
    ulf_thresholded = output_dir / 'ulf_threshold-otsu.nii.gz'
    rigid_transform = output_dir / 'ulf_2_phantom_rigid-transform.mat'
    warp_field = output_dir / 'ulf_2_phantom_warp-field.nii.gz'
    warp_field_inv = output_dir / 'phantom_2_ulf_warp-field-inv.nii.gz'
    ulf_aligned_rigid = output_dir / 'ulf_2_phantom_aligned-rigid-only.nii.gz'
    phantom_aligned_rigid_syn = output_dir / 'phantom_2_ulf_aligned-rigid+syn.nii.gz'
    phantom_aligned_syn_only = output_dir / 'phantom_2_ulf_aligned-syn-only.nii.gz'
    ulf_aligned_rigid_syn = output_dir / 'ulf_2_phantom_aligned-rigid+syn.nii.gz'
    distortion_magnitude = output_dir / 'phantom_2_ulf_distortion-magnitude.nii.gz'

    print(f"\n{'='*70}")
    print(f"Distortion Computation Pipeline (Phantom vs ULF)")
    print(f"{'='*70}")
    print(f"Phantom:   {args.phantom}")
    print(f"ULF:       {args.ulf}")
    print(f"Output:    {output_dir}\n")

    # ========================================================================
    # STEP 0: COPY ORIGINAL IMAGES
    # ========================================================================
    print(f"STEP 0: Copying original images")
    phantom_original = output_dir / 'phantom.nii.gz'
    ulf_original = output_dir / 'ulf_mri.nii.gz'
    labels_original = output_dir / 'labels.nii.gz'

    import shutil
    shutil.copy2(args.phantom, str(phantom_original))
    print(f"  ✓ Phantom copied → {phantom_original.name}")
    shutil.copy2(args.ulf, str(ulf_original))
    print(f"  ✓ ULF copied → {ulf_original.name}")
    shutil.copy2(args.labels, str(labels_original))
    print(f"  ✓ Labels copied → {labels_original.name}\n")

    # ========================================================================
    # STEP 1: RESAMPLE ULF TO PHANTOM RESOLUTION
    # ========================================================================
    print(f"STEP 1: Resampling ULF to phantom resolution")
    ulf_resampled = output_dir / 'ulf_mri_resampled.nii.gz'

    phantom_img_ref = nib.load(str(phantom_original))
    phantom_spacing = phantom_img_ref.header.get_zooms()[:3]

    cmd_resample = [
        'ResampleImage',
        '3',
        str(ulf_original),
        str(ulf_resampled),
        ' '.join(map(str, phantom_spacing)),
        '0',
        '0', #(0=linear, 1=nn, 2=gaussian, 3=windowedSinc, 4=bspline)
    ]

    try:
        subprocess.run(cmd_resample, check=True, capture_output=True)
        print(f"  ✓ ULF resampled → {ulf_resampled.name}")
        print(f"    Phantom spacing: {phantom_spacing}\n")
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Error: {e.stderr.decode() if e.stderr else str(e)}")
        sys.exit(1)

    # ========================================================================
    # STEP 2: N4 BIAS FIELD CORRECTION (ULF)
    # ========================================================================
    print(f"STEP 2: N4 bias field correction on resampled ULF")
    ulf_n4 = output_dir / 'ulf_mri_n4.nii.gz'

    cmd_n4 = [
        'N4BiasFieldCorrection',
        '-d', '3',
        '-i', str(ulf_resampled),
        '-o', str(ulf_n4),
    ]

    try:
        subprocess.run(cmd_n4, check=True, capture_output=True)
        print(f"  ✓ N4 bias field correction applied → {ulf_n4.name}\n")
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Error: {e.stderr.decode() if e.stderr else str(e)}")
        sys.exit(1)

    # ========================================================================
    # STEP 3: THRESHOLDING (Otsu method)
    # ========================================================================
    print(f"STEP 3: Thresholding images")
    print(f"  Processing phantom: {phantom_prefix}")
    phantom_img = nib.load(str(phantom_original))
    phantom_data = phantom_img.get_fdata()
    phantom_threshold = filters.threshold_otsu(phantom_data)
    nib.save(nib.Nifti1Image((phantom_data > phantom_threshold).astype(np.uint8),
                             phantom_img.affine, phantom_img.header), str(phantom_thresholded))
    print(f"    → {phantom_thresholded.name} (threshold={phantom_threshold:.1f})")

    print(f"  Processing ULF: {ulf_prefix}")
    ulf_img = nib.load(str(ulf_n4))
    ulf_data = ulf_img.get_fdata()
    ulf_threshold = filters.threshold_otsu(ulf_data)
    nib.save(nib.Nifti1Image((ulf_data > ulf_threshold).astype(np.uint8),
                             ulf_img.affine, ulf_img.header), str(ulf_thresholded))
    print(f"    → {ulf_thresholded.name} (threshold={ulf_threshold:.1f})\n")

    # ========================================================================
    # STEP 4: ANTs REGISTRATION (Rigid + SyN)
    # ========================================================================
    print(f"STEP 4: ANTs registration (rigid + SyN)")

    cmd_registration = [
        'antsRegistration',
        '-d', '3',
        '-o', f'[{str(output_dir / "ants_tmp_")},{str(output_dir / "phantom_warped.nii.gz")},{str(output_dir / "phantom_inverse_warped.nii.gz")}]',
        '--initial-moving-transform', f'[{str(phantom_thresholded)},{str(ulf_thresholded)},1]',
        # Rigid stage
        '-t', 'Rigid[0.1]',
        '-m', f'MeanSquares[{str(phantom_thresholded)},{str(ulf_thresholded)}]',
        '-c', '100x50x25',
        '-s', '3x2x1vox',
        '-f', '4x2x1',
        # SyN stage
        '-t', 'SyN[0.1]',
        '-m', f'MeanSquares[{str(phantom_thresholded)},{str(ulf_thresholded)}]',
        '-c', '100x50x25',
        '-s', '3x2x1vox', # Smoothing due to the use of binary images
        '-f', '4x2x1',
        '-v', '1',
    ]

    try:
        subprocess.run(cmd_registration, check=True, capture_output=True)
        print(f"  ✓ Registration completed\n")
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Error: {e.stderr.decode() if e.stderr else str(e)}")
        sys.exit(1)

    # ========================================================================
    # STEP 5: RIGID ALIGNMENT SANITY CHECK
    # ========================================================================
    print(f"STEP 5: Rigid alignment sanity check")
    rigid_transform_tmp = output_dir / 'ants_tmp_0GenericAffine.mat'

    cmd_apply_rigid = [
        'antsApplyTransforms',
        '-d', '3',
        '-i', str(ulf_thresholded),
        '-r', str(phantom_thresholded),
        '-o', str(ulf_aligned_rigid),
        '-t', str(rigid_transform_tmp),
    ]

    try:
        subprocess.run(cmd_apply_rigid, check=True, capture_output=True)
        print(f"  ✓ Rigid transformation applied → {ulf_aligned_rigid.name}\n")
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Error: {e.stderr.decode() if e.stderr else str(e)}")
        sys.exit(1)

    # ========================================================================
    # STEP 6: RENAME ANTs OUTPUT FILES
    # ========================================================================
    print(f"STEP 6: Organizing output files")

    files_to_rename = {
        'ants_tmp_0GenericAffine.mat': rigid_transform,
        'ants_tmp_1Warp.nii.gz': warp_field,
        'ants_tmp_1InverseWarp.nii.gz': warp_field_inv,
        'phantom_warped.nii.gz': ulf_aligned_rigid_syn,              # ANTs outputImage: ULF warped to phantom space
        'phantom_inverse_warped.nii.gz': phantom_aligned_rigid_syn,  # ANTs inverseOutputImage: phantom warped to ULF space
    }

    for src, dst in files_to_rename.items():
        src_path = output_dir / src
        if not src_path.exists():
            print(f"  ⚠ {src} not found")
            continue
        src_path.rename(dst)
        print(f"  ✓ {src} → {dst.name}")
    print()

    # ========================================================================
    # STEP 7: COMPUTE DISTORTION MAGNITUDE
    # ========================================================================
    print(f"STEP 7: Computing distortion magnitude")

    warp_img = nib.load(str(warp_field_inv))
    warp_data = warp_img.get_fdata()

    # Handle 5D warp fields
    if warp_data.ndim == 5 and warp_data.shape[3] == 1:
        warp_data = np.squeeze(warp_data, axis=3)

    # Compute magnitude: sqrt(dx^2 + dy^2 + dz^2)
    magnitude = np.sqrt(np.sum(warp_data**2, axis=3))

    # Save magnitude
    magnitude_img = nib.Nifti1Image(magnitude, warp_img.affine, warp_img.header)
    nib.save(magnitude_img, str(distortion_magnitude))
    print(f"  ✓ Magnitude computed → {distortion_magnitude.name}\n")

    # ========================================================================
    # STEP 8: APPLY SyN ONLY TO PHANTOM (without rigid)
    # ========================================================================
    print(f"STEP 8: Applying SyN deformation only to phantom (no rigid)")

    cmd_apply_syn_only = [
        'antsApplyTransforms',
        '-d', '3',
        '-i', str(phantom_thresholded),
        '-r', str(ulf_thresholded),  # Reference space: ULF
        '-o', str(phantom_aligned_syn_only),
        '-t', str(warp_field_inv),  # Only SyN warp, no rigid transform
    ]

    try:
        subprocess.run(cmd_apply_syn_only, check=True, capture_output=True)
        print(f"  ✓ SyN-only deformation applied → {phantom_aligned_syn_only.name}\n")
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Error: {e.stderr.decode() if e.stderr else str(e)}")
        sys.exit(1)

    # ========================================================================
    # STEP 9: DISTORTION MAGNITUDE PER SEGMENTATION LABEL
    # ========================================================================
    print(f"STEP 9: Computing distortion magnitude per label")

    label_img = nib.load(str(labels_original))
    label_data = label_img.get_fdata().astype(np.int32)
    unique_labels = sorted(label for label in np.unique(label_data) if label != 0)

    # Save magnitude as a 4D image, one volume per label (4th dimension = label index)
    distortion_magnitude_by_label = output_dir / 'phantom_2_ulf_distortion-magnitude-by-label.nii.gz'
    magnitude_by_label = np.stack(
        [np.where(label_data == label, magnitude, 0) for label in unique_labels],
        axis=-1,
    )
    nib.save(
        nib.Nifti1Image(magnitude_by_label, warp_img.affine, warp_img.header),
        str(distortion_magnitude_by_label),
    )
    print(
        f"  ✓ Distortion magnitude by label → {distortion_magnitude_by_label.name} "
        f"({len(unique_labels)} labels, volume order: {unique_labels})"
    )

    # Per-label statistics
    distortion_stats_per_label = output_dir / 'phantom_2_ulf_distortion-magnitude-per-label.csv'
    with open(distortion_stats_per_label, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['label', 'n_voxels', 'mean', 'std', 'min', 'max'])

        # Label 0 = all labels combined (exclude background)
        values = magnitude[label_data != 0]
        writer.writerow([0, values.size, values.mean(), values.std(), values.min(), values.max()])
        print(
            f"    label 0 (all): n={values.size}, "
            f"mean={values.mean():.3f}, std={values.std():.3f}, "
            f"min={values.min():.3f}, max={values.max():.3f}"
        )

        # Individual labels
        for label in unique_labels:
            values = magnitude[label_data == label]
            writer.writerow([
                int(label),
                values.size,
                values.mean(),
                values.std(),
                values.min(),
                values.max(),
            ])
            print(
                f"    label {int(label)}: n={values.size}, "
                f"mean={values.mean():.3f}, std={values.std():.3f}, "
                f"min={values.min():.3f}, max={values.max():.3f}"
            )

    print(f"  ✓ Per-label statistics → {distortion_stats_per_label.name}\n")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print(f"{'='*70}")
    print(f"Pipeline completed successfully!")
    print(f"{'='*70}")
    print(f"\nOutput files in: {output_dir}")
    print(f"  • {phantom_original.name}")
    print(f"  • {ulf_original.name}")
    print(f"  • {labels_original.name}")
    print(f"  • {ulf_resampled.name}")
    print(f"  • {ulf_n4.name}")
    print(f"  • {phantom_thresholded.name}")
    print(f"  • {ulf_thresholded.name}")
    print(f"  • {rigid_transform.name}")
    print(f"  • {ulf_aligned_rigid.name}")
    print(f"  • {phantom_aligned_rigid_syn.name}")
    print(f"  • {phantom_aligned_syn_only.name}")
    print(f"  • {ulf_aligned_rigid_syn.name}")
    print(f"  • {warp_field.name}")
    print(f"  • {warp_field_inv.name}")
    print(f"  • {distortion_magnitude.name}")
    print(f"  • {distortion_magnitude_by_label.name}")
    print(f"  • {distortion_stats_per_label.name}\n")

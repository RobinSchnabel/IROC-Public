from pathlib import Path
import argparse
import csv
import numpy as np
import nibabel as nib

"""
Average per-position distortion magnitude maps and compute per-label
statistics.

Given a set of per-position distortion magnitude NIfTI files (one per
phantom position), this script:

  1. Averages the maps voxel-wise across positions to produce a single
     position-averaged distortion magnitude map.
  2. Splits that averaged map into one volume per segmentation label,
     saved as a 4D NIfTI (all other voxels zeroed out per volume).
  3. Computes per-label statistics (n_voxels, mean, std, min, max) and
     writes them to a CSV, plus a combined "label 0" row for all
     non-background voxels.

Mean/std are computed from the across-position averaged magnitude map.
Min/max, however, are computed from the per-position by-label NIfTI
files (--distortion_maps_by_label), so they reflect the true extremes
seen in any individual position rather than being smoothed out by
averaging.


Example Call:
python scripts/script_distortion_computation_position_avg.py \
    -u \
        data/all_distortion_metrics/position1.../phantom_2_ulf_distortion-magnitude.nii.gz \           # Calculated by using "script_distortion_computation.py"
        data/all_distortion_metrics/position2.../phantom_2_ulf_distortion-magnitude.nii.gz \           # Calculated by using "script_distortion_computation.py"
        data/all_distortion_metrics/position3.../phantom_2_ulf_distortion-magnitude.nii.gz \           # Calculated by using "script_distortion_computation.py"
    -b \
        data/all_distortion_metrics/position1.../phantom_2_ulf_distortion-magnitude-by-label.nii.gz \  # Calculated by using "script_distortion_computation.py"
        data/all_distortion_metrics/position2.../phantom_2_ulf_distortion-magnitude-by-label.nii.gz \  # Calculated by using "script_distortion_computation.py"
        data/all_distortion_metrics/position3.../phantom_2_ulf_distortion-magnitude-by-label.nii.gz \  # Calculated by using "script_distortion_computation.py"
    -l data/ground_truth/LABEL_SEGMENTATION_ALL.nii.gz \
    -o "data/all_distortion_metrics/{output_folder_name}"
"""






if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Average distortion magnitude maps across positions and compute statistics."
    )

    parser.add_argument(
        "-u",
        "--distortion_maps",
        nargs="+",
        required=True,
        help="Distortion magnitude NIfTI files to average."
    )

    parser.add_argument(
        "-b",
        "--distortion_maps_by_label",
        nargs="+",
        required=True,
        help=(
            "Per-position distortion-magnitude-by-label NIfTI files "
            "(4D, one volume per label, same label order as computed "
            "from --labels). Used to compute min/max per label across "
            "positions rather than from the averaged magnitude map. "
            "Must be given in the same order as --distortion_maps."
        )
    )

    parser.add_argument(
        "-l",
        "--labels",
        required=True,
        help="Label segmentation NIfTI file."
    )

    parser.add_argument(
        "-o",
        "--output_dir",
        required=True,
        help="Output directory."
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(args.distortion_maps_by_label) != len(args.distortion_maps):
        raise ValueError(
            "--distortion_maps_by_label must contain the same number of "
            "files (one per position) as --distortion_maps: "
            f"got {len(args.distortion_maps_by_label)} vs "
            f"{len(args.distortion_maps)}."
        )


    # ========================================================================
    # STEP 1: AVERAGE DISTORTION MAGNITUDE MAPS
    # ========================================================================
    print("STEP 1: Averaging distortion magnitude maps")

    magnitude_list = []

    for file in args.distortion_maps:
        print(f"  Loading {file}")
        img = nib.load(str(file))
        magnitude_list.append(img.get_fdata())

    # voxel-wise average over positions
    magnitude = np.mean(magnitude_list, axis=0)

    # use first image for affine/header
    warp_img = nib.load(str(args.distortion_maps[0]))

    print(
        f"  ✓ Averaged {len(magnitude_list)} distortion maps\n"
    )


    # ========================================================================
    # STEP 2: SAVE AVERAGED DISTORTION MAGNITUDE
    # ========================================================================
    distortion_magnitude = output_dir / "phantom_2_ulf_distortion-magnitude.nii.gz"

    nib.save(
        nib.Nifti1Image(
            magnitude,
            warp_img.affine,
            warp_img.header
        ),
        str(distortion_magnitude),
    )

    print(
        f"  ✓ Distortion magnitude → {distortion_magnitude.name}\n"
    )


    # ========================================================================
    # STEP 3: DISTORTION MAGNITUDE PER SEGMENTATION LABEL
    # ========================================================================
    print("STEP 3: Computing distortion magnitude per label")

    label_img = nib.load(str(args.labels))
    label_data = label_img.get_fdata().astype(np.int32)

    unique_labels = sorted(
        label for label in np.unique(label_data)
        if label != 0
    )


    # Save magnitude as a 4D image, one volume per label
    distortion_magnitude_by_label = (
        output_dir /
        'phantom_2_ulf_distortion-magnitude-per-label.nii.gz'
    )

    magnitude_by_label = np.stack(
        [
            np.where(label_data == label, magnitude, 0)
            for label in unique_labels
        ],
        axis=-1,
    )

    nib.save(
        nib.Nifti1Image(
            magnitude_by_label,
            warp_img.affine,
            warp_img.header
        ),
        str(distortion_magnitude_by_label),
    )

    print(
        f"  ✓ Distortion magnitude by label → "
        f"{distortion_magnitude_by_label.name} "
        f"({len(unique_labels)} labels, volume order: {unique_labels})"
    )


    # Load per-position by-label volumes, used to pull min/max per label
    # from the individual positions rather than the across-position average.
    print("  Loading per-position distortion-magnitude-by-label files")

    by_label_per_position = []  # list of arrays, shape (X, Y, Z, n_labels)

    for file in args.distortion_maps_by_label:
        print(f"    Loading {file}")
        by_label_img = nib.load(str(file))
        by_label_data = by_label_img.get_fdata()

        if by_label_data.shape[-1] != len(unique_labels):
            raise ValueError(
                f"{file} has {by_label_data.shape[-1]} label volumes, "
                f"expected {len(unique_labels)} to match labels found in "
                f"{args.labels}."
            )

        by_label_per_position.append(by_label_data)


    # Per-label statistics
    distortion_stats_per_label = (
        output_dir /
        'phantom_2_ulf_distortion-magnitude-per-label.csv'
    )

    with open(distortion_stats_per_label, 'w', newline='') as f:

        writer = csv.writer(f)

        writer.writerow(
            ['label', 'n_voxels', 'mean', 'std', 'min', 'max']
        )


        # Label 0 = all labels combined (exclude background)
        # mean/std still come from the averaged magnitude map; min/max
        # come from the per-position, per-label voxel values.
        all_mask = label_data != 0
        values = magnitude[all_mask]

        # Each position's by-label volume has exactly one non-background
        # label volume set per voxel, so summing over the label axis
        # recovers that position's per-voxel magnitude within labeled
        # regions.
        per_position_all_values = [
            position_data.sum(axis=-1)[all_mask]
            for position_data in by_label_per_position
        ]
        all_min = min(v.min() for v in per_position_all_values)
        all_max = max(v.max() for v in per_position_all_values)

        writer.writerow(
            [
                0,
                values.size,
                values.mean(),
                values.std(),
                all_min,
                all_max,
            ]
        )

        print(
            f"    label 0 (all): n={values.size}, "
            f"mean={values.mean():.3f}, std={values.std():.3f}, "
            f"min={all_min:.3f}, max={all_max:.3f}"
        )


        # Individual labels
        for i, label in enumerate(unique_labels):

            label_mask = label_data == label
            values = magnitude[label_mask]

            per_position_label_values = [
                position_data[..., i][label_mask]
                for position_data in by_label_per_position
            ]
            label_min = min(v.min() for v in per_position_label_values)
            label_max = max(v.max() for v in per_position_label_values)

            writer.writerow(
                [
                    int(label),
                    values.size,
                    values.mean(),
                    values.std(),
                    label_min,
                    label_max,
                ]
            )

            print(
                f"    label {int(label)}: n={values.size}, "
                f"mean={values.mean():.3f}, std={values.std():.3f}, "
                f"min={label_min:.3f}, max={label_max:.3f}"
            )


    print(
        f"  ✓ Per-label statistics → "
        f"{distortion_stats_per_label.name}\n"
    )

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec


def display_all_slices(
    image_3d: np.ndarray | torch.Tensor,
    slice_axis: int = 2,
    title: str = "All_Slices",
    export_path: str | pathlib.Path | None = None,
    cmap: str = "gray",
    vmin: float | None = None,
    vmax: float | None = None,
    n_cols: int = 15,
    show: bool = True,
) -> None:
    """Display all slices of a 3-D image as a grid of subplots with a shared colorbar.

    Slices are arranged left-to-right, top-to-bottom, with their 1-based index shown
    in the top-left corner of each tile. A single colorbar spanning three rows is
    placed on the right edge. The colour scale defaults to the full minimum and the
    99th-percentile of the magnitude to suppress intensity outliers.

    Args:
        image_3d (np.ndarray | torch.Tensor): 3-D image to display. Extra singleton
            dimensions are squeezed automatically. Complex arrays are converted to
            their magnitude before display. PyTorch tensors are detached and moved to
            CPU automatically.
        slice_axis (int): Axis along which to iterate when extracting slices (0, 1, or
            2). Defaults to 2 (axial slices along the last dimension).
        title (str): Figure super-title. Defaults to ``"All_Slices"``.
        export_path (str | pathlib.Path | None): If provided, the figure is saved to
            this path at 150 dpi before displaying. Defaults to ``None``.
        cmap (str): Matplotlib colormap name. Defaults to ``"gray"``.
        vmin (float | None): Lower bound of the colour scale. Defaults to the array
            minimum.
        vmax (float | None): Upper bound of the colour scale. Defaults to the 99th
            percentile of the magnitude.
        n_cols (int): Number of slice columns per row in the grid. Defaults to 15.
        show (bool): If ``True``, call ``plt.show()`` after rendering. If ``False``,
            close the figure to free memory instead. Defaults to ``True``.

    Returns:
        None

    Raises:
        ValueError: If the input cannot be reduced to exactly 3 dimensions, or if
            ``slice_axis`` is not 0, 1, or 2.
    """

    # ---- Ensure NumPy array ----
    if isinstance(image_3d, torch.Tensor):
        image_3d = image_3d.detach().cpu().numpy()  # detach in case it has gradients
    # ---- Ensure Non Complex ----
    if np.iscomplexobj(image_3d):
        image_3d = np.abs(image_3d)
        print("Input is complex, displaying absolute value:")
    # ---- Ensure Right Dimansion ----
    if image_3d.ndim < 3:
        raise ValueError(f"3D input needed but {image_3d.ndim}D input was given.")
    if image_3d.ndim > 3:
        image_3d = np.squeeze(image_3d)  # Try to remove singleton dimensions
        if image_3d.ndim != 3:
            raise ValueError(
                f"3D input needed but {image_3d.ndim}D input was given "
                f"and could not be reduced to 3D by squeezing."
            )

    # ---- Validate slice_axis-----
    if slice_axis not in [0, 1, 2]:
        raise ValueError(f"slice_axis must be 0, 1, or 2, but got {slice_axis}")

    n_slices = image_3d.shape[slice_axis]
    n_rows = int(np.ceil(n_slices / n_cols))

    # vmin / vmax defaults
    if vmin is None:
        vmin = float(np.min(image_3d))
    if vmax is None:
        vmax = float(
            np.percentile(abs(image_3d), 99)
        )  # remove extreme intensity outliers

    # Figure size
    fig_width = n_cols * 1.8
    fig_height = n_rows * 2.2
    fig = plt.figure(figsize=(fig_width, fig_height))

    # GridSpec: last column thinner for colorbar
    gs = GridSpec(
        n_rows,
        n_cols + 1,
        figure=fig,
        width_ratios=[1] * n_cols + [0.25],
        wspace=0.02,
        hspace=0.06,
    )

    fig.suptitle(title, fontsize=26, y=0.98)
    plt.subplots_adjust(top=0.85)

    # Create axes for slices
    axes = []
    for r in range(n_rows):
        for c in range(n_cols):
            axes.append(fig.add_subplot(gs[r, c]))

    # Plot slices
    for i, ax in enumerate(axes):
        if i < n_slices:
            # Extract slice based on slice_axis
            if slice_axis == 0:
                slice_data = image_3d[i, :, :]
            elif slice_axis == 1:
                slice_data = image_3d[:, i, :]
            else:  # slice_axis == 2
                slice_data = image_3d[:, :, i]

            ax.imshow(slice_data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
            # Slice number with semi-transparent black background
            ax.text(
                0.02,
                0.95,
                f"{i + 1}",
                color="white",
                fontsize=9,
                transform=ax.transAxes,
                ha="left",
                va="top",
                bbox=dict(facecolor="black", alpha=0.6, pad=1, edgecolor="none"),
            )
        ax.axis("off")

    # Colorbar: always occupy exactly 3 rows, centered vertically
    cbar_height = 3
    start_row = max(0, (n_rows - cbar_height) // 2)  # Center it
    end_row = start_row + cbar_height

    # Make sure we don't exceed the number of rows
    end_row = min(end_row, n_rows)
    start_row = max(0, end_row - cbar_height)

    cbar_ax = fig.add_subplot(gs[start_row:end_row, -1])
    # Move the colorbar to the right to create more space
    pos = cbar_ax.get_position()
    cbar_ax.set_position(
        (pos.x0 + 0.02, pos.y0, pos.width, pos.height)
    )  # Shift right by 0.02

    sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])

    cbar = fig.colorbar(sm, cax=cbar_ax)
    # cbar.set_label('Intensity', rotation=270, labelpad=15)
    cbar.ax.tick_params(labelsize=14)

    if export_path is not None:
        plt.savefig(export_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)  # Close the figure to free memory




# ----- Masks -----
def create_cylinder_mask(volume_shape, cylinder_center,
                         height_in_vox, radius_in_vox):

    nx, ny, nz = volume_shape
    x0, y0 , center_z= cylinder_center

    x, y, z = np.ogrid[:nx, :ny, :nz]

    # Circular cross-section
    circle = (x - x0)**2 + (y - y0)**2 <= radius_in_vox**2

    # Z extent
    z_min = center_z - height_in_vox / 2
    z_max = center_z + height_in_vox / 2

    cylinder = circle & (z >= z_min) & (z < z_max)

    return cylinder.astype(np.uint8)

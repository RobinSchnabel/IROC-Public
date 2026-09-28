
import numpy as np
import torch
from skimage import filters, restoration, segmentation
from sphericart import SolidHarmonics
from scipy import ndimage
import math





def _parse_axis(ndim: int, axis: int | tuple[int, ...] | None) -> tuple[int, ...]:
    """Normalise an axis argument to a tuple of ints."""
    if axis is None:
        return tuple(range(ndim))
    return (axis,) if isinstance(axis, int) else tuple(axis)


def ifft(
    kspace: np.ndarray | torch.Tensor,
    axis: int | tuple[int, ...] | None = None,
    normalize: bool = False,
) -> np.ndarray | torch.Tensor:
    """Centered inverse N-dimensional FFT.
    Dispatches on the input type:
    - ``np.ndarray``: centered iFFT via ``numpy.fft`` (ifftshift → ifftn → fftshift).
    - ``torch.Tensor``: same logic via ``torch.fft``.
    Args:
        kspace: K-space data as a NumPy array or PyTorch tensor.
        axis: Axes over which to apply the transform. If ``None``, all axes are
            used (or ``(-3, -2, -1)`` for KData).
        normalize: If ``True``, scales the result by ``sqrt(N)`` where ``N`` is
            the product of the transformed dimension sizes. Ignored for KData.
    Returns:
        Image-space data in the same container type as the input
    """
    if isinstance(kspace, np.ndarray):
        axis = _parse_axis(kspace.ndim, axis)
        kspace_shifted = np.fft.ifftshift(kspace, axes=axis)
        img_shifted = np.fft.ifftn(kspace_shifted, axes=axis)
        img = np.fft.fftshift(img_shifted, axes=axis)
        if normalize:
            return img * np.sqrt(np.prod([img.shape[d] for d in axis]))
        return img

    if isinstance(kspace, torch.Tensor):
        axis = _parse_axis(kspace.ndim, axis)
        kspace_shifted = torch.fft.ifftshift(kspace, dim=axis)
        img_shifted = torch.fft.ifftn(kspace_shifted, dim=axis)
        img = torch.fft.fftshift(img_shifted, dim=axis)
        if normalize:
            return img * math.sqrt(math.prod(img.shape[d] for d in axis))
        return img

    raise TypeError(
        "ifft expected kspace to be a numpy.ndarray or torch.Tensor, "
        f"got {type(kspace).__name__}"
    )


def fft(
    image: np.ndarray | torch.Tensor,
    axis: int | tuple[int, ...] | None = None,
    normalize: bool = False,
) -> np.ndarray | torch.Tensor:
    """Centered forward N-dimensional FFT.
    Dispatches on the input type:
    - ``np.ndarray``: centered FFT via ``numpy.fft`` (ifftshift → fftn → fftshift).
    - ``torch.Tensor``: same logic via ``torch.fft``.
    Args:
        image: Image-space data as a NumPy array or PyTorch tensor.
        axis: Axes over which to apply the transform. If ``None``, all axes are
            used (or ``(-3, -2, -1)`` for IData).
        normalize: If ``True``, scales the result by ``1/sqrt(N)`` where ``N``
            is the product of the transformed dimension sizes. Ignored for IData.
    Returns:
        K-space data in the same container type as the input
    """
    if isinstance(image, np.ndarray):
        axis = _parse_axis(image.ndim, axis)
        # Shift → FFT → Shift
        image_shifted = np.fft.ifftshift(image, axes=axis)
        kspace_shifted = np.fft.fftn(image_shifted, axes=axis)
        kspace = np.fft.fftshift(kspace_shifted, axes=axis)
        if normalize:
            return kspace / np.sqrt(np.prod([kspace.shape[d] for d in axis]))
        return kspace

    if isinstance(image, torch.Tensor):
        axis = _parse_axis(image.ndim, axis)
        image_shifted = torch.fft.ifftshift(image, dim=axis)
        kspace_shifted = torch.fft.fftn(image_shifted, dim=axis)
        kspace = torch.fft.fftshift(kspace_shifted, dim=axis)

        if normalize:
            return kspace / math.sqrt(math.prod(kspace.shape[d] for d in axis))
        return kspace


    raise TypeError(
        "fft expected kspace to be a numpy.ndarray, torch.Tensor "
        f"got {type(image).__name__}"
    )



# ============================================================================================================================================
# ============================ CPR and SSH ===================================================================================================
# ============================================================================================================================================

def tse_time_array(
    kspace: np.ndarray,
    frq_axis: int = 0,
    ph1_axis: int = 1,
    ph2_axis: int = 2,
    dwell_time: float = 5e-05,
    readout_padding: float = 0,
    symmetric_time: bool = True
) -> np.ndarray:
    """Generate a per-voxel readout time array for a TSE k-space acquisition.

    Creates an array where each entry holds the readout time of the corresponding
    k-space sample. Time increases uniformly along the frequency-encode (readout)
    axis with steps of ``dwell_time``, and is tiled to match the full k-space shape.

    Args:
        kspace (np.ndarray): 3D k-space array used only to determine the output shape.
        frq_axis (int): Axis index of the frequency-encode (readout) dimension.
            Defaults to 0.
        ph1_axis (int): Axis index of the first phase-encode dimension. Defaults to 1.
        ph2_axis (int): Axis index of the second phase-encode dimension. Defaults to 2.
        dwell_time (float): Sampling dwell time in seconds (time between consecutive
            readout samples). Defaults to ``5e-5`` s (50 µs).
        readout_padding (float): Additional time padding in seconds added to the
            total readout duration. Defaults to 0.
        symmetric_time (bool): If True, the time array is centered around the
            middle echo (i.e., t=0 at the center). Defaults to False.
            This is is a pure definition issue but yields favorable properties in the context of symmetric echo trains.

    Returns:
        np.ndarray: Time array of shape ``(n_readout, nPhase1, nPhase2)`` in seconds,
            where ``n_readout = kspace.shape[frq_axis]``.
    """
    n_readout = kspace.shape[frq_axis]
    # Make time steps in Shape: (n_readout,)
    t_readout = np.arange(0, n_readout) * (dwell_time + readout_padding/n_readout)


    if symmetric_time:
        # Center time at k-space middle (t = 0 at center echo)
        t_readout = t_readout - t_readout[n_readout // 2]

    # Repeat time steps for all phases to get shape: (n_readout, Nphase1, Nphase2)
    t_readout = np.tile(
        t_readout[:, None, None], (1, kspace.shape[ph1_axis], kspace.shape[ph2_axis])
    )

    return t_readout


def direct_reciprocal_coordinates(
    image_shape: np.ndarray | torch.Tensor,
    fov_size: np.ndarray | torch.Tensor,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[np.ndarray, np.ndarray] | tuple[torch.Tensor, None]:
    """Compute cartesian coordinates for all voxels in FOV.

    Args:
        image_shape(np.ndarray): shape of image array.
        fov_size(np.ndarray): size of field-of-view in each direction in SI units.

    Returns:
        (tuple): voxel coordinates in image/physical space and k-space
    """
    if isinstance(image_shape, np.ndarray) and isinstance(fov_size, np.ndarray):
        ndim = image_shape.size

        # direct space coordinates
        xyz_ = []
        voxel_size = fov_size / image_shape
        for i, fovi in enumerate(fov_size):
            xyz_.append(
                np.linspace(
                    -fovi / 2 + voxel_size[i] / 2,
                    fovi / 2 - voxel_size[i] / 2,
                    int(image_shape[i]),
                )
            )

        # direct space coordinates
        direct_coords = np.stack(np.meshgrid(*xyz_, indexing="ij"), axis=-1)

        # k-space vectors
        k_xyz_step = 2 * np.pi / (fov_size)
        k_xyz_ = [
            np.arange(-image_shape[i] // 2, image_shape[i] // 2) * k_xyz_step[i]
            for i in range(ndim)
        ]
        k_vectors = np.stack(np.meshgrid(*k_xyz_, indexing="ij"), axis=-1)

        limits = 2 * np.pi / voxel_size
        delta_k = 2 * np.pi / fov_size
        k_xyz_ = []
        for i, limitsi in enumerate(limits):
            k_xyz_.append(
                np.linspace(
                    -limitsi / 2 + delta_k[i] / 2,
                    limitsi / 2 - delta_k[i] / 2,
                    int(image_shape[i]),
                )
            )

        k_vectors = np.stack(np.meshgrid(*k_xyz_, indexing="ij"), axis=-1)

        return direct_coords, k_vectors

    if isinstance(image_shape, torch.Tensor) and isinstance(fov_size, torch.Tensor):
        image_shape = image_shape.to(device=device)
        fov_size = fov_size.to(device=device)

        ndim = image_shape.numel()

        # voxel size
        voxel_size = fov_size / image_shape

        # -------------------------
        # Direct (image) space grid
        # -------------------------
        xyz = []
        for i in range(ndim):
            fov_i = fov_size[i]
            n_i = int(image_shape[i].item())

            coords_i = torch.linspace(
                -fov_i / 2 + voxel_size[i] / 2,
                fov_i / 2 - voxel_size[i] / 2,
                steps=n_i,
                device=device,
                dtype=dtype,
            )
            xyz.append(coords_i)

        mesh = torch.meshgrid(*xyz, indexing="ij")
        direct_coords = torch.stack(mesh, dim=-1)

        return direct_coords, None

    raise TypeError(
        "direct_reciprocal_coordinates expected all input to be either numpy.ndarray OR torch.Tensor"
        f"got {type(image_shape).__name__} and {type(fov_size).__name__}"
    )


def signal_region_extraction(
    data: np.ndarray,
    slice_axis: int | None = None,
    lower: float = 75,
    upper: float = 95,
) -> np.ndarray:
    """Extract the region where signal is strong enough to reduce influence of noise.

    Args:
        data (np.ndarray): 2D or 3D signal strength array.
        slice_axis (int, optional): Axis along which to slice for 3D data.
                                    None for 2D data.
        lower (float): Percentile for low threshold (0-100).
        upper (float): Percentile for high threshold (0-100).

    Returns:
        np.ndarray: Boolean mask array of same shape as input data.
    """
    assert data.ndim in (2, 3), "Only 2D or 3D data is supported."
    assert 0 < lower <= 100 and 0 < upper <= 100

    # Compute thresholds based on percentiles
    lower_val = np.percentile(data, lower)
    upper_val = np.percentile(data, upper)

    def _extract_2d(slice_2d):
        edges = filters.sobel(slice_2d)
        markers = np.zeros_like(slice_2d, dtype=int)
        markers[slice_2d < lower_val] = 1
        markers[slice_2d > upper_val] = 2
        segmented = segmentation.watershed(edges, markers)
        return segmented == 1

    if data.ndim == 2:
        return _extract_2d(data)

    # For 3D data
    assert slice_axis is not None, "slice_axis must be provided for 3D data"
    mask_list = []
    for i in range(data.shape[slice_axis]):
        if slice_axis == 0:
            slice_2d = data[i, ...]
        elif slice_axis == 1:
            slice_2d = data[:, i, :]
        else:
            slice_2d = data[:, :, i]
        mask_list.append(_extract_2d(slice_2d))

    return np.stack(mask_list, axis=slice_axis)

def direct_db0_estimation(
    image0: np.ndarray,
    image1: np.ndarray,
    t_shift: float,
    thresholding: float | None = None,
) -> np.ma.MaskedArray:
    """Estimate directly the B0 inhomogeneity by computing phase difference of 2 time-shifted scan.

    If thresholding is None then watershed method is used. If thresholding is provided
    see implementation in args.

    Args:
        image0(np.ndarray): image in complex data without shift
        image1(np.ndarray): time-shifted image in complex data
        t_shift(float): shift time
        thresholding(float | None): denominator of max value of absolute image to mask no signal region

    Return:
        (np.ndarray): B0 inhomogeneity map in frequency (Hz)
    """

    image_intensity = np.abs(image0)
    if thresholding is None:
        if image0.ndim == 2:
            mask = signal_region_extraction(image_intensity, slice_axis=None)
        elif image0.ndim == 3:
            mask = signal_region_extraction(image_intensity, slice_axis=2)
        else:
            raise ValueError("dimension cannot be over 3")
    else:
        # masked_less masks values BELOW the threshold (low-signal regions).
        # .mask is True where intensity < max/thresholding, i.e. where there is no signal.
        # These masked-out voxels are excluded from the phase difference computation below.
        mask = np.ma.masked_less(image_intensity, np.max(image_intensity) / thresholding).mask

    phase_diff_all = restoration.unwrap_phase(np.angle(image1)) - restoration.unwrap_phase(
        np.angle(image0)
    )

    phase_diff = -1 * np.ma.masked_array(phase_diff_all, mask=mask)

    frequency_diff = phase_diff / (2 * np.pi * t_shift)

    return frequency_diff

def db0_SSH_expansion(
    d_b0: np.ma.MaskedArray,
    coords: tuple[np.ndarray, ...],
    obj_coords: tuple[np.ndarray, ...] | None = None,
    max_order: int = 2,
) -> np.ndarray:
    """Fit a B0 inhomogeneity map using solid spherical harmonics (SSH) expansion.

    Fits the measured B0 map ``d_b0`` to a basis of solid spherical harmonics up to
    ``max_order`` using least-squares, then evaluates the fitted field on a target
    coordinate grid. Equivalent to ``b0_spherical_harmonics_expansion`` from the
    Multiwave toolkit but implemented via the ``sphericart`` library.

    Args:
        d_b0 (np.ma.MaskedArray): Measured B0 inhomogeneity map in Hz. Masked entries
            are excluded from the least-squares fit. Shape must match ``coords``.
        coords (tuple): Coordinate arrays of the measurement grid. Either
            ``(x, y)`` for 2-D or ``(x, y, z)`` for 3-D data, where each element is
            an ``np.ndarray`` of voxel positions in SI units.
        obj_coords (tuple | None): Coordinate arrays of the target (object) grid on
            which to evaluate the fitted field. Defaults to ``coords`` (same grid).
        max_order (int): Maximum order of solid spherical harmonics to include in the
            expansion. Defaults to 2.

    Returns:
        np.ndarray: Reconstructed B0 field evaluated on ``obj_coords``, with the same
            spatial shape as the arrays in ``obj_coords``.
    """
    if obj_coords is None:
        obj_coords = coords

    # ---- Unpack coordinates ----
    if len(coords) == 2:
        x, y = coords
        x_obj, y_obj = obj_coords
        z = np.zeros_like(x)
        z_obj = np.zeros_like(x_obj)
    else:
        x, y, z = coords
        x_obj, y_obj, z_obj = obj_coords

    # ---- Prepare coordinate arrays for sphericart ----
    pts = np.stack([x, y, z], axis=-1).reshape(-1, 3)  # (N,3)
    pts_obj = np.stack([x_obj, y_obj, z_obj], axis=-1).reshape(-1, 3)

    # ---- Build solid spherical harmonics with sphericart ----
    sh_gen = SolidHarmonics(max_order)

    SH = sh_gen.compute(pts)  # (N, ncoef)
    SH_obj = sh_gen.compute(pts_obj)

    mask = d_b0.mask.flatten()
    y_data = d_b0.compressed()
    SH_masked = SH[~mask] # type: ignore

    # ---- Solve least squares for SH coefficients ----
    coef = np.linalg.lstsq(SH_masked, y_data, rcond=None)[0]

    # ---- Evaluate reconstructed field in object space ----
    b0_fit = SH_obj @ coef
    return b0_fit.reshape(x_obj.shape)


def time_segmented_cpr(
    k_space: np.ndarray | torch.Tensor,
    t: np.ndarray | torch.Tensor,
    d_b0: np.ndarray | torch.Tensor,
    t_shift: float,
    readout_axis: int,
) -> np.ndarray | torch.Tensor:
    """Time-segmented Conjugate Phase Reconstruction (CPR) for B0-corrected imaging.

    Performs B0 inhomogeneity correction by iterating over readout lines, applying a
    voxel-wise phase correction based on the local B0 offset and readout time, and
    accumulating the results via IFFT. This time-segmented approach avoids storing the
    full phase correction matrix in memory.

    Dispatches on the input type:

    - ``np.ndarray``: NumPy implementation.
    - ``torch.Tensor``: PyTorch implementation.

    Args:
        k_space: Complex k-space data with ``ndim=2`` for 2-D or ``ndim=3`` for 3-D.
        t: Phase-accumulation time array (seconds) with the same shape as ``k_space``.
            Typically generated by :func:`tse_time_array`.
        d_b0: B0 inhomogeneity map in Hz with the same shape as ``k_space``.
        t_shift: Global time shift (seconds) added to ``t`` when computing the
            phase correction term.
        readout_axis: Axis index of the readout (frequency-encode) dimension.

    Returns:
        B0-corrected complex image with the same shape and type as ``k_space``.

    Raises:
        TypeError: If ``k_space`` is neither an ``np.ndarray`` nor a ``torch.Tensor``.
    """
    if isinstance(k_space, np.ndarray):
        assert k_space.shape == t.shape == d_b0.shape

        res = np.zeros_like(k_space)
        n_readout = k_space.shape[readout_axis]

        # Build per-axis index grids to identify individual readout lines.
        indices = np.meshgrid(*[np.arange(s) for s in k_space.shape], indexing="ij")

        # t is constant along the phase-encode axes, so reading at phase position [0, 0]
        # (with readout index i) gives the correct scalar time for each line.
        t_index = [0] * k_space.ndim
        for i in range(n_readout):
            # Retain only readout line i; set all other entries to zero.
            mask = indices[readout_axis] == i
            k_readout = np.where(mask, k_space, 0)

            t_index[readout_axis] = i
            t_readout = t[tuple(t_index)]  # scalar: readout time for line i

            # Phase correction: e^(j2π · Δf · t)
            e_b = np.exp(2j * np.pi * d_b0 * (t_readout + t_shift))
            res += ifft(k_readout, normalize=True) * e_b

        return res

    if isinstance(k_space, torch.Tensor):
        assert k_space.shape == t.shape == d_b0.shape

        res = torch.zeros_like(k_space)
        n_readout = k_space.shape[readout_axis]

        # Build per-axis index grids to identify individual readout lines.
        ranges = [torch.arange(s, device=k_space.device) for s in k_space.shape]
        indices = torch.meshgrid(*ranges, indexing="ij")

        # t is constant along the phase-encode axes, so reading at phase position [0, 0]
        # (with readout index i) gives the correct scalar time for each line.
        t_index = [0] * k_space.ndim
        for i in range(n_readout):
            # Retain only readout line i; set all other entries to zero.
            mask = indices[readout_axis] == i
            k_readout = k_space * mask

            t_index[readout_axis] = i
            t_readout = t[tuple(t_index)]  # scalar: readout time for line i

            # Phase correction: e^(j2π · Δf · t)
            e_b = torch.exp(2j * torch.pi * d_b0 * (t_readout + t_shift))  # type: ignore
            res += ifft(k_readout, normalize=True) * e_b

        return res

    raise TypeError(f"k_space must be np.ndarray or torch.Tensor, got {type(k_space)}")

def iterative_cpr(
    kspace_regul,
    kspace_shift,
    fov,
    t_shift,
    dwell_time,
    n_iter=5,
    max_order=2,
    sigma=1.0,
    alpha=0.2,
    readout_axis=0,
):
    """
    Iterative Conjugate Phase Reconstruction (CPR) with B0 field estimation.

    Performs iterative B0 inhomogeneity correction using two k-space acquisitions
    with different readout timing. At each iteration, both datasets are reconstructed
    using the current B0 estimate, a B0 difference map is estimated from the resulting
    images, and the estimate is smoothed and projected onto a spherical harmonic (SSH)
    basis. A relaxed update is then applied to improve stability.

    The iteration history of the B0 estimates and reconstructed images is stored and
    used for a two-step convergence criterion.

    Args:
        kspace_regul: Complex k-space data from the regular readout acquisition.
        kspace_shift: Complex k-space data from the time-shifted acquisition.
        fov: Field of view used to generate the SSH spatial coordinates.
        t_shift: Global time shift (seconds) between the two acquisitions.
        dwell_time: Sampling dwell time used to generate the readout time arrays.
        n_iter: Maximum number of iterations.
        max_order: Maximum spherical harmonic (SSH) order used for B0 expansion.
        sigma: Standard deviation of the Gaussian smoothing applied to the estimated
            B0 field.
        alpha: Relaxation parameter for updating the B0 estimate.
        readout_axis: Axis index of the readout dimension.

    Returns:
        u1: Final CPR reconstruction of the regular acquisition.
        u2: Final CPR reconstruction of the time-shifted acquisition.
        b0: Final estimated B0 inhomogeneity map in Hz.
        history: Dictionary containing the B0 estimates and reconstructed images
            from each iteration.
    """

    # --------------------------------------------------
    # initial B0
    # --------------------------------------------------
    b0 = np.zeros(kspace_regul.shape[-3:])

    history = {"b0": [], "u1": [], "u2": []}

    # --------------------------------------------------
    # SSH coordinates
    # --------------------------------------------------
    coord, _ = direct_reciprocal_coordinates(
        np.array(b0.shape),
        fov,
    )
    coord_tuple = (coord[..., 0], coord[..., 1], coord[..., 2])

    # --------------------------------------------------
    # time encoding (constant across iterations)
    # --------------------------------------------------
    time_array_reg = tse_time_array(
        kspace_regul,
        frq_axis=0,
        ph1_axis=1,
        ph2_axis=2,
        dwell_time=dwell_time,
    )

    time_array_shift = tse_time_array(
        kspace_shift,
        frq_axis=0,
        ph1_axis=1,
        ph2_axis=2,
        dwell_time=dwell_time,
    )

    # --------------------------------------------------
    # iteration loop
    # --------------------------------------------------
    for it in range(n_iter):

        print(f"Iteration {it+1}/{n_iter}")

        # ==================================================
        # 1. CPR reconstruction
        # ==================================================

        u1 = time_segmented_cpr(
            kspace_regul,
            time_array_reg,
            b0,
            t_shift=0.0,
            readout_axis=readout_axis,
        )

        u2 = time_segmented_cpr(
            kspace_shift,
            time_array_shift,
            b0,
            t_shift=t_shift,
            readout_axis=readout_axis,
        )

        # ==================================================
        # 2. B0 estimation (DO NOT re-mask manually)
        # ==================================================

        db0 = direct_db0_estimation(
            u1,  # type: ignore
            u2,  # type: ignore
            t_shift,
        )

        db0_mask = np.ma.getmaskarray(db0)

        # Remove global bias if needed
        db0 = db0 - np.mean(db0)


        # ==================================================
        # 3. Smoothness prior (safe)
        # ==================================================

        db0 = ndimage.gaussian_filter(
            db0,
            sigma=sigma,
            mode="nearest",
        )

        db0 = np.ma.masked_array(db0, mask=db0_mask)  # re-apply mask if needed


        # ==================================================
        # 4. SSH projection
        # ==================================================

        b0_new = db0_SSH_expansion(
            db0,
            coord_tuple,  # type: ignore
            max_order=max_order,
        )

        # ==================================================
        # 5. Relaxed update (CRITICAL for stability)
        # ==================================================

        b0 = (1 - alpha) * b0 + alpha * b0_new

        # ==================================================
        # store
        # ==================================================

        history["b0"].append(b0.copy())
        history["u1"].append(u1.copy()) # type: ignore
        history["u2"].append(u2.copy()) # type: ignore

        # ==================================================
        # STOPPING CRITERION (B0-based)
        # ==================================================

        # ==================================================
        # TWO-STEP STOPPING CRITERION (B0 stability)
        # ==================================================

        if len(history["b0"]) >= 3:

            b0_k   = history["b0"][-1]
            b0_km1 = history["b0"][-2]
            b0_km2 = history["b0"][-3]

            d1 = np.linalg.norm(b0_k - b0_km1) / (np.linalg.norm(b0_km1) + 1e-12)
            d2 = np.linalg.norm(b0_km1 - b0_km2) / (np.linalg.norm(b0_km2) + 1e-12)

            print(f"B0 changes: d(k)= {d1:.3e}, d(k-1)= {d2:.3e}")

            eps = 1e-3

            if (d1 < eps) and (d2 < eps):
                print("Converged (2-step stable B0) → stopping")
                break

    return u1, u2, b0, history



# ============================================================================================================================================
# ============================ IROC ==========================================================================================================
# ============================================================================================================================================

def crop_center_3d(x: torch.Tensor|np.ndarray, output_shape: tuple[int,...]) -> torch.Tensor:
    """
    Center crop a 3D tensor or 3D np.ndarray to the specified output shape. Assumes input shape is larger than output shape in all dimensions.
    """
    n0, n1, n2 = x.shape
    k0, k1, k2 = output_shape

    s0 = (n0 - k0) // 2
    s1 = (n1 - k1) // 2
    s2 = (n2 - k2) // 2

    x_low_res = x[s0:s0+k0, s1:s1+k1, s2:s2+k2]

    return x_low_res # type: ignore

import time


def iroc_optimize_b0_ssh(
    k_space_reg: torch.Tensor,
    k_space_inv: torch.Tensor,
    time_array: torch.Tensor,
    fov: torch.Tensor,
    readout_axis: int = 0,
    max_order: int = 5,
    n_iter: int = 50,
    lr: float = 1.0,
    init_coef: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Estimate B0 map by minimizing ||CPR(k1, d_b0) - CPR(k2, -d_b0)||^2
    over SSH coefficients.

    The B0 field is parameterized as d_b0 = SH_obj @ coef, where SH_obj is the
    solid spherical harmonics basis evaluated on obj_coords. Only the SSH
    coefficients are optimized (low-dimensional), with the gradient flowing
    through both CPR calls via PyTorch autograd.

    Args:
        k_space_reg: First k-space acquisition. Shape (Nx, Ny) or (Nx, Ny, Nz).
        k_space_inv: Second k-space acquisition. Same shape as k_space1 but acquired with reversed readouts.
        t: Readout time array in seconds. Same shape as k_space1.
        t_shift: Global time shift in seconds passed to time_segmented_cpr.
        readout_axis: Readout (frequency-encode) axis index.
        fov: Field of view (x, y, z) in meters.
        max_order: Maximum SSH order. Number of coefficients = (max_order+1)^2.
        n_iter: Maximum number of LBFGS iterations.
        lr: LBFGS learning rate (1.0 is usually fine).

    Returns:
        d_b0: Estimated B0 map in Hz, shape matching obj_coords arrays.
        coef: Optimized SSH coefficients, shape ((max_order+1)^2,).
    """

    device = k_space_reg.device

    # ---- Precompute SSH basis ----
    coord, _ = direct_reciprocal_coordinates(
        torch.tensor(k_space_reg.shape),
        fov
    )

    coord_tuple = (coord[..., 0], coord[..., 1], coord[..., 2])
    x, y, z = coord_tuple

    pts = np.stack([x, y, z], axis=-1).reshape(-1, 3)

    scale = max(float(np.abs(pts).max()), 1e-9)
    pts_normalized = pts / scale

    sh_gen = SolidHarmonics(max_order)
    SH_obj = sh_gen.compute(pts_normalized)

    SH_obj_t = torch.tensor(
        SH_obj,
        dtype=torch.float64,
        device=device
    )

    spatial_shape = coord_tuple[0].shape
    n_coef = SH_obj_t.shape[1]

    # ---- Optimizable coefficients ----
    if init_coef is None:
        coef = torch.zeros(
            n_coef,
            dtype=torch.float64,
            device=device,
            requires_grad=True
        )
    else:
        coef = init_coef.clone().detach().to(device).requires_grad_(True)

    coef.grad = None

    optimizer = torch.optim.LBFGS(
        [coef],
        lr=lr,
        max_iter=20,
        line_search_fn="strong_wolfe",
    )

    # ---- Tracking ----
    losses = []
    iteration_times = []
    cumulative_times = []

    stage_start_time = time.perf_counter()

    def closure() -> torch.Tensor:
        optimizer.zero_grad()

        d_b0 = (SH_obj_t @ coef).reshape(spatial_shape)

        img1 = time_segmented_cpr(
            k_space_reg,
            time_array,
            d_b0,
            t_shift=0,
            readout_axis=readout_axis
        )

        img2 = time_segmented_cpr(
            k_space_inv,
            time_array,
            -d_b0,
            t_shift=0,
            readout_axis=readout_axis
        )

        residual = torch.abs(img1) - torch.abs(img2) # type: ignore

        loss = torch.mean(residual**2)

        loss.backward()

        return loss

    print("Starting optimization...")

    for i in range(n_iter):

        iter_start = time.perf_counter()

        loss = optimizer.step(closure)

        iter_time = time.perf_counter() - iter_start
        cumulative_time = time.perf_counter() - stage_start_time

        loss_value = loss.item()

        losses.append(loss_value)
        iteration_times.append(iter_time)
        cumulative_times.append(cumulative_time)

        print(
            f"iter {i+1:3d} | "
            f"loss {loss_value:.6e} | "
            f"time {iter_time:.2f} s | "
            f"total {cumulative_time:.2f} s"
        )

        # Early stopping
        if (
            len(losses) > 1
            and abs(losses[-2] - losses[-1])
            / (losses[-2] + 1e-30)
            < 1e-6
        ):
            print(f"Converged at iter {i+1}")
            break

    stage_time = time.perf_counter() - stage_start_time

    with torch.no_grad():
        d_b0_final = (SH_obj_t @ coef).reshape(spatial_shape)

    history = {
        "loss": losses,
        "iteration_time": iteration_times,
        "cumulative_time": cumulative_times,
        "stage_time": stage_time,
        "n_iterations": len(losses),
    }

    return d_b0_final, coef, history






def iroc_multires(
    k_space_reg: torch.Tensor,
    k_space_inv: torch.Tensor,
    time_array: torch.Tensor,
    fov: torch.Tensor,
    readout_axis: int = 0,
    res_steps: list[tuple[int, ...]] = [],
    include_full_res: bool = True,
    max_order: int = 5,
    n_iter: list[int] = [50],
    lr: float = 1.0,
) -> tuple[torch.Tensor | None, torch.Tensor | None, dict]:

    """
    Multi-resolution IROC B0 field estimation using spherical harmonic optimization.

    Performs iterative B0 estimation at multiple spatial resolutions. At each
    resolution stage, the k-space data and time array are center-cropped, and the
    B0 field is optimized using :func:`iroc_optimize_b0_ssh`. The optimized
    spherical harmonic coefficients are passed from one resolution stage to the
    next, providing a coarse-to-fine optimization strategy.

    Timing and optimization information is recorded for each resolution stage and
    for the complete multi-resolution procedure.

    Args:
        k_space_reg: Complex k-space data from the regular readout acquisition.
        k_space_inv: Complex k-space data from the inverted readout acquisition.
        time_array: Phase-accumulation time array (seconds).
        fov: Field of view used for the spatial coordinate system.
        readout_axis: Axis index of the readout dimension.
        res_steps: List of spatial resolutions used for the multi-resolution
            optimization. Each resolution is specified as a tuple of dimensions.
        include_full_res: If ``True``, ensures that the full input resolution is
            included as the final resolution stage.
        max_order: Maximum spherical harmonic (SSH) order used for B0 expansion.
        n_iter: Number of optimization iterations for each resolution stage.
            If fewer values are provided than resolution stages, the last value
            is used for the remaining stages.
        lr: Learning rate used for the B0 optimization.

    Returns:
        d_b0_final: Final estimated B0 inhomogeneity map in Hz.
        coef: Final optimized spherical harmonic coefficients.
        history: Dictionary containing optimization and timing information.

            ``history["stages"]``:
                List containing the history from each resolution stage. Each stage
                contains the optimization history with the additional fields:

                - ``stage``: Resolution stage number.
                - ``shape``: Spatial resolution used for the stage.
                - ``requested_iterations``: Number of iterations requested.
                - ``actual_stage_time``: Total runtime of the stage in seconds.

            ``history["total_time"]``:
                Total runtime of the complete multi-resolution optimization in
                seconds.

            ``history["total_iterations"]``:
                Total number of optimization iterations performed across all
                resolution stages.

            Each ``history["stages"][i]`` also contains the optimization history
            returned by :func:`iroc_optimize_b0_ssh`:

                - ``loss``: Optimization loss at each iteration.
                - ``iteration_time``: Runtime of each individual optimization
                iteration in seconds.
                - ``cumulative_time``: Cumulative optimization time in seconds.
                - ``stage_time``: Timing information recorded by the optimizer.
                - ``n_iterations``: Actual number of iterations performed.
    """

    coef = None
    d_b0_final = None

    full_res = tuple(k_space_reg.shape)

    print("Full Resolution:", full_res)

    # ---- ensure final step is full resolution ----
    if include_full_res and (
        len(res_steps) == 0 or res_steps[-1] != full_res
    ):
        res_steps.append(full_res)

    # ---- Global tracking ----
    history = {
        "stages": [],
        "total_time": None,
        "total_iterations": 0,
    }

    total_start_time = time.perf_counter()

    for i, shape in enumerate(res_steps):

        print(
            f"\n=== Resolution stage "
            f"{i+1}/{len(res_steps)}: {shape} ==="
        )

        stage_start_time = time.perf_counter()

        # ---- Crop everything consistently ----
        k_reg_low = crop_center_3d(k_space_reg, shape)
        k_inv_low = crop_center_3d(k_space_inv, shape)
        time_array_low = crop_center_3d(time_array, shape)

        n_iter_i = (
            n_iter[i]
            if i < len(n_iter)
            else n_iter[-1]
        )

        # ---- Optimize ----
        d_b0_final, coef, stage_history = iroc_optimize_b0_ssh(
            k_space_reg=k_reg_low,
            k_space_inv=k_inv_low,
            time_array=time_array_low,
            readout_axis=readout_axis,
            fov=fov,
            max_order=max_order,
            n_iter=n_iter_i,
            lr=lr,
            init_coef=coef,
        )

        stage_time = time.perf_counter() - stage_start_time

        # Add resolution-specific information
        stage_history["stage"] = i + 1
        stage_history["shape"] = shape
        stage_history["requested_iterations"] = n_iter_i
        stage_history["actual_stage_time"] = stage_time

        history["stages"].append(stage_history)

        history["total_iterations"] += stage_history["n_iterations"]

        print(
            f"Stage {i+1} finished: "
            f"{stage_history['n_iterations']} iterations | "
            f"{stage_time:.2f} s"
        )

    history["total_time"] = (
        time.perf_counter() - total_start_time
    )

    print(
        f"\n=== IROC multires finished ===\n"
        f"Total iterations: {history['total_iterations']}\n"
        f"Total time: {history['total_time']:.2f} s "
        f"({history['total_time']/60:.2f} min)"
    )

    return d_b0_final, coef, history

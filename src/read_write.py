import h5py
from pathlib import Path
import nibabel as nib
from nibabel.nifti1 import Nifti1Extension
import numpy as np
import xml.etree.ElementTree as ET
import re
import pandas as pd

def load_h5(file_path: str | Path) -> tuple:
    """
    Load all datasets from an HDF5 file and return them as a tuple.

    Iterates over the top-level keys and nested keys of the file in order and loads each dataset
    into memory. Non-dataset objects (e.g. groups) are skipped with a warning.

    Args:
        file_path (str | Path): Path to the ``.h5`` or ``.hdf5`` or ``.mrd`` file to load.

    Returns:
        tuple: One element per dataset key, in the order they appear.
            Each element is the dataset's data loaded as a NumPy array (or bytes for
            string datasets). Example::

                header, kspace = load_h5("file_path.h5")

    """
    values = []
    names = []
    with h5py.File(file_path, "r", track_order=True) as hf:

        def recurse(group, prefix=""):
            for key in group:
                obj = group[key]
                full_name = f"{prefix}/{key}" if prefix else key
                if isinstance(obj, h5py.Dataset):
                    values.append(obj[()])
                    names.append(full_name)
                elif isinstance(obj, h5py.Group):
                    recurse(obj, full_name)

        recurse(hf)
    return tuple(values)

def load_nii(file_path: str | Path) -> tuple[ bytes | None, np.ndarray]:
    """Load a NIfTI file and extract the image data and XML header (if xml is present as extension nr 6).

    Args:
        filename (str | Path): Path to the NIfTI file.

    Returns:
        tuple[np.ndarray, Nifti1Extension]: A tuple containing the XML header (or None if not present) and the image data.
    """
    img = nib.load(file_path)  # type: ignore
    img_data = img.get_fdata(dtype=np.float64)  # type: ignore
    xml_hdr = None

    for ext in img.header.extensions:  # type: ignore
        if ext.get_code() == 6:
            xml_hdr = ext.get_content()
            break

    return xml_hdr, img_data


def GetParam_from_xml(param: str, xml_header: str | bytes, print_all: bool = True) -> np.ndarray:
    """Search an ISMRMRD header XML for a named parameter (case-insensitive).

    Applies a three-level priority search and returns all matching values as a NumPy
    array. The search path taken is printed to stdout.

    Priority rules:
        1. If any match exists inside ``<sequenceParameters>``, return only those matches.
        2. Else if any match exists in ``<userParameters>`` by the ``<name>`` child element,
           return only the corresponding ``<value>`` entries.
        3. Else return generic XML-path matches from the whole header.
        4. A 2-letter ``param`` (e.g. ``"tr"``, ``"te"``, ``"ti"``) triggers a full-word
           match against the leaf tag name instead of a substring search.

    Special rule for FOV:
        If ``param`` is ``"fov"`` (or a recognised alias), only matches under
        ``*.encodedSpace.*`` are returned (``reconSpace`` is excluded).

    Special rule for matrix size:
        If ``param`` is ``"matrixsize"`` (or a recognised alias), only matches under
        ``*.encodedSpace.*`` are returned (``reconSpace`` is excluded).

    Args:
        param (str): Parameter name to search for (e.g. ``"TR"``, ``"fov"``,
            ``"bandwidth"``). Case-insensitive.
        xml_header (str | bytes): Raw ISMRMRD XML header string or bytes.

    Returns:
        np.ndarray: Matched values as a 1-D array with dtype:
            - ``float64`` if all values consist of digits and ``'.'`` only,
            - ``bool`` if all values are ``"True"`` or ``"False"``,
            - ``str`` (``<U...``) otherwise.
            Returns an empty ``float64`` array if ``param`` is empty or no matches are
            found.
    """
    raw = (param or "").strip()
    raw_l = raw.lower()

    is_two_letter = bool(re.fullmatch(r"[A-Za-z]{2}", raw))

    if isinstance(xml_header, bytes):
        xml_text = xml_header.decode("utf-8", errors="ignore")
    else:
        xml_text = xml_header

    if isinstance(xml_text, memoryview):
        xml_text = xml_text.tobytes().decode("utf-8")

    xml_text = (xml_text or "").strip()

    root = ET.fromstring(xml_text)

    def local(tag: str) -> str:
        return tag.split("}", 1)[-1] if "}" in tag else tag

    def text_of(elem: ET.Element | None) -> str:
        if elem is None:
            return ""
        return (elem.text or "").strip()

    aliases = {
        "fov": [
            "fov",
            "fieldofview",
            "fieldofview_mm",
            "fieldofviewmm",
            "field_of_view",
        ],
        "matrixsize": [
            "matrixsize",
            "matrix_size",
            "matrix",
            "matricesize",
            "baseResolution".lower(),  # optional: Siemens-style naming
        ],
        "tr": ["tr", "repetitiontime", "reptime"],
        "ti": ["ti", "inversiontime"],
        "te": ["te", "echotime"],
    }

    fov_alias_set = set(aliases["fov"])
    is_fov_search = raw_l in fov_alias_set

    matrix_alias_set = set(aliases["matrixsize"])
    is_matrix_search = raw_l in matrix_alias_set

    # tokens for substring matching (non-2-letter)
    tokens = [raw_l] if is_two_letter else aliases.get(raw_l, [raw_l])

    # Match helpers
    if is_two_letter:

        def matches_path(path: str) -> bool:
            last = path.rsplit(".", 1)[-1]
            return last.lower() == raw_l
    else:

        def matches_path(path: str) -> bool:
            p = path.lower()
            return any(t in p for t in tokens)

    if is_two_letter:

        def matches_userparam_name(name: str) -> bool:
            return name.strip().lower() == raw_l
    else:

        def matches_userparam_name(name: str) -> bool:
            n = (name or "").lower()
            return any(t in n for t in tokens)

    def allow_path(path: str) -> bool:
        # Apply the same exception rule for FOV and matrix size:
        # only allow matches under encodedSpace (exclude reconSpace).
        if not (is_fov_search or is_matrix_search):
            return True
        return ".encodedspace." in path.lower()

    # Collect (path, value) with stable de-dupe
    results: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def add(path: str, value: str) -> None:
        if not allow_path(path):
            return
        key = (path, value)
        if key in seen:
            return
        seen.add(key)
        results.append(key)

    def walk_path_matches(elem: ET.Element, path: str) -> None:
        tag = local(elem.tag)
        cur_path = f"{path}.{tag}" if path else tag

        txt = text_of(elem)
        if txt and matches_path(cur_path):
            add(cur_path, txt)

        for child in list(elem):
            walk_path_matches(child, cur_path)

    _re_numeric_dot = re.compile(r"^[0-9.]+$")

    def print_paths_and_convert() -> np.ndarray:
        if not results:
            print("- Parameters returned as np.array: []")
            return np.array([], dtype=float)

        paths = [p for (p, _v) in results]
        vals = [v for (_p, v) in results]

        lowered = [v.strip().lower() for v in vals]
        is_bool = all(v in ("true", "false") for v in lowered)
        is_float = (not is_bool) and all(_re_numeric_dot.fullmatch(v.strip()) for v in vals)

        if is_bool:
            if print_all:
                print("- Parameters returned as bool in np.array:")
                print("\n".join(f"   - {p} -> {v}" for p, v in zip(paths, vals)))
            return np.array([v == "true" for v in lowered], dtype=bool)

        if is_float:
            if print_all:
                print("- Parameters returned as float in np.array:")
                print("\n".join(f"   - {p} -> {v}" for p, v in zip(paths, vals)))
            return np.array([float(v) for v in vals], dtype=float)

        if print_all:
            print("- Parameters returned as str in np.array:")
            print("\n".join(f"   - {p} -> {v}" for p, v in zip(paths, vals)))
        return np.array(vals, dtype=str)

    # ------------------ 1) sequenceParameters only ------------------
    seq_results: list[tuple[str, str]] = []
    seq_seen: set[tuple[str, str]] = set()

    for elem in root.iter():
        if local(elem.tag) == "sequenceParameters":
            # collect into temporary list so priority rule can return only these
            def _add_seq(path: str, value: str) -> None:
                if not allow_path(path):
                    return
                key = (path, value)
                if key in seq_seen:
                    return
                seq_seen.add(key)
                seq_results.append(key)

            def _walk_seq(e: ET.Element, p: str) -> None:
                tag = local(e.tag)
                cur = f"{p}.{tag}" if p else tag
                txt = text_of(e)
                if txt and matches_path(cur):
                    _add_seq(cur, txt)
                for ch in list(e):
                    _walk_seq(ch, cur)

            _walk_seq(elem, "hdr")

    if seq_results:
        results = seq_results
        return print_paths_and_convert()

    # ------------------ 2) userParameters ------------------
    for user_params in root.iter():
        if local(user_params.tag) != "userParameters":
            continue

        for up in list(user_params):
            up_tag = local(up.tag)
            if up_tag not in {
                "userParameterDouble",
                "userParameterLong",
                "userParameterString",
                "userParameterBase64",
            }:
                continue

            name_elem = None
            value_elem = None
            for child in list(up):
                t = local(child.tag)
                if t == "name":
                    name_elem = child
                elif t == "value":
                    value_elem = child

            pname = text_of(name_elem)
            pval = text_of(value_elem)

            if pname and matches_userparam_name(pname) and pval:
                add(f"hdr.userParameters.{up_tag}.{pname}.value", pval)

    if results:
        return print_paths_and_convert()

    # ------------------ 3) fallback: whole header ------------------
    walk_path_matches(root, "hdr")
    return print_paths_and_convert()




def write_nii_from_xml(
    data: np.ndarray,
    xml_hdr,
    filename: str | Path = "test_nii-delete-or-rename.nii",
    MRpro_recon: bool = False,
) -> None:
    """Save a 3D array as a NIfTI file, with orientation and voxel size read from an ISMRMRD XML header.

    The acquisition plane and voxel resolution are extracted from ``xml_hdr``.  The data
    axes are permuted with ``np.transpose`` to bring them into the canonical scanner order
    ``(y, z, x)`` regardless of the original acquisition plane.  A non-diagonal NIfTI
    affine then encodes a 90° clockwise rotation in the ``(y, z)`` plane so that the
    volume is displayed with the correct anatomical orientation in NIfTI viewers.  The
    magnitude of the array is saved as ``float32``. The xml_header is attached to the nii file as a Nifti1Extension.

    Args:
        data (np.ndarray): 3D image array of shape ``(A, B, C)`` matching the acquisition
            axes ``(plane[0], plane[1], slice-encoding axis)``.  Complex arrays are
            supported; the magnitude is taken automatically.
        xml_hdr: ISMRMRD XML header object.  Must contain ``"plane"``, ``"matrix"``, and
            ``"fov"`` parameters readable by :func:`fmi.io.readers.GetParam_from_xml`.
        filename (str | Path): Output path for the ``.nii`` or ``.nii.gz`` file.

    Returns:
        None
    """

    # -------------------------- CHECK plane and reorient to standard -----------------------

    # ---- Check input array dimensions ----
    if data.ndim != 3:
        raise ValueError("Input array must be 3D.")

    if MRpro_recon:
        print("MRpro Reconstruction Specified. Data Transposed.")
        data = np.transpose(data, (2, 1, 0))

    # ---- Get orientation (plane) and voxel size from ISMRMRD XML header ----
    # 'plane' is a 2-letter string describing the two in-plane axes (e.g. "yz", "xz").
    plane = str(GetParam_from_xml("plane", xml_hdr, print_all=False)[0])

    matrix = GetParam_from_xml("matrix", xml_hdr, print_all=False)
    fov = GetParam_from_xml("fov", xml_hdr, print_all=False)

    # Voxel size in mm along each scanner axis (same order as matrix/fov arrays).
    res = fov / matrix

    # ---- Check plane string ----
    all_axes = ["y", "z", "x"]
    if len(plane) != 2 or any(c not in all_axes for c in plane):
        raise ValueError("axes_str must be length 2 and only contain letters from 'x', 'y', 'z'")

    # ---- Determine the missing (slice-encoding) axis ----
    # The input data axes are: [plane[0], plane[1], missing].
    # We need to reorder them to the canonical MQ order: [y, z, x].
    missing = (set(all_axes) - set(plane)).pop()

    # ---- Build a mapping from axis name to its current index in the data ----
    # e.g. for plane="xz": axis_map = {'x': 0, 'z': 1, 'y': 2}
    axis_map = {ax: i for i, ax in enumerate(plane + missing)}

    # ---- Compute the permutation that brings data to (y, z, x) order ----
    # new_order[i] = index in the original array of the axis to place at position i.
    new_order = [axis_map[ax] for ax in all_axes]

    # After np.transpose, data_new_order.shape = (Ny, Nz, Nx) regardless of 'plane'.
    data_new_order = np.transpose(data, axes=new_order)

    # Apply the same permutation to the voxel sizes so res_new_order = [dy, dz, dx].
    res_new_order = res[new_order]
    dy, dz, dx = res_new_order

    # ---- Build the NIfTI affine ----
    # After the transpose, voxel index (i, j, k) maps to anatomical axis (y, z, x).
    # We apply a 90° clockwise rotation in the (y, z) plane to match NIfTI convention.
    # This rotation (without modifying the data array) is encoded directly in the affine:
    #
    #   x_world = -dz * j                          (column index → L/R world axis, flipped)
    #   y_world = -dy * i + dy*(Ny-1)              (row index → A/P world axis, flipped and shifted to keep origin positive)
    #   z_world = -dx * k                          (slice index → S/I world axis, flipped)
    #
    # The column-vector [x_world, y_world, z_world, 1]^T = affine @ [i, j, k, 1]^T.
    Ny = data_new_order.shape[0]  # number of voxels along the y (row) axis

    affine = np.array(
        [
            [0, -dz, 0, 0],
            [-dy, 0, 0, dy * (Ny - 1)],
            [0, 0, -dx, 0],
            [0, 0, 0, 1],
        ]
    )

    img = nib.Nifti1Image(abs(data_new_order).astype(np.float32), affine)  # type: ignore

    # ---- Add XML header as a NIfTI extension ----
    ext = Nifti1Extension(6, xml_hdr)  # 6 == comment extension
    img.header.extensions.append(ext)

    print("img shape : ", data_new_order.shape)

    # ---- Set voxel-size metadata (zooms, qform, sform) ----
    # zooms are set from the reordered resolution to match the transposed array axes.
    img.header.set_zooms(res_new_order.tolist())
    img.set_qform(affine, code=1)
    img.set_sform(affine, code=1)

    nib.save(img, filename)  # type: ignore
    print(f"Saved NIfTI file to {filename}")









def extract_position_technique_and_order(input: Path):
    filename = input.name
    parts = filename.split("_")

    # Technique: 3rd element after splitting
    #technique = parts[3] if len(parts) > 2 else None
        # Technique
    techniques = ["IDC", "IFFT", "IROC", "CPRIter", "CPROne"]
    technique = next((t for t in techniques if t.lower() in filename.lower()), pd.NA)

    # Order
    if "Order" in filename:
        match = re.search(r"Order(\d+)", filename, re.IGNORECASE)
        order = match.group(1) if match else pd.NA
    else:
        order = pd.NA

    # Position
    if "position" in filename:
        match = re.search(r"position(\w)", filename, re.IGNORECASE)
        position = match.group(1) if match else pd.NA
    else:
        position = pd.NA

    return position, technique, order

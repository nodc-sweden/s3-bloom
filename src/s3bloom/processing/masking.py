"""WQSF (Water Quality Science Flags) quality / cloud masking.

The WQSF layer in every OLCI L2 product is a per-pixel **bitfield**:
each bit indicates one quality condition (cloud, sun glint, saturation,
algorithm failure, ...). The masking step:

1. Reads the flag-name → bit-mask table from the WQSF variable's
   ``flag_meanings`` and ``flag_masks`` attributes (rather than
   hardcoding bit positions, since these can change between processing
   baselines).
2. OR-combines the bit masks for the flags the caller wants to reject.
3. Marks every pixel where ``wqsf & combined_bitmask != 0`` as bad.
4. Returns a boolean :class:`xarray.DataArray` (``True`` = masked / bad).

The mask is applied with :func:`apply_mask`, which sets bad pixels to
NaN. NaN propagates correctly through resampling and is naturally
ignored by the NaN-skipping geometric-mean compositing.

Why use product metadata instead of a static table
--------------------------------------------------
EUMETSAT recommends reading flag masks from the product itself because
the bit assignments are not formally guaranteed to be stable across
processing baselines. Reading from metadata makes the pipeline
forward-compatible with future reprocessings.
"""

from __future__ import annotations

import logging

import dask.array as da
import numpy as np
import xarray as xr
from satpy import Scene

from s3bloom.config import MaskingConfig

logger = logging.getLogger(__name__)


def build_quality_mask(
    scene: Scene,
    masking_config: MaskingConfig,
    *,
    product: str | None = None,
) -> xr.DataArray:
    """Build a boolean per-pixel quality mask from WQSF flags.

    Parameters
    ----------
    scene : satpy.Scene
        Scene that has been ``load()``-ed with ``"wqsf"`` available.
    masking_config : MaskingConfig
        Strictness preset (or custom flag list) to apply.
    product : str, optional
        Dataset name (``"chl_nn"``, ``"chl_oc4me"``, …). If given,
        product-aware flags (BAC processing-chain, per-product algorithm
        failure) are added on top of the common flags. If ``None``, only
        common flags are used.

    Returns
    -------
    xarray.DataArray
        Boolean mask with the same dims/coords as ``scene["wqsf"]``;
        ``True`` means "pixel is bad, mask it out".

    Notes
    -----
    Returns an all-``False`` mask (and warns) in two non-fatal cases:

    * the WQSF layer carries no ``flag_meanings`` / ``flag_masks``
      metadata at all;
    * none of the requested flag names match anything in the metadata.

    Both situations indicate something is wrong upstream, but masking
    nothing is more useful than failing the entire run.
    """
    if product is not None:
        flags_to_mask = masking_config.flags_for_product(product)
    else:
        flags_to_mask = masking_config.flags
    logger.info("Building quality mask with flags: %s", flags_to_mask)

    wqsf = scene["wqsf"]

    flag_meanings, flag_masks = _get_flag_definitions(wqsf)

    if not flag_meanings:
        logger.warning(
            "No flag definitions found in WQSF dataset. "
            "Falling back to no masking."
        )
        return xr.DataArray(
            da.zeros(wqsf.shape, dtype=bool, chunks=wqsf.data.chunks),
            dims=wqsf.dims,
            coords=wqsf.coords,
        )

    combined_bitmask = _build_combined_bitmask(
        flags_to_mask, flag_meanings, flag_masks
    )

    if combined_bitmask == 0:
        logger.warning(
            "None of the requested flags were found in WQSF metadata. "
            "No pixels will be masked."
        )
        return xr.DataArray(
            da.zeros(wqsf.shape, dtype=bool, chunks=wqsf.data.chunks),
            dims=wqsf.dims,
            coords=wqsf.coords,
        )

    # Cast to uint64 because some flag-mask arrays in product metadata
    # exceed int32; the bitwise-AND must happen at >= the widest mask's
    # bit-width.
    wqsf_values = wqsf.values if not hasattr(wqsf.data, "dask") else wqsf.data
    mask_data = (wqsf_values.astype(np.uint64) & np.uint64(combined_bitmask)) != 0

    mask = xr.DataArray(
        mask_data,
        dims=wqsf.dims,
        coords=wqsf.coords,
        attrs={"description": f"Quality mask ({masking_config.preset})"},
    )

    if hasattr(mask_data, "compute"):
        masked_count = "deferred (dask)"
    else:
        masked_count = int(np.sum(mask_data))
        total = int(np.prod(mask_data.shape))
        pct = 100.0 * masked_count / total if total > 0 else 0
        logger.info(
            "Mask: %d / %d pixels masked (%.1f%%)",
            masked_count,
            total,
            pct,
        )

    return mask


def apply_mask(
    data: xr.DataArray,
    mask: xr.DataArray,
) -> xr.DataArray:
    """Set every pixel where ``mask`` is ``True`` to ``NaN``.

    Returns a new DataArray; the input is not mutated. NaN is preferred
    over a sentinel value because xarray and dask propagate it through
    arithmetic, and the NaN-skipping geometric mean ignores it during
    compositing.
    """
    return data.where(~mask, other=np.nan)


def _get_flag_definitions(
    wqsf: xr.DataArray,
) -> tuple[list[str], list[int]]:
    """Extract ``flag_meanings`` / ``flag_masks`` from a WQSF DataArray.

    Both attributes are normalised to plain Python lists. The two
    attributes are aligned positionally: ``meanings[i]`` is the name of
    the flag whose bitmask is ``masks[i]``. If the lengths disagree
    (which has been observed in malformed products) the lists are
    truncated to the shorter common prefix and a warning is logged.
    """
    attrs = wqsf.attrs

    meanings_raw = attrs.get("flag_meanings", "")
    if isinstance(meanings_raw, str):
        meanings = meanings_raw.split()
    elif isinstance(meanings_raw, (list, tuple)):
        meanings = list(meanings_raw)
    else:
        meanings = []

    masks_raw = attrs.get("flag_masks", [])
    if isinstance(masks_raw, np.ndarray):
        masks = masks_raw.tolist()
    elif isinstance(masks_raw, (list, tuple)):
        masks = list(masks_raw)
    else:
        masks = []

    if len(meanings) != len(masks):
        logger.warning(
            "flag_meanings (%d) and flag_masks (%d) length mismatch",
            len(meanings),
            len(masks),
        )
        min_len = min(len(meanings), len(masks))
        meanings = meanings[:min_len]
        masks = masks[:min_len]

    return meanings, masks


def _build_combined_bitmask(
    flags_to_mask: list[str],
    flag_meanings: list[str],
    flag_masks: list[int],
) -> int:
    """OR-combine the bitmasks of every flag name in *flags_to_mask*.

    If a requested flag name is not in ``flag_meanings`` exactly, a
    *prefix* match is attempted. This handles cases like
    ``RWNEG_O2``..``RWNEG_O8`` (negative water-leaving reflectance per
    band) where the precise flag spelling has varied between processing
    baselines.

    Unmatched flags are logged at WARNING level but do not raise; the
    pipeline continues with whichever flags it could resolve.
    """
    meaning_to_mask = dict(zip(flag_meanings, flag_masks))
    combined = 0
    matched = []
    unmatched = []

    for flag in flags_to_mask:
        if flag in meaning_to_mask:
            combined |= int(meaning_to_mask[flag])
            matched.append(flag)
        else:
            for meaning in flag_meanings:
                if meaning.startswith(flag):
                    combined |= int(meaning_to_mask[meaning])
                    matched.append(meaning)
                    break
            else:
                unmatched.append(flag)

    if matched:
        logger.debug("Matched flags: %s", matched)
    if unmatched:
        logger.warning(
            "Flags not found in product metadata: %s. "
            "Available flags: %s",
            unmatched,
            flag_meanings[:30],
        )

    return combined

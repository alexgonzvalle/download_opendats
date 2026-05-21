from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import xarray as xr


def open_local_datasets(
        paths: Sequence[str],
        concat_dim: str = "time",
        combine: str = "by_coords",
        dask_chunks: Optional[Dict[str, int]] = None,
        var_save: Optional[Union[str, Sequence[str]]] = None,
        lonlat_target: Optional[Tuple[float, float]] = None,
        inds_time: Optional[Union[int, slice, Sequence[int]]] = None,
) -> xr.Dataset:
    paths_list = list(paths)
    if not paths_list:
        raise ValueError("No hay ficheros para abrir.")

    def preprocess(ds: xr.Dataset) -> xr.Dataset:
        return _filter_dataset(ds, var_save, lonlat_target, inds_time)

    if len(paths_list) == 1:
        kwargs = {"chunks": dask_chunks} if dask_chunks is not None else {}
        return preprocess(xr.open_dataset(paths_list[0], **kwargs))

    kwargs = {"combine": combine}
    if dask_chunks is not None:
        kwargs["chunks"] = dask_chunks
    return xr.open_mfdataset(paths_list, concat_dim=concat_dim, preprocess=preprocess, **kwargs)


def _filter_dataset(
        ds: xr.Dataset,
        var_save: Optional[Union[str, Sequence[str]]],
        lonlat_target: Optional[Tuple[float, float]],
        inds_time: Optional[Union[int, slice, Sequence[int]]],
) -> xr.Dataset:
    if var_save is not None:
        requested = [var_save] if isinstance(var_save, str) else list(var_save)
        existing = [var for var in requested if var in ds.variables]
        ds = ds[existing] if existing else ds

    if lonlat_target is not None:
        ds = select_nearest_lonlat(ds, lonlat_target)

    if inds_time is not None and ("time" in ds.dims or "time" in ds.coords):
        ds = ds.isel(time=inds_time)

    return ds


def select_nearest_lonlat(ds: xr.Dataset, lonlat_target: Tuple[float, float]) -> xr.Dataset:
    lon_target, lat_target = lonlat_target

    if ("lon" in ds and "lat" in ds) and ("x" in ds.dims and "y" in ds.dims):
        dist = np.sqrt((ds["lon"] - lon_target) ** 2 + (ds["lat"] - lat_target) ** 2)
        dims = [dim for dim in dist.dims if dim in ("x", "y")]
        if len(dims) < 2:
            if "lon" in ds.coords and "lat" in ds.coords:
                return ds.sel(lon=lon_target, lat=lat_target, method="nearest")
            raise ValueError("No se pudo aplicar lonlat_target.")

        min_idx = dist.argmin(dim=dims)
        x_sel = ds["x"].isel(x=int(min_idx["x"])) if "x" in min_idx else ds["x"]
        y_sel = ds["y"].isel(y=int(min_idx["y"])) if "y" in min_idx else ds["y"]
        return ds.sel(x=x_sel, y=y_sel)

    if "lon" in ds.coords and "lat" in ds.coords:
        return ds.sel(lon=lon_target, lat=lat_target, method="nearest")

    raise ValueError("No encuentro lon/lat para aplicar lonlat_target.")

# opendat.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import tempfile
import logging
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import xarray as xr
import xmltodict
import requests
from requests.auth import HTTPBasicAuth
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

JsonLike = Dict[str, Any]
UrlLike = Union[str, Sequence[str]]


def _default_logger(name: str = "opendat") -> logging.Logger:
    """
    Create a default logger with a stream handler if no handlers exist.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        fmt = logging.Formatter("[%(levelname)s] %(name)s - %(message)s")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


def _as_list(x: Any) -> List[Any]:
    if x is None:
        return []
    return x if isinstance(x, list) else [x]


def _safe_get(d: Dict[str, Any], *keys: str) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


class Opendat:
    """
    Cliente para leer catálogos XML (típicamente THREDDS) y descargar ficheros netCDF.

    :param user : str | None. Usuario (si el servidor requiere auth básica).
    :param passw : str | None. Contraseña (si el servidor requiere auth básica).
    :param logger : logging.Logger | None. Logger externo. Si no se pasa, se crea uno por defecto.
    """

    def __init__(self, user: Optional[str] = None, passw: Optional[str] = None, logger: Optional[logging.Logger] = None):
        self.user = user
        self.passw = passw
        self.logger = logger or _default_logger()

        # Resultados
        self.files_avbl: List[Dict[str, Any]] = []
        self.ds: Optional[Union[xr.Dataset, List[xr.Dataset]]] = None

        # Sesión (siguiente nivel): reusar conexión + retries
        self._session: Optional[requests.Session] = None

    def _get_session(
            self,
            retries: int = 3,
            backoff_factor: float = 0.5,
            status_forcelist: Tuple[int, ...] = (429, 500, 502, 503, 504),
            allowed_methods: Tuple[str, ...] = ("GET",),
    ) -> requests.Session:
        if self._session is not None:
            return self._session

        session = requests.Session()

        retry = Retry(
            total=retries,
            connect=retries,
            read=retries,
            status=retries,
            backoff_factor=backoff_factor,
            status_forcelist=status_forcelist,
            allowed_methods=allowed_methods,
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        self._session = session
        return session

    def close(self) -> None:
        """
        Cierra la sesión HTTP si existe.
        """
        if self._session is not None:
            self._session.close()
            self._session = None

    def __enter__(self) -> "Opendat":
        self._get_session()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def catalog(self, url_catalog: UrlLike, key_file: str = "@name", text_find: Optional[str] = None, last_file: bool = False, timeout: int = 60, verify: bool = True, sort_by_name: bool = True) -> List[Dict[str, Any]]:
        """ Lee uno o varios catálogos XML y construye la lista de ficheros .nc disponibles.

        :param url_catalog: url to catalog
        :param key_file: key to get file name
        :param text_find: text to filter
        :param last_file: last file from catalog
        :return: files available
        """

        self.files_avbl = []

        urls = [url_catalog] if isinstance(url_catalog, str) else list(url_catalog)
        if not urls:
            raise ValueError("url_catalog está vacío.")

        session = self._get_session()
        self.logger.info(f"Leyendo catálogo(s): {len(urls)}")

        # Get file of catalog
        for url in urls:
            self.logger.info(f"Descargando XML de catálogo: {url}")

            resp = session.get(url, timeout=timeout, verify=verify)
            if resp.status_code != 200:
                msg = f"Error leyendo catálogo (HTTP {resp.status_code}) en {url}"
                self.logger.error(msg)
                raise ValueError(msg)

            try:
                data = xmltodict.parse(resp.content)
            except Exception as e:
                msg = f"No se pudo parsear XML del catálogo en {url}. Error: {e}"
                self.logger.error(msg)
                raise ValueError(msg)

            # Extraer datasets
            datasets = self._extract_datasets_from_catalog(data)
            if not datasets:
                self.logger.warning(f"No se encontraron datasets en el catálogo: {url}")
                continue

            # Filtrar a .nc
            nc_datasets = []
            for d in datasets:
                name = d.get(key_file)
                if isinstance(name, str) and name.endswith(".nc"):
                    nc_datasets.append(d)

            self.logger.info(f"Catálogo OK: {url}. Ficheros .nc: {len(nc_datasets)}")
            self.files_avbl.extend(nc_datasets)

        if sort_by_name:
            self.files_avbl.sort(key=lambda d: str(d.get(key_file, "")))

        self.logger.info(f"Ficheros totales disponibles: {len(self.files_avbl)}")

        # Filter by text
        if text_find is not None:
            before = len(self.files_avbl)
            self.files_avbl = [d for d in self.files_avbl if text_find in str(d.get(key_file, ""))]
            self.logger.info(f'Filtro: "{text_find}". {before} -> {len(self.files_avbl)} ficheros.')

        # Filter by last file
        if last_file and self.files_avbl:
            self.files_avbl = [self.files_avbl[-1]]
            self.logger.info("Filtro: 'Fichero más reciente' (último tras ordenado). 1 fichero.")

        return self.files_avbl

    def _extract_datasets_from_catalog(self, parsed_xml: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extrae recursivamente listas de 'dataset' desde un catálogo tipo THREDDS.
        Maneja casos donde dataset es dict o list y anidaciones.

        :param parsed_xml:
        :return: list[dict]
        """

        # Estructura típica: catalog -> dataset -> dataset (nested)
        root_ds = _safe_get(parsed_xml, "catalog", "dataset")
        if root_ds is None:
            # A veces el root puede tener otro nombre, o ya estar en dataset directamente.
            root_ds = parsed_xml.get("dataset")

        datasets: List[Dict[str, Any]] = []

        def walk(node: Any) -> None:
            if node is None:
                return
            if isinstance(node, list):
                for it in node:
                    walk(it)
                return
            if isinstance(node, dict):
                # Si es un dataset "hoja" (puede tener @name)
                # Lo añadimos y seguimos buscando nested datasets
                if any(k.startswith("@") for k in node.keys()) or "dataset" in node:
                    # Si parece dataset, lo añadimos como candidato
                    datasets.append(node)

                # THREDDS suele anidar: node['dataset'] -> list/dict
                nested = node.get("dataset")
                if nested is not None:
                    walk(nested)
                return

        walk(root_ds)

        # A veces se duplica el root (por cómo añadimos node siempre); deduplicación por id/name si existe
        seen = set()
        unique: List[Dict[str, Any]] = []
        for d in datasets:
            # clave de dedupe razonable:
            k = d.get("@ID") or d.get("@urlPath") or d.get("@name") or str(id(d))
            if k in seen:
                continue
            seen.add(k)
            unique.append(d)

        return unique

    def _select_nearest_lonlat(self, ds: xr.Dataset, lonlat_target: Tuple[float, float]) -> xr.Dataset:
        """
        Selecciona el punto más cercano a (lon,lat), soportando dos escenarios:
        1) lon/lat en 2D con dims típicas (x,y)
        2) lon/lat como coords 1D y selección nearest directa

        Si tu OPENDAT siempre es x/y con lon/lat 2D, este método te vale tal cual.
        """
        lon_target, lat_target = lonlat_target

        # Caso 2D típico: lon y lat como DataArrays (x,y)
        if ("lon" in ds and "lat" in ds) and ("x" in ds.dims and "y" in ds.dims):
            lon = ds["lon"]
            lat = ds["lat"]

            # dist en el grid
            dist = np.sqrt((lon - lon_target) ** 2 + (lat - lat_target) ** 2)

            # dist puede tener dims (x,y) o (y,x), cogemos las que existan:
            dims = [d for d in dist.dims if d in ("x", "y")]
            if len(dims) < 2:
                # fallback
                self.logger.warning("lon/lat existen pero no encuentro dims x/y claras; intento nearest simple.")
                if "lon" in ds.coords and "lat" in ds.coords:
                    return ds.sel(lon=lon_target, lat=lat_target, method="nearest")
                raise ValueError("No se pudo aplicar lonlat_target (estructura inesperada).")

            min_idx = dist.argmin(dim=dims)

            # min_idx tiene coords por dim
            if "x" in min_idx:
                x_idx = int(min_idx["x"])
                x_sel = ds["x"].isel(x=x_idx)
            else:
                x_sel = ds["x"]
            if "y" in min_idx:
                y_idx = int(min_idx["y"])
                y_sel = ds["y"].isel(y=y_idx)
            else:
                y_sel = ds["y"]

            out = ds.sel(x=x_sel, y=y_sel)
            self.logger.info(f"Filtro coords (grid): x={out['x'].values}, y={out['y'].values}")
            return out

        # Caso coords 1D (lon, lat)
        if ("lon" in ds.coords) and ("lat" in ds.coords):
            out = ds.sel(lon=lon_target, lat=lat_target, method="nearest")
            self.logger.info("Filtro coords (1D nearest) aplicado.")
            return out

        raise ValueError("No encuentro lon/lat para aplicar lonlat_target.")

    def get_file_names(self, key_file: str = "@name") -> List[str]:
        """
        Devuelve solo los nombres de fichero del catálogo.
        """
        return [str(d.get(key_file)) for d in self.files_avbl if d.get(key_file) is not None]

    def download_to_dir(
            self,
            url_netcdf: str,
            out_dir: str,
            key_file: str = "@name",
            timeout: int = 120,
            verify: bool = True,
            reverse: bool = True,
            stream_chunk_bytes: int = 1024 * 1024,
            overwrite: bool = False,
    ) -> List[str]:
        """ Descarga los ficheros del catálogo a un directorio local (cache).
        :param url_netcdf:
        :param out_dir:
        :param key_file:
        :param timeout:
        :param verify:
        :param reverse:
        :param stream_chunk_bytes:
        :param overwrite:
        :return:
        """

        if not self.files_avbl:
            raise ValueError("No hay ficheros en 'files_avbl'. Ejecuta catalog() primero.")

        os.makedirs(out_dir, exist_ok=True)

        session = self._get_session()
        auth = HTTPBasicAuth(self.user, self.passw) if (self.user and self.passw) else None

        file_list = self.files_avbl[::-1] if reverse else self.files_avbl
        paths: List[str] = []

        for item in file_list:
            fname = item.get(key_file)
            if not fname:
                raise ValueError(f"Entrada sin '{key_file}': {item}")

            url_file = url_netcdf + str(fname)
            out_path = os.path.join(out_dir, str(fname))
            if os.path.exists(out_path) and not overwrite:
                self.logger.info(f"Skip (ya existe): {out_path}")
                paths.append(out_path)
                continue

            self.logger.info(f"GET {url_file} -> {out_path}")
            resp = session.get(url_file, auth=auth, timeout=timeout, verify=verify, stream=True)
            try:
                if resp.status_code == 401:
                    raise ValueError("Auth fallida (401). Revisa usuario/contraseña.")
                if resp.status_code == 403:
                    raise ValueError("Acceso prohibido (403). No tienes permisos.")
                if resp.status_code != 200:
                    raise ValueError(f"HTTP {resp.status_code} en {url_file}")

                with open(out_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=stream_chunk_bytes):
                        if chunk:
                            f.write(chunk)
                paths.append(out_path)
            finally:
                resp.close()

        self.logger.info(f"Descargados {len(paths)} fichero(s) a {out_dir}")
        return paths

    def open_dataset_local(
            self,
            paths: Sequence[str],
            concat_dim: str = "time",
            combine: str = "by_coords",
            dask_chunks: Optional[Dict[str, int]] = None,
            var_save: Optional[Union[str, Sequence[str]]] = None,
            lonlat_target: Optional[Tuple[float, float]] = None,
            inds_time: Optional[Union[int, slice, Sequence[int]]] = None,
    ) -> xr.Dataset:
        """
        Siguiente nivel: abre múltiples netCDF locales de golpe con xarray.open_mfdataset.

        - Ideal para muchos ficheros y para uso con Dask (chunks).
        - Aplica filtros vía preprocess para no cargar de más.

        Nota: open_mfdataset funciona mejor si los ficheros son homogéneos.
        """
        vars_list: Optional[List[str]] = None
        if var_save is not None:
            vars_list = [var_save] if isinstance(var_save, str) else list(var_save)

        def _preprocess(ds: xr.Dataset) -> xr.Dataset:
            if vars_list is not None:
                existing = [v for v in vars_list if v in ds.variables]
                ds = ds[existing] if existing else ds
            if lonlat_target is not None:
                ds = self._select_nearest_lonlat(ds, lonlat_target)
            if inds_time is not None and ("time" in ds.dims or "time" in ds.coords):
                ds = ds.isel(time=inds_time)
            return ds

        kwargs: Dict[str, Any] = dict(combine=combine)
        if dask_chunks is not None:
            kwargs["chunks"] = dask_chunks

        ds = xr.open_mfdataset(list(paths), concat_dim=concat_dim, preprocess=_preprocess, **kwargs)
        self.logger.info(f"open_mfdataset OK. dims={dict(ds.dims)}")
        return ds

    def open_dataset_url(
            self,
            url_netcdf: str,
            key_file: str = "@name",
            concat: bool = False,
            var_save: Optional[Union[str, Sequence[str]]] = None,
            lonlat_target: Optional[Tuple[float, float]] = None,
            inds_time: Optional[Union[int, slice, Sequence[int]]] = None,
            timeout: int = 120,
            verify: bool = True,
            reverse: bool = True,
            dask_chunks: Optional[Dict[str, int]] = None,
            keep_in_memory: bool = True,
    ) -> Union[xr.Dataset, List[xr.Dataset]]:
        """
        Lee datasets sin dejar ficheros en disco (usa temporales internos y los borra).

        - keep_in_memory=True: hace ds.load() para que, tras borrar el temporal, el dataset siga usable.
        - concat=True: devuelve un xr.Dataset concatenado en 'time'
        - concat=False: devuelve list[xr.Dataset]
        """
        if not self.files_avbl:
            raise ValueError("No hay ficheros en 'files_avbl'. Ejecuta catalog() primero.")

        s = self._get_session()
        auth = self._auth()

        vars_list: Optional[List[str]] = None
        if var_save is not None:
            vars_list = [var_save] if isinstance(var_save, str) else list(var_save)

        items = self.files_avbl[::-1] if reverse else self.files_avbl
        ds_all: List[xr.Dataset] = []

        self.logger.info(f"Lectura al vuelo: {len(items)} fichero(s). concat={concat}")

        for it in tqdm(items, total=len(items), desc="Reading on the fly"):
            fname = it.get(key_file)
            if not fname:
                raise ValueError(f"Entrada sin '{key_file}': {it}")

            url_file = url_netcdf + str(fname)
            self.logger.info(f"GET {url_file}")

            tmp_path: Optional[str] = None
            ds_c: Optional[xr.Dataset] = None

            resp = s.get(url_file, auth=auth, timeout=timeout, verify=verify)
            try:
                if resp.status_code == 401:
                    raise ValueError("Auth fallida (401). Revisa usuario/contraseña.")
                if resp.status_code == 403:
                    raise ValueError("Acceso prohibido (403). No tienes permisos.")
                if resp.status_code != 200:
                    raise ValueError(f"HTTP {resp.status_code} en {url_file}")

                # temporal interno (NO se considera "descarga local" porque se borra)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".nc") as tmp:
                    tmp.write(resp.content)  # si quieres menos RAM: implementar stream->tmp (igual que download_to_dir)
                    tmp_path = tmp.name

                open_kwargs: Dict[str, Any] = {}
                if dask_chunks is not None:
                    open_kwargs["chunks"] = dask_chunks

                ds_c = xr.open_dataset(tmp_path, **open_kwargs)

                # filtros
                if vars_list is not None:
                    existing = [v for v in vars_list if v in ds_c.variables]
                    ds_c = ds_c[existing] if existing else ds_c

                if lonlat_target is not None:
                    ds_c = self._select_nearest_lonlat(ds_c, lonlat_target)

                if inds_time is not None and ("time" in ds_c.dims or "time" in ds_c.coords):
                    ds_c = ds_c.isel(time=inds_time)

                if keep_in_memory:
                    ds_c = ds_c.load()

                ds_all.append(ds_c)

            finally:
                resp.close()
                # borra temporal (si keep_in_memory=True, seguro)
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        if keep_in_memory:
                            os.remove(tmp_path)
                        else:
                            # si no cargas en memoria, el dataset puede depender del fichero
                            # y borrar puede romperlo; lo dejamos.
                            pass
                    except Exception:
                        pass

        if concat:
            if not ds_all:
                raise ValueError("No se cargó ningún dataset.")
            out = xr.concat(ds_all, dim="time") if len(ds_all) > 1 else ds_all[0]
            self.ds = out
            return out

        self.ds = ds_all
        return ds_all
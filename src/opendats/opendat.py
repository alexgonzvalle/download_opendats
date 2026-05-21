from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import requests
import xmltodict
import xarray as xr
from requests.adapters import HTTPAdapter
from requests.auth import HTTPBasicAuth
from urllib3.util.retry import Retry

from .catalog import extract_datasets
from .config import (
    ConfigLike,
    JsonLike,
    UrlLike,
    first_value,
    format_config_template,
    load_json_config,
    normalize_catalog_urls,
    section,
    urls_from_config,
)
from .datasets import open_local_datasets
from .logging_utils import default_logger


class Opendat:
    """
    Cliente JSON-first para resolver catalogos THREDDS y descargar ficheros NetCDF.
    """

    def __init__(self, config: ConfigLike, logger: Optional[logging.Logger] = None):
        self.config = load_json_config(config)
        auth_config = section(self.config, "auth")

        self.user = first_value(self.config, auth_config, "user", "username")
        self.passw = first_value(self.config, auth_config, "passw", "password")
        self.logger = logger or default_logger()

        self.files_avbl: List[Dict[str, Any]] = []
        self.paths: List[str] = []
        self.ds: Optional[xr.Dataset] = None
        self._session: Optional[requests.Session] = None

        self.paths = self._run_from_config()

    def close(self) -> None:
        if self._session is not None:
            self._session.close()
            self._session = None

    def __enter__(self) -> "Opendat":
        self._get_session()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def get_file_names(self, key_file: str = "@name") -> List[str]:
        return [str(item.get(key_file)) for item in self.files_avbl if item.get(key_file) is not None]

    def open_downloaded(
            self,
            concat_dim: str = "time",
            combine: str = "by_coords",
            dask_chunks: Optional[Dict[str, int]] = None,
            var_save: Optional[Union[str, Sequence[str]]] = None,
            lonlat_target: Optional[Tuple[float, float]] = None,
            inds_time: Optional[Union[int, slice, Sequence[int]]] = None,
    ) -> xr.Dataset:
        self.ds = open_local_datasets(
            self.paths,
            concat_dim=concat_dim,
            combine=combine,
            dask_chunks=dask_chunks,
            var_save=var_save,
            lonlat_target=lonlat_target,
            inds_time=inds_time,
        )
        self.logger.info(f"open_downloaded OK. dims={dict(self.ds.sizes)}")
        return self.ds

    def _run_from_config(self) -> List[str]:
        self._catalog_from_config()
        return self._download_from_config()

    def _catalog_from_config(self) -> List[Dict[str, Any]]:
        catalog_config = section(self.config, "catalog")
        url_catalog = urls_from_config(
            self.config,
            catalog_config,
            url_keys=("url_catalog", "url"),
            template_keys=("url_catalog_template", "url_template"),
        )
        if not url_catalog:
            raise ValueError("Falta 'url_catalog' o 'url_catalog_template' en la configuracion JSON.")

        return self._load_catalog(
            url_catalog=normalize_catalog_urls(url_catalog),
            key_file=first_value(self.config, catalog_config, "key_file", default="@name"),
            text_find=first_value(self.config, catalog_config, "text_find"),
            last_file=bool(first_value(self.config, catalog_config, "last_file", default=False)),
            timeout=int(first_value(self.config, catalog_config, "timeout", default=60)),
            verify=bool(first_value(self.config, catalog_config, "verify", default=True)),
            sort_by_name=bool(first_value(self.config, catalog_config, "sort_by_name", default=True)),
        )

    def _download_from_config(self) -> List[str]:
        download_config = section(self.config, "download")
        url_netcdf = first_value(self.config, download_config, "url_netcdf", "url_file_server", "url")
        if not url_netcdf:
            raise ValueError("Falta 'url_netcdf' en la configuracion JSON.")

        out_dir = first_value(self.config, download_config, "out_dir", "download_dir", default="data")
        return self._download_to_dir(
            url_netcdf=format_config_template(url_netcdf, self.config, download_config),
            out_dir=os.fspath(format_config_template(out_dir, self.config, download_config)),
            key_file=first_value(self.config, download_config, "key_file", default="@name"),
            timeout=int(first_value(self.config, download_config, "timeout", default=120)),
            verify=bool(first_value(self.config, download_config, "verify", default=True)),
            reverse=bool(first_value(self.config, download_config, "reverse", default=True)),
            stream_chunk_bytes=int(
                first_value(self.config, download_config, "stream_chunk_bytes", default=1024 * 1024)
            ),
            overwrite=bool(first_value(self.config, download_config, "overwrite", default=False)),
        )

    def _load_catalog(
            self,
            url_catalog: UrlLike,
            key_file: str = "@name",
            text_find: Optional[str] = None,
            last_file: bool = False,
            timeout: int = 60,
            verify: bool = True,
            sort_by_name: bool = True,
    ) -> List[Dict[str, Any]]:
        self.files_avbl = []
        urls = [url_catalog] if isinstance(url_catalog, str) else list(url_catalog)
        if not urls:
            raise ValueError("url_catalog esta vacio.")

        session = self._get_session()
        self.logger.info(f"Leyendo catalogo(s): {len(urls)}")

        for url in urls:
            datasets = self._read_catalog_datasets(session, url, timeout, verify)
            nc_datasets = []
            for dataset in datasets:
                name = dataset.get(key_file)
                if isinstance(name, str) and name.endswith(".nc"):
                    dataset["_catalog_url"] = url
                    nc_datasets.append(dataset)

            self.logger.info(f"Catalogo OK: {url}. Ficheros .nc: {len(nc_datasets)}")
            self.files_avbl.extend(nc_datasets)

        if sort_by_name:
            self.files_avbl.sort(key=lambda item: str(item.get(key_file, "")))

        self.logger.info(f"Ficheros totales disponibles: {len(self.files_avbl)}")
        self._filter_files(key_file=key_file, text_find=text_find, last_file=last_file)
        return self.files_avbl

    def _download_to_dir(
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
        if not self.files_avbl:
            raise ValueError("No hay ficheros en 'files_avbl'.")

        os.makedirs(out_dir, exist_ok=True)
        session = self._get_session()
        auth = HTTPBasicAuth(self.user, self.passw) if (self.user and self.passw) else None
        paths: List[str] = []

        for item in self.files_avbl[::-1] if reverse else self.files_avbl:
            fname = item.get(key_file)
            if not fname:
                raise ValueError(f"Entrada sin '{key_file}': {item}")

            url_file = self._file_url(url_netcdf, item, str(fname))
            out_path = os.path.join(out_dir, str(fname))
            if os.path.exists(out_path) and not overwrite:
                self.logger.info(f"Skip (ya existe): {out_path}")
                paths.append(out_path)
                continue

            self._download_file(session, url_file, out_path, auth, timeout, verify, stream_chunk_bytes)
            paths.append(out_path)

        self.logger.info(f"Descargados {len(paths)} fichero(s) a {out_dir}")
        return paths

    def _get_session(self) -> requests.Session:
        if self._session is not None:
            return self._session

        session = requests.Session()
        retry = Retry(
            total=3,
            connect=3,
            read=3,
            status=3,
            backoff_factor=0.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET",),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        self._session = session
        return session

    def _read_catalog_datasets(
            self,
            session: requests.Session,
            url: str,
            timeout: int,
            verify: bool,
    ) -> List[Dict[str, Any]]:
        self.logger.info(f"Descargando XML de catalogo: {url}")
        response = session.get(url, timeout=timeout, verify=verify)
        if response.status_code != 200:
            raise ValueError(f"Error leyendo catalogo (HTTP {response.status_code}) en {url}")

        try:
            return extract_datasets(xmltodict.parse(response.content))
        except Exception as exc:
            raise ValueError(f"No se pudo parsear XML del catalogo en {url}. Error: {exc}") from exc

    def _filter_files(self, key_file: str, text_find: Optional[str], last_file: bool) -> None:
        if text_find is not None:
            before = len(self.files_avbl)
            self.files_avbl = [item for item in self.files_avbl if text_find in str(item.get(key_file, ""))]
            self.logger.info(f'Filtro: "{text_find}". {before} -> {len(self.files_avbl)} ficheros.')

        if last_file and self.files_avbl:
            self.files_avbl = [self.files_avbl[-1]]
            self.logger.info("Filtro: fichero mas reciente. 1 fichero.")

    @staticmethod
    def _file_url(url_netcdf: str, item: JsonLike, fname: str) -> str:
        if item.get("@urlPath") and url_netcdf.rstrip("/").endswith("/fileServer"):
            return url_netcdf.rstrip("/") + "/" + str(item["@urlPath"]).lstrip("/")
        return url_netcdf + fname

    def _download_file(
            self,
            session: requests.Session,
            url_file: str,
            out_path: str,
            auth: Optional[HTTPBasicAuth],
            timeout: int,
            verify: bool,
            chunk_size: int,
    ) -> None:
        self.logger.info(f"GET {url_file} -> {out_path}")
        response = session.get(url_file, auth=auth, timeout=timeout, verify=verify, stream=True)
        try:
            if response.status_code == 401:
                raise ValueError("Auth fallida (401). Revisa usuario/contrasena.")
            if response.status_code == 403:
                raise ValueError("Acceso prohibido (403). No tienes permisos.")
            if response.status_code != 200:
                raise ValueError(f"HTTP {response.status_code} en {url_file}")

            with open(out_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
        finally:
            response.close()

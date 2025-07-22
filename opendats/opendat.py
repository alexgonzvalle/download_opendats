import os
import xarray as xr
import xmltodict
from urllib.request import urlopen

import wget
import bz2
import numpy as np
import datetime as dt

import requests
from requests.auth import HTTPBasicAuth
import tempfile

from IPython import get_ipython
if get_ipython().__class__.__name__ == 'ZMQInteractiveShell':
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


class Opendat:
    """ Class to get data from opendat

    :param url_catalog: url to catalog
    :param url_netcdf: url to netCDF files"""

    def __init__(self, url_catalog, url_netcdf, user=None, passw=None):
        self.url_catalog = url_catalog
        self.url_netcdf = url_netcdf
        self.user = user
        self.passw = passw

        self.dt_now = None
        self.dt_now_s = ''
        self.date_now()

        self.files_avbl = []
        self.files_avbl_find = []

        self.ds = None

    def catalog(self, last=False, date_s='', text_find=None, key_file='@name', struct_data=['catalog', 'dataset', 'dataset']):
        """ Get catalog from url and return files available

        :param last: get last file
        :param date_s: date to get
        :param text_find: list of text to find
        :param key_file: key to get file name
        :return: files available"""

        self.files_avbl = []
        self.files_avbl_find = []
        msg_all = ''

        if type(self.url_catalog) == str:
            self.url_catalog = [self.url_catalog]

        for url in tqdm(self.url_catalog, desc='Getting catalog'):
            # Read url
            _file = urlopen(url)
            data = _file.read()
            _file.close()

            # Parse to dict format (xml) and get files available in catalog (nc)
            data = xmltodict.parse(data)
            files_avbl = data['catalog']['dataset']['dataset']
            files_avbl = [_f for _f in files_avbl if '.nc' in _f[key_file]]
            print(f'Url para conexión: {url}. Ficheros disponibles: {len(files_avbl)}')

            self.files_avbl.append(files_avbl)

            # Get message to show
            msg, files_avbl_find = '', []
            if date_s != '':
                files_avbl_find = [_f for _f in files_avbl if date_s in _f[key_file]]
                msg = f'(Fecha: {date_s}). '
                msg_all = 'Filtro fecha'
            if text_find is not None:
                files_avbl_text = files_avbl_find.copy() if len(files_avbl_find) > 0 else files_avbl.copy()
                for text_find_c in text_find:
                    files_avbl_find = [_f for _f in files_avbl_text if text_find_c in _f[key_file]]
                    files_avbl_text = files_avbl_find.copy()
                    msg += f'(Text: {text_find_c}). '
                msg_all = 'Filtro texto'
            if len(files_avbl_find) == 0 and last:
                files_avbl_find = [files_avbl[-1]]
                msg = f"(Ultima disponible: {files_avbl_find[0]['@name']}). "
                msg_all = 'Filtro ultimo disponible'

            print(f'Url para conexión: {url}. Ficheros disponibles: {len(files_avbl_find)}. {msg}')
            self.files_avbl_find.append(files_avbl_find)

        self.files_avbl = [e for sb in self.files_avbl for e in sb]
        self.files_avbl_find = [e for sb in self.files_avbl_find for e in sb]

        print(f'Ficheros totales disponibles: {len(self.files_avbl)}')
        print(f'Ficheros totales disponibles: {len(self.files_avbl_find)}. {msg_all}')

    def download_nc(self, files_date, key_file, concat, var_save=None, lonlat_target=None, path_save=None, aux_fname=None):
        """ Download netCDF files from url and return data in dataset format

        :param files_date: files to download
        :param key_file: key to get file name
        :param concat: concat vars in one dataset
        :param var_save: save variables in one dataset
        :param lonlat_target: target lon/lat
        :param path_save: path to save files
        :param aux_fname: aux file name
        :return: data in dataset format"""

        ds, ds_all = None, []
        try:
            # Download files and read vars
            for i, file_date_c in enumerate(tqdm(files_date[::-1], total=len(files_date), desc='Downloading files')):
                resp = requests.get(self.url_netcdf + file_date_c[key_file], auth=HTTPBasicAuth(self.user, self.passw))
                if resp.status_code != 200:
                    if resp.status_code == 100:
                        print(f'ValueError: La conexión con usuario y contraseña ha fallado.')
                    else:
                        msg_error = 'ValueError: La conexión ha fallado.'
                    print(msg_error)
                    exit(-4)

                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    temp_file.write(resp.content)

                url = temp_file.name
                resp.close()

                ds = xr.open_dataset(url)
                temp_file.close()

                if var_save is not None:
                    ds = ds[var_save]

                if lonlat_target is not None:
                    lon_target, lat_target = lonlat_target[0], lonlat_target[1]

                    # Cálculo de distancia mínima (hipótesis: lon/lat en 2D con dims x, y)
                    dist = np.sqrt((ds.lon - lon_target) ** 2 + (ds.lat - lat_target) ** 2)
                    min_dist_idx = dist.argmin(dim=["x", "y"])

                    # Extraer el índice exacto
                    x_sel = ds.x[int(min_dist_idx['x'])]
                    y_sel = ds.y[int(min_dist_idx['y'])]

                    ds = ds.sel(x=x_sel, y=y_sel)

                if concat:
                    ds_all.append(ds)
                else:
                    if path_save is not None:
                        ds.to_netcdf(os.path.join(path_save, file_date_c['@name']))
        except Exception as e:
            print(f'ValueError: {e.args[0]}')
            exit(-1)

        # Concat vars in one dataset
        if concat:
            if len(ds_all) > 1:
                self.ds = xr.concat(ds_all, dim='time')
                if path_save is not None:
                    aux_fname = files_date[0]['@name'] + '_' + files_date[-1]['@name'] if aux_fname is None else aux_fname
                    self.ds.to_netcdf(os.path.join(path_save, aux_fname))
            else:
                self.ds = ds_all[0]
        else:
            self.ds = ds

        print('Descarga de datos completada')

    def download_bz2_grib2(self, files_date, folder_pred, _vars):
        """ Download bz2 files from url and return data in dataset format

        :param files_date: files to download
        :param folder_pred: folder to save files
        :param _vars: variables to download
        :return: data in dataset format"""

        ds_all = []

        folder_pred.create()

        try:
            for i, file_date_c in enumerate(files_date):
                file_bz2 = wget.download(file_date_c, folder_pred.path)

                data = bz2.BZ2File(file_bz2).read()
                file_grib2 = file_bz2[:-4]
                open(file_grib2, 'wb').write(data)  # write a uncompressed file

                ds = xr.open_dataset(file_grib2, engine="cfgrib")
                for j, _v in enumerate(_vars):
                    ds_all.append(ds[_v])

                print(f'Descargado {file_date_c} ({i + 1}/{len(files_date)})')
        except Exception as e:
            print(f'ValueError: {e.args[0]}')
            exit(-1)

        ds_out = []
        for j in range(len(_vars)):
            ds_c = [_ds for i, _ds in enumerate(ds_all) if i in np.array([k for k in range(j, len(ds_all), len(_vars))])]
            for i, _ds_c in enumerate(ds_c):
                da_dates = xr.DataArray(data=[dt.datetime.utcfromtimestamp((ds_c[0].time.data + _ds_c.time.step.values).tolist() / 1e9) for _ds_c in ds_c], dims=["time"])
                ds_c[i] = _ds_c.assign_attrs(time_dt=da_dates)
            ds_c = xr.concat(ds_c, dim='time')
            ds_out.append(ds_c)

        folder_pred.remove()
        print('Descarga de datos completada')

        return ds_out

    def date_now(self):
        """ Get date now in format YYYYMMDD_00 or YYYYMMDD_12"""

        # Get date now
        self.dt_now = dt.datetime.now()

        # Get date in format YYYYMMDD
        self.dt_now_s = dt.datetime.strftime(self.dt_now, '%Y%m%d')

        # Get hour and add to date in format YYYYMMDD_00 or YYYYMMDD_12
        if self.dt_now.hour < 17:
            self.dt_now_s += '_00'
        else:
            self.dt_now_s += '_12'

    @staticmethod
    def check_date_format(dt_now_s):
        """ Check if date is correct

        :return: True if date is correct else False"""

        # Check date
        try:
            dt.datetime.strptime(dt_now_s, '%Y%m%d_%H')
        except ValueError:
            return False

        # Get hour and add to date in format YYYYMMDD_00 or YYYYMMDD_12
        if dt_now_s[9:] != '00' and dt_now_s[9:] != '12':
            return False

        return True

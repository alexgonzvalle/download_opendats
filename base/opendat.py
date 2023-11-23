import xarray as xr
import xmltodict
from urllib.request import urlopen

import wget
import bz2
import numpy as np
import datetime as dt


class Opendat:
    """ Class to get data from opendat

    :param url_catalog: url to catalog
    :param url_netcdf: url to netCDF files"""

    def __init__(self, url_catalog, url_netcdf):
        self.url_catalog = url_catalog
        self.url_netcdf = url_netcdf

        self.dt_now = None
        self.dt_now_s = ''
        self.date_now()

        self.files_avbl = []
        self.files_avbl_find = []

        self.ds = None

    def catalog(self, last=False, date_s='', text_find='', key_file='@name'):
        """ Get catalog from url and return files available

        :param last: get last file
        :param date_s: date to get
        :param text_find: text to find
        :param key_file: key to get file name
        :return: files available"""

        # Read url
        _file = urlopen(self.url_catalog)
        data = _file.read()
        _file.close()

        # Parse to dict format (xml) and get files available in catalog (nc)
        data = xmltodict.parse(data)
        self.files_avbl = data['catalog']['dataset']['dataset']
        self.files_avbl = [_f for _f in self.files_avbl if '.nc' in _f['@name']]
        print(f'Url para conexión: {self.url_catalog}. Ficheros disponibles: {len(self.files_avbl)}')

        # Get message to show
        msg, self.files_avbl_find = '', []
        if date_s != '':
            self.files_avbl_find = [_f for _f in self.files_avbl if date_s in _f[key_file]]
            msg = f'(Fecha: {date_s})'
        if text_find != '':
            self.files_avbl_find = [_f for _f in self.files_avbl if text_find in _f[key_file]]
            msg = f'(Fecha: {date_s})'
        if len(self.files_avbl_find) == 0 and last:
            self.files_avbl_find = [self.files_avbl[-1]]
            msg = f"(Ultima disponible: {self.files_avbl_find[0]['@name']})"

        print(f'Url para conexión: {self.url_catalog}. Ficheros disponibles: {len(self.files_avbl_find)}. {msg}')

    def download_nc(self, files_date, key_file, path_save=None):
        """ Download netCDF files from url and return data in dataset format

        :param files_date: files to download
        :param key_file: key to get file name
        :return: data in dataset format"""

        ds_all = []
        try:
            # Download files and read vars
            for i, file_date_c in enumerate(files_date[::-1]):
                ds = xr.open_dataset(self.url_netcdf + file_date_c[key_file])
                ds_all.append(ds)
                print(f"Descargado {file_date_c['@name']} ({i + 1}/{len(files_date)})")
        except Exception as e:
            print(f'ValueError: {e.args[0]}')
            exit(-1)

        # Concat vars in one dataset
        if len(ds_all) > 1:
            self.ds = xr.concat(ds_all, dim='time')
        else:
            self.ds = ds_all[0]

        print('Descarga de datos completada')

        if path_save is not None:
            self.ds.to_netcdf(path_save)

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

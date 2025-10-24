from opendats.opendat import Opendat


class IH(Opendat):
    """ Class to get data from opendat IH

    :param name_catalog: name catalog"""
    
    def __init__(self, name_catalog, user=None, passw=None):
        self.name_catalog = name_catalog
        url_catalog = 'https://ihthredds.ihcantabria.com/thredds/catalog/' + name_catalog + '/catalog.xml'
        url_netcdf = 'https://ihthredds.ihcantabria.com/thredds/fileServer/'
        super().__init__(url_catalog, url_netcdf, user, passw)

    def catalog(self, last=False, date_s='', text_find='', key_file='@name'):
        """ Get catalog from url PdE and return files available

        :param last: get last file
        :param date_s: date to get
        :param text_find: text to find
        :param key_file: key to get file name
        :return: files available"""

        super().catalog(last, date_s, text_find, key_file)
        if len(self.files_avbl) == 0 and len(self.files_avbl_find) == 0:
            print(f'ValueError: No hay datos disponibles de {self.name_catalog} para la fecha {date_s}')
            exit(-2)

    def get(self, files_date=None, path_save=None):
        """ Download netCDF files from url PdE and return data in dataset format

        :param files_date: files to download
        :param path_save: path to save files
        :return: data in dataset format"""

        if files_date is None:
            files_date = []
        if len(files_date) == 0:
            files_date = self.files_avbl_find
        if len(files_date) == 0:
            files_date = self.files_avbl

        super().download_nc(files_date, '@urlPath', path_save=path_save)

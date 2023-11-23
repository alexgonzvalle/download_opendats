from base.opendat import Opendat


class PDE(Opendat):
    """ Class to get data from opendat PdE

    :param name_catalog: name catalog"""

    def __init__(self, name_catalog, YYYY='', MM='', is_hourly=True):
        """ Class to get data from opendat PdE

        :param name_catalog: name catalog
        :param folder_date: folder date. Default: HOURLY or YYYY"""

        if not is_hourly and YYYY == '':
            print('ValueError: YYYY is empty')
            exit(-3)
        if YYYY != '' and MM == '':
            print('ValueError: MM is empty')
            exit(-3)
        if YYYY != '' and MM != '':
            is_hourly = False

        self.name_catalog = name_catalog

        year_folder = 'HOURLY' if is_hourly else YYYY
        month_folder = '' if is_hourly else '/' + MM
        url_catalog = f'http://opendap.puertos.es/thredds/catalog/{name_catalog}/{year_folder}{month_folder}/catalog.xml'

        url_netcdf = 'http://opendap.puertos.es/thredds/dodsC/'

        super().__init__(url_catalog, url_netcdf)

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

        super().download_nc(files_date, '@ID', path_save)

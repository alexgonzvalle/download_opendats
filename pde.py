from base.opendat import Opendat


class PDE(Opendat):
    """ Class to get data from opendat PdE

    :param name_catalog: name catalog"""

    def __init__(self, name_catalog, YYYY=None, MM=None, is_hourly=True, user=None, passw=None):
        """ Class to get data from opendat PdE

        :param name_catalog: name catalog
        :param YYYY: year
        :param MM: month
        :param is_hourly: hourly data"""

        if not is_hourly and YYYY is None:
            print('ValueError: YYYY is empty')
            exit(-3)
        elif YYYY is not None:
            if type(YYYY) is not list:
                print('ValueError: YYYY is not list')
                exit(-3)
            if type(MM) is not list:
                print('ValueError: MM is not list')
                exit(-3)
        if YYYY is not None and MM is None:
            print('ValueError: MM is empty')
            exit(-3)
        elif MM is not None:
            if type(MM) is not list:
                print('ValueError: MM is not list')
                exit(-3)
        if YYYY is not None and MM is not None:
            is_hourly = False

        self.name_catalog = name_catalog

        year_folder = ['HOURLY'] if is_hourly else YYYY
        month_folder = [''] if is_hourly else ['/' + mm for mm in MM]

        url_catalog = []
        for yyyy in year_folder:
            for mm in month_folder:
                url_catalog.append(f'http://opendap.puertos.es/thredds/catalog/{name_catalog}/{yyyy}{mm}/catalog.xml')

        url_netcdf = 'http://opendap.puertos.es/thredds/fileServer/'

        super().__init__(url_catalog, url_netcdf, user, passw)

    def catalog(self, last=False, date_s='', text_find=None, key_file='@name'):
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

    def get(self, files_date=None, concat=False, path_save=None, aux_fname=None):
        """ Download netCDF files from url PdE and return data in dataset format

        :param files_date: files to download
        :param concat: concatenate files
        :param path_save: path to save files
        :param aux_fname: auxiliary file name
        :return: data in dataset format"""

        if files_date is None:
            files_date = []
        if len(files_date) == 0:
            files_date = self.files_avbl_find
        if len(files_date) == 0:
            files_date = self.files_avbl

        super().download_nc(files_date, '@ID', concat, path_save, aux_fname)

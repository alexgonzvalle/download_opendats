from pde import PDE

# pde = PDE('nivmar_large_nivmar', ['2019', '2020', '2021', '2022'], ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'])
pde = PDE('circulation_regional_gib', ['2022'], ['12'])
# user, passw = 'Beatriz_Rodriguez', 'pz3D@*Vvnn'
# pde = PDE('atmosphere_regional_harmonie25/ALBO', ['2019', '2020', '2021', '2022'], ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'], user=user, passw=passw)

pde.catalog(date_s='20221231', text_find=['hm', "HC"])
pde.get(path_save=r'P:\99_PROAS_FASE_2\09_F2E9_1_SWAN_DINAMICO\DATA_CORRIENTE')

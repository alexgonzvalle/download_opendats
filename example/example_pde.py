from pde import PDE

pde = PDE('nivmar_large_nivmar', '2019', '01')
pde.catalog(text_find="FC")
pde.get()
print(pde.ds)

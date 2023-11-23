from ih import IH

ih = IH('noaagfs025/Global')
ih.catalog(last=True)
ih.get()
print(ih.ds)


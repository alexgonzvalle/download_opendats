import logging

from opendats import ih
from opendats import pde

logging.basicConfig(level=logging.INFO)


def main():
    logging.info(ih)
    logging.info(pde)


if __name__ == '__main__':
    logging.debug('>>> Estamos comenzando la ejecución del paquete.')

    main()

    logging.debug('>>> Estamos finalizando la ejecución del paquete.')

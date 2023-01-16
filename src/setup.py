from dataloader import DataLoader
from utils import process_data
import os

#Angabe der Pfade der Annotationen und der Train-, Val- und Test-Aufspaltung
anns_path = os.path.abspath(os.path.join('config', 'annotations', 'annotations.json'))
tvt_path = os.path.abspath(os.path.join( 'config', 'annotations', 'train_val_test_distribution_file.json'))

#Herunterladen und Verarbeiten des Datensatzes
def __main__():
    dataloader = DataLoader(anns_path=anns_path, tvt_path=tvt_path)
    process_data(dataloader)


                    
if __name__ == '__main__':
    __main__()

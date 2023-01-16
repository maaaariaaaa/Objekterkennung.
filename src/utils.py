from PIL import Image
import numpy as np
import os
import shutil

#Angabe des Pfades der Labels
labels_path = os.path.abspath(os.path.join( 'data', 'labels'))

#Erstellt Ordner, falls diese noch nicht existieren
def create_dirs(dirs):
    if isinstance(dirs, list):
        for cur_dir in dirs:
            if not os.path.exists(cur_dir):
                os.makedirs(cur_dir)
                print(f'Directory {cur_dir} created')
    else:
        if not os.path.exists(dirs):
            os.makedirs(dirs)
            print(f'Directory {dirs} created')

#Verarbeitet die Dateien 
def process_data(dataloader):
    train = dataloader.train
    val = dataloader.val
    test = dataloader.test
    create_dirs(os.path.join(labels_path, 'train'))
    create_dirs(os.path.join(labels_path, 'val'))
    create_dirs(os.path.join(labels_path, 'test'))
      
    #Kreiert die Labels in dem für YOLOv5 benötigten Format
    def process_mode(datapoints, mode:str):
        print(len(datapoints))
        for i in range(0, len(datapoints)):
            yolo_data = []
            datapoint = datapoints[i]
            image_w = datapoint.image.width
            image_h = datapoint.image.height
            for j in range(0, len(datapoint.anns)):
                bbox = datapoint.anns[j].bbox
                x = bbox[0]
                y = bbox[1]
                w = bbox[2]
                h = bbox[3]
                x_center = x+w/2
                y_center = y+h/2
                x_center /= image_w
                y_center /= image_h
                w /= image_w
                h /= image_h
                category_id = datapoint.anns[j].category_id
                yolo_data.append([category_id, x_center, y_center, w, h])
            yolo_data = np.array(yolo_data)
            image_name_txt = datapoint.image.file_name.replace("jpg", "txt")
            np.savetxt(os.path.join(labels_path, mode, image_name_txt), yolo_data, fmt=["%d", "%f", "%f", "%f", "%f"])
            #shutil.copy(
            #    os.path.join(img_path, datapoint.image.file_name),
            #    os.path.join(img_path, mode, datapoint.image.file_name)
            #)
            
    process_mode(train, 'train')
    process_mode(val, 'val')
    process_mode(test, 'test')

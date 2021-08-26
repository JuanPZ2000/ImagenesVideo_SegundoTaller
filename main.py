import cv2
import numpy as np
import os
import sys
from matplotlib import pyplot as plt
from thetaFilter import tethaFilter #TethaFilter Respecto a la vertical

if __name__ == '__main__':
    # Se lee la imagen y se pasa a grises
    path = sys.argv[1]
    image_name = sys.argv[2]
    path_file = os.path.join(path, image_name)
    image = cv2.imread(path_file)
    assert image is not None, "No hay ninguna imagen en {}".format(path_file)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Se llaman los filtros
    tethaFilter1 = tethaFilter(image_gray)
    tethaFilter2 = tethaFilter(image_gray)
    tethaFilter3 = tethaFilter(image_gray)
    tethaFilter4 = tethaFilter(image_gray)

    # Se asignan los tetha y los delta tetha
    delta = 5
    tethaFilter1.set_theta(0, delta)
    tethaFilter2.set_theta(45, delta)
    tethaFilter3.set_theta(90, delta)
    tethaFilter4.set_theta(135, delta)

    #Se llaman las imagenes
    img1 = tethaFilter1.filtering()
    img2 = tethaFilter2.filtering()
    img3 = tethaFilter3.filtering()
    img4 = tethaFilter4.filtering()


    # Se muestran las imagenes
    cv2.imshow("Imagen original",image)
    cv2.imshow("tetha {} y delta {}".format(tethaFilter1.tetha, tethaFilter1.deltaTetha), img1)
    cv2.imshow("tetha {} y delta {}".format(tethaFilter2.tetha, tethaFilter2.deltaTetha), img2)
    cv2.imshow("tetha {} y delta {}".format(tethaFilter3.tetha, tethaFilter3.deltaTetha), img3)
    cv2.imshow("tetha {} y delta {}".format(tethaFilter4.tetha, tethaFilter4.deltaTetha), img4)

    #Promedio de las imagenes

    promedio = (img1+img2+img3+img4)/ 4
    cv2.imshow("Imagen promediada", promedio)

    cv2.waitKey(0)


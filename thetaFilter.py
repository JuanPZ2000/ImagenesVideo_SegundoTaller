# tethaFilter
# Juan Pablo Zuluaga C 2021 PUJ Procesamiento de imagenes y video
import cv2
import numpy as np
import os
import sys


class tethaFilter:

    def __init__(self, Ibw):
        self.image = Ibw
        self.tetha = 0
        self.deltaTetha = 0



    def set_theta(self, tetha1, deltaTetha1):
        self.tetha = tetha1
        self.deltaTetha = deltaTetha1

    def filtering(self):
        # fft
        image_gray_fft = np.fft.fft2(self.image)
        image_gray_fft_shift = np.fft.fftshift(image_gray_fft)

        # Pre procesamineto
        num_rows, num_cols = (self.image.shape[0], self.image.shape[1])
        enum_rows = np.linspace(0, num_rows - 1, num_rows)
        enum_cols = np.linspace(0, num_cols - 1, num_cols)
        col_iter, row_iter = np.meshgrid(enum_cols, enum_rows)
        half_size = num_rows / 2 - 1  # Se asume numero de columnas = numero de filas

        # +deltaTetha -deltaTetha
        deltaTethaUp = self.tetha + self.deltaTetha
        deltaTethaDown = self.tetha - self.deltaTetha

        # filter mask
        tetha_mask1 = np.zeros_like(self.image)
        tetha_mask2 = np.zeros_like(self.image)
        tetha_mask3 = np.zeros_like(self.image)
        tetha_mask4 = np.zeros_like(self.image)

        # Se halla el angulo en grados respecto a la vertical
        a = row_iter - half_size
        b = col_iter - half_size
        condicion = 180 / np.pi * np.arctan(np.divide(b, a, out=np.zeros_like(a), where=a != 0))

        # Se pregunta por la primera condicion
        idx1 = condicion > deltaTethaDown
        tetha_mask1[idx1] = 1

        # Se pregunta por la segunda condicion
        idx2 = condicion < deltaTethaUp
        tetha_mask2[idx2] = 1

        # Se declaran estos valores ya que la funcion arctan devuelve valores negativos, es para comparar en este
        # cuadrante
        deltaTethaUpNeg = deltaTethaUp - 180
        deltaTethaDownNeg = deltaTethaDown - 180

        # Se evalua esta condicion en el cuadrante inferior
        idx3 = condicion < deltaTethaUpNeg
        tetha_mask3[idx3] = 1

        # Se evalua la condicion en el cuadrante inferior
        idx4 = condicion > deltaTethaDownNeg
        tetha_mask4[idx4] = 1

        # Se hace el and entre la mascara 1 y 2 para obtener la del primer cuadrante
        mask = cv2.bitwise_and(tetha_mask1, tetha_mask2)

        # Se hace el and entre la mascara 3 y 4 para obtener la del segundo cuadrante
        mask2 = cv2.bitwise_and(tetha_mask3, tetha_mask4)

        # Se aplica un OR entre estas dos mascaras para obtener la final
        maskdefinitiva = cv2.bitwise_or(mask2, mask)

        # Se muestra la respuesta en frecuencia
        cv2.imshow("freq tetha {} y delta {}".format(self.tetha, self.deltaTetha), maskdefinitiva * 255)

        # Filtrando via FFT con la mascara obtenida
        mask = maskdefinitiva
        fft_filtered = image_gray_fft_shift * mask
        image_filtered = np.fft.ifft2(np.fft.fftshift(fft_filtered))
        image_filtered = np.absolute(image_filtered)
        image_filtered /= np.max(image_filtered)

        return image_filtered

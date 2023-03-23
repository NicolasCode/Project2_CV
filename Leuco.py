# INPUT : Cv2 image 
# OUTPUT : El número de leucocitos en la imagen
import cv2
import numpy as np
import matplotlib.pyplot as plt

def Leucos(img, do_erode = False,  plotly = False):

    # Escogemos los espacios de color hsv y rgb
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # aplicamos un filtro gaussiano en la imagen hsv
    hsv_filter = cv2.GaussianBlur(hsv, (9,9), 60)

    H,S,V = cv2.split(hsv_filter)

    if plotly:
        plt.figure(figsize=(4,4))
        plt.imshow(V, cmap='gray')
        plt.title('Componente V en HSV')
        plt.xticks([]), plt.yticks([])
        plt.show()       
    
    # Modificación de contraste
    T = 170
    c = 0.2
    V[V < T] = np.uint8(V[V < T] * c)

    if plotly:
        plt.figure(figsize=(4,4))
        plt.imshow(V, cmap='gray')
        plt.title('Contraste modificado')
        plt.xticks([]), plt.yticks([])
        plt.show()  

    # UMbralización OTSU
    T_opt, OTSU = cv2.threshold(V, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    if plotly:
        plt.figure(figsize=(4,4))
        plt.imshow(OTSU, cmap='gray')
        plt.title('OTSU')
        plt.xticks([]), plt.yticks([])
        plt.show()

    # --- MORFOLOGIA ---
    # apertura
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    openning = cv2.morphologyEx(OTSU, cv2.MORPH_OPEN, kernel, iterations=3)

    # erosión
    if plotly:
        plt.figure(figsize=(4,4))
        plt.imshow(openning, cmap='gray')
        plt.title('Apertura')
        plt.xticks([]), plt.yticks([])
        plt.show()

    if do_erode:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
        erode = cv2.erode(openning, kernel, iterations=5)

        # apertura 
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        openning = cv2.morphologyEx(erode, cv2.MORPH_OPEN, kernel, iterations=5)
        it = 2
    else:
        it = 3
        erode = openning

    # dilatación

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    dilate = cv2.dilate(openning, kernel, iterations=it)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    clausura = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, kernel, iterations=8)

    if plotly:
        plt.figure(figsize=(4,4))
        plt.imshow(clausura, cmap='gray')
        plt.title('Morfologia')
        plt.xticks([]), plt.yticks([])
        plt.show()

    count, markers = cv2.connectedComponents(clausura)

    if plotly:
        plt.figure(figsize=(4,4))
        plt.imshow(markers, cmap='jet')
        plt.title('Objetos')
        plt.xticks([]), plt.yticks([])
        plt.show()

    # Detectar la regiones compartidas
    unknown = cv2.subtract(dilate, erode)

    if plotly:
        plt.figure(figsize=(4,4))
        plt.imshow(unknown, cmap='gray')
        plt.title('Regiones compartidas')
        plt.xticks([]), plt.yticks([])
        plt.show()

    markers = markers + 1 
    markers[unknown==255] = 0 

    if plotly:
        plt.figure(figsize=(16,8))
        plt.subplot(121)
        plt.imshow(markers,  cmap = 'jet', label = count -1 )
        plt.title('Conteo')
        plt.xticks([]), plt.yticks([])

        plt.subplot(122)
        plt.imshow(rgb)
        plt.title('Imagen')
        plt.xticks([]), plt.yticks([])
        plt.show()

    return clausura, count - 1 

    
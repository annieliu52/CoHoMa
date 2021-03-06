import numpy as np
import cv2 as cv



##########PARAMETRES A REMPLIR###########
#Taille de référence en pixel pour un objet de 1 mètre pris à 5mètres
#à regler en fonction de la caméra
taille_ref = 250
#Altitude/distance en metre à l'objet
d=0.5
#Taille de l'objet à detecter en metre
taille = 0.2
#Filtre de couleur, ici le rouge
MIN_H = 170
MAX_H = 179
MIN_S = 100
MAX_S = 255
MIN_V = 90
MAX_V = 255
MIN2_H = 0
MAX2_H= 8
#Tolerance sur la taille de l'objet (>=1), facteur multiplicatif
tolerance=5
#Nombre d'images par seconde qu'on veut traiter
n_fps = 5


#######PARAMETRES CALCULES###############
theta_ref = np.arctan(0.2)
theta_obj = np.arctan(taille/d)

#TAILLE EN PIXEL DE L'OBJET
t_px= taille*taille_ref * (theta_obj/theta_ref)





#########FONCTIONS

#prend une enveloppe convexe et renvoie en retour si la taille de l'enveloppe
#est cohérente avec celle de l'objet cherché
def coherance_taille(hull):
    aire=cv.contourArea(hull)
    if(aire < tolerance*t_px*t_px and aire > t_px*t_px/tolerance ) :
        return True
    return False


#détecte les cubes rouges, renvoie l'image surlignée
# et la position du centre des cubes
def detec_image(img):
    # convertit les couleurs en format HSV
    img_HSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # applique un seuil ici on cherche le rouge

    img_seuil = cv.inRange(img_HSV, (MIN_H, MIN_S, MIN_V), (MAX_H, MAX_S, MAX_V))
    img_seuil_2 = cv.inRange(img_HSV, (MIN2_H, MIN_S, MIN_V), (MAX2_H, MAX_S, MAX_V))
    img_seuil+=img_seuil_2

    seuil_rgb = cv.cvtColor(img_seuil, cv.COLOR_GRAY2BGR)

    # detection de l'image avec le seuil
    contours, hierarchy = cv.findContours(
        img_seuil, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    #liste des centres des cubes
    centres = [];

    if( len(contours)!=0) :

        # recherche du contour avec le plus grand contour
        for i in range(0,len(contours) ):
            if( len(contours[i])!=0):
                hull = cv.convexHull(contours[i])

                if( coherance_taille(hull) ) :
                    #rajoute le centre
                    M = cv.moments(hull)
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    centres.append( [cX,cY] )
                    # Dessine et enregistre l'image en sortie
                    isClosed = True
                    # Blue color in BGR
                    color = (255, 0, 0)
                    # Line thickness of 2 px
                    thickness = 5
                    #img_sortie=cv.drawContours(img, contours[indice_max], 0, (255,255,255), thickness)
                    pts= np.array([hull], np.int32);
                    img = cv.polylines(img, pts, isClosed, color, thickness,cv.LINE_AA)

                    pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)

    return img, seuil_rgb, centres


#######MAIN##########

cap =cv.VideoCapture(0)

w = int(cap.get(3))
h = int(cap.get(4))
fps = int(cap.get(cv.CAP_PROP_FPS))

print("Largeur video : ",w)
print("Hauteur video : ", h)
print("Nombre de fps : ", fps)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

cpt=0
while True:
    ret, frame = cap.read()
    cpt+=1
    if(cpt > fps// n_fps):
        cont, seuil, centres = detec_image(frame)
        #cont contient l'image avec les contours
        #seuil contient l'image à laquelle on a appliqué le filtre
        #centre contient les centres des cubes
        cv.imshow('Input',cont)
        print("Centres des cubes : ", centres)
        cpt=0

    c = cv.waitKey(1)
    if c == 27:
        break

cap.release()
cv.destroyAllWindows()











































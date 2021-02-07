import cv2
from PIL import Image
import numpy as np
import time


#Görüntü içindeki yüzleri tespit eder
yuztanımaf = cv2.CascadeClassifier(r'C:\Users\qpc\AppData\Local\Programs\Python\Python37\haarcascade_frontalface_default.xml')

#maskeyi okur
maske= Image.open(r"C:\Users\qpc\AppData\Local\Programs\Python\Python37\mask.png")

#Maske fonksiyonu(girdi olarak gelen görüntüye maskeyi ekler)
def thuglife_filtresi(görüntü):
    
    #girdiyi gri seviyeye dönüştürür
    gri = cv2.cvtColor(görüntü, cv2.COLOR_BGR2GRAY)

    # gri görüntüden yüzleri algılar
    faces = yuztanımaf.detectMultiScale(gri, 1.15)
    ggörüntü = Image.fromarray(görüntü)

    for (x,y,w,h) in faces:
        #maske yeniden boyutlandırırlır
        ybmaske = maske.resize((w,h),Image.ANTIALIAS)
        offset = (x,y)
        # maskeyi gelen görüntü üzerine uygular
        ggörüntü.paste(ybmaske, offset, mask=ybmaske)

        #maskeyi uyguladığı görüntüyü döndürür.
    return np.asarray(ggörüntü)

# Videoyu yakalar
cam = cv2.VideoCapture(0)

while True:
    # videodan frameler döndürür
    ret, frame = cam.read()

    if ret == True:
    # fonksiyonlara  gelen frameleri gönderir
        cv2.imshow('kamera',thuglife_filtresi(frame))

    # escye basıldığında çıkılmasını sağlar
    if cv2.waitKey(1) == 27:
        break

cam.release()
cv2.destroyAllWindows()

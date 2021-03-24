# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 12:03:25 2019

@author: Rifqi
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob
import os
import time

#Akusisi Citra
#img = cv2.imread('static/img/Capture.png')


def grayscalling(img):
    #Grayscaling
   
    H,W = img.shape[:2]
    gray = np.zeros((H,W), np.uint8)
    
    for i in range (H):
        for j in range (W) :
            gray [i,j] = np.clip(0.07 * img[i,j,0] + 0.72 * 
                 img [i,j,1] + 0.21 * img[i,j,2],0,255)
    cv2.imwrite('static/Grayscale/Grayscale.png',gray)
    return gray

def proses_lpf(gray):
    lpf = np.array([[0.111,0.111,0.111], 
                [0.111,0.111,0.111],
                [0.111,0.111,0.111]])

    hasil_lpf = cv2.filter2D(gray,-1,lpf)
    cv2.imwrite('static/lpf/lpf.png',hasil_lpf)
    return hasil_lpf


#Thresholding
def thresholding(hasil_lpf):
    ret,thresh = cv2.threshold(hasil_lpf,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    cv2.imwrite('static/Threshold/Threshold.png',thresh)
    #cv2.imshow('Thresh',thresh)
    #cv2.waitKey(0)
    return thresh

#Dilasi
def dilation(thresh):
    kernel = np.ones((10,100), np.uint8)
    img_dilation = cv2.dilate(thresh, kernel, iterations=1)
    cv2.imwrite('static/dilasi/Dilasi.png', img_dilation)
    return img_dilation

def segmentasi(thresh, img_dilation):
#cari contours
    ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    #urutkan contours dari yang teratas
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[1])
    l = 0
  
    hasil = ""
    for i, ctr in enumerate(sorted_ctrs):
        
        # lakukan pemotongan baris
        x, y, w, h = cv2.boundingRect(ctr)
        if h > 15 :
            roi = thresh[y:y+h, x:x+w]
            ret,thresh1 = cv2.threshold(roi,0,255,cv2.THRESH_OTSU)
            roi=thresh1
            # penyimpanan pada file direktori sistem
            cv2.imwrite('static/line_seg/line'+str(i)+'.png',roi)
            
            kernel = np.ones((10, 25), np.uint8)
            # lakukan dilasi dengan kernel 10,25
            joined = cv2.dilate(roi, kernel, iterations=1)
            # Cari contour kata
            ctrs_1, hier = cv2.findContours(joined.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
            # urutkan contour kata dari kiri ke kanan dan atas kebawah
            sorted_ctrs_1 = sorted(ctrs_1, key=lambda ctr: cv2.boundingRect(ctr)[0] + cv2.boundingRect(ctr)[1])
                
            for j, ctr_1 in enumerate(sorted_ctrs_1):
            
                # lakukan pemotongan citra pada setiap kata
                x_1, y_1, w_1, h_1 = cv2.boundingRect(ctr_1)
                
                if h_1 > 15 :
                    roi_1 = thresh1[y_1:y_1+h_1, x_1:x_1+w_1]
                    # penyimpanan pada file direktori sistem
                    cv2.imwrite('static/words_seg/line'+str(i)+'word'+str(j)+'.png',roi_1)
                
           
                    #untuk tiap-tiap kata yang ditemukan,  lakukan proses berikut
                    #kernel char (untuk menemukan huruf)
                    kernel = np.ones((2, 1), np.uint8)
                    joined2 = cv2.dilate(roi_1, kernel, iterations=1)
                        # Cari contours
                    ctrs_2, hier = cv2.findContours(joined2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                        # Urutkan contours
                    sorted_ctrs_2 = sorted(ctrs_2, key=lambda ctr: cv2.boundingRect(ctr)[0] + cv2.boundingRect(ctr)[1])
                    
                    for k, ctr_2 in enumerate(sorted_ctrs_2):
                        
                        # Lakukan pemotongan citra pada setiap huruf
                        x_2, y_2, w_2, h_2 = cv2.boundingRect(ctr_2)
                        if h_2 > 15 :
                            roi_2 = roi_1[y_2:y_2 + h_2, x_2:x_2 + w_2]
                            #penyimpanan pada file direktori sistem
                            cv2.imwrite('static/char_seg/line'+str(i)+'words'+str(j)+'char'+str(k)+'.png',roi_2)
                            
                            
                            #Resizing
                            resized = cv2.resize(roi_2,(15,15), interpolation = cv2.INTER_AREA)
                            ret,thresh2 = cv2.threshold(resized,0,255,cv2.THRESH_OTSU)
                            
                            #Hilditch's Algorithm Thinning
                            size = np.size(thresh2)
                            skel = np.zeros(thresh2.shape,np.uint8)
                    
                            element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
                            done = False
                 
                            while( not done):
                                eroded = cv2.erode(thresh2,element)
                                temp = cv2.dilate(eroded,element)
                                temp = cv2.subtract(thresh2,temp)
                                skel = cv2.bitwise_or(skel,temp)
                                thresh2 = eroded.copy()
                 
                                zeros = size - cv2.countNonZero(thresh2)
                                if zeros==size:
                                    done = True
                            
                            cv2.imwrite('static/normal/line'+str(i)+'words'+str(j)+'char'+str(k)+'.png', skel)
                            
                            outputString = Template_Matching (roi_2,skel,l)
                            l = l + 1
                            #Hasil konversi berupa teks untuk setiap citra hasil normalisasi
                            hasil += outputString
            #Pemberian Spasi untuk setiap hasil segmentasi kata    
                hasil += " "
        #Pemberian Enter untuk setiap hasil segmentasi baris
            hasil += '<br>'  
        
           
    print (hasil)
    
    #Menampilkan hasil konversi teks pada antarmuka
    return hasil
        
def Template_Matching (roi_2,skel,l):
    #Inisialisasi Label-Label pada File Direktori data latih
    #Menggunakan semua data latih
    labels = ['A', 'B', 'D', 'E', 'G', 'ē', 'I', 'K', 'Ch', 'L', 'M', 'N', 
           'O', 'P', 'Ph', 'R', 'S', 'T', 'Th', 'U', "ō", 'X', 'Z', 'Psi']
    #Menggunakan Model
    #labels = ['A', 'B', 'D', 'E', 'G', 'H', 'I', 'K', 'Kh', 'L', 'M', 'N', 
    #      'O', 'P', 'Ph', 'R', 'S', 'T', 'Th', 'U', 'W', 'X', 'Z', 'Psi']
    maxValue = 0.1
    
    #Inisialisasi variabel untuk menampung
    Prediksi_text = ""
    
    #Pemanggilan Data Latih pada file direktori
    for label in labels:
        #non model
        files1= glob.glob(os.path.join("../Dataset (trained)/"+label+"/*.png"))
        #Menggunakan Model
        #files1= glob.glob("../Model/"+label+"/*.png")
        test_image=skel
        template_data=[]
        
        for myfile in files1:
            image = cv2.imdecode(np.fromfile(myfile, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            #Penyimpanan sementara data latih untuk selanjutnya dihitung nilai korelasi
            template_data.append(image)
        
        #Panggil satu per satu Citra latih
        for tmp in template_data:
            w, h = tmp.shape[::-1]
            
            #Perhitungan Nilai Korelasi untuk setiap citra uji dan citra latih
            result = cv2.matchTemplate(test_image, tmp, cv2.TM_CCORR_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            #penyaringan nilai korelasi
            threshold = 0.2
            
            #Pencarian nilai korelasi terbesar
            if max_val > threshold :
                if max_val > maxValue:
                        print ('Nilai Korelasi = ',maxValue)
                        print ('character_ke : '+str(l))
                        print ('Huruf = '+label)    
                        maxValue = max_val
                        #penyimpanan sementara label dengan nilai korelasi terbesar
                        letter = label
    #penyimpanan akhir dari setiap label citra latih yang memiliki nilai korelasi 
    #terbesar untuk pada citra uji     
    Prediksi_text += letter 
    return Prediksi_text
    
def prosesAll(img):
    image = cv2.imread(img) 
    start_time = time.time()
    grayscale = grayscalling(image)
    lpf = proses_lpf(grayscale)
    treshold = thresholding(lpf)
    img_dilation = dilation (treshold)  
    segmen = segmentasi(treshold, img_dilation)
    elapsed_time = time.time() - start_time
    print ('Waktu pemrosesan = ' ,elapsed_time)
    return segmen

def delete():
    for z in range (20):
        for x in range (20):
            for c in range (200):
                if os.path.exists('static/normal/line'+str(z)+'words'+str(x)+'char'+str(c)+'.png'):
                    os.remove('static/normal/line'+str(z)+'words'+str(x)+'char'+str(c)+'.png')
                if os.path.exists('static/char_seg/line'+str(z)+'words'+str(x)+'char'+str(c)+'.png'):
                    os.remove('static/char_seg/line'+str(z)+'words'+str(x)+'char'+str(c)+'.png')
                if os.path.exists('static/words_seg/line'+str(z)+'word'+str(x)+'.png'):
                    os.remove('static/words_seg/line'+str(z)+'word'+str(x)+'.png')
                if os.path.exists('static/line_seg/line'+str(z)+'.png'):
                    os.remove('static/line_seg/line'+str(z)+'.png')
            
           

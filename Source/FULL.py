# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 19:56:08 2019

@author: Rifqi
"""

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


img = cv2.imread('static/img/Capture.png')



def grayscalling(img):
    #Grayscaling
    #print(img)
    H,W = img.shape[:2]
    gray = np.zeros((H,W), np.uint8)
    
    for i in range (H):
        for j in range (W) :
            gray [i,j] = np.clip(0.07 * img[i,j,0] + 0.72 * 
                 img [i,j,1] + 0.21 * img[i,j,2],0,255)
            
    #print ("")
    #print ("Grayscale :")
    #print (gray)
    #cv2.imshow('grayscale',gray)
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
    kernel = np.ones((20,100), np.uint8)
    img_dilation = cv2.dilate(thresh, kernel, iterations=1)
    #cv2.imshow('dilated',img_dilation)
    #cv2.waitKey(0)
    return img_dilation

def segmentasi(thresh, img_dilation):
#find contours
    ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    #sort contours
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[1])
    l = 0
    Benar = 0
    d = 0
    hasil = ""
    for i, ctr in enumerate(sorted_ctrs):
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(ctr)
    
        # Getting ROI
        roi = thresh[y:y+h, x:x+w]
        ret,thresh1 = cv2.threshold(roi,0,255,cv2.THRESH_OTSU)
        roi=thresh1
        cv2.imwrite('static/line_seg/line'+str(i)+'.png',roi)

        d+= 1
        
        
        kernel = np.ones((2, 1), np.uint8)
        joined = cv2.dilate(roi, kernel, iterations=1)
        # find contours
        ctrs_2, hier = cv2.findContours(joined.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
        # sort contours
        sorted_ctrs_2 = sorted(ctrs_2, key=lambda ctr: cv2.boundingRect(ctr)[0])
            
        
        for k, ctr_2 in enumerate(sorted_ctrs_2):
            
            # Get bounding box
            x_2, y_2, w_2, h_2 = cv2.boundingRect(ctr_2)
            
                # Getting ROI
            roi_2 = roi[y_2:y_2 + h_2, x_2:x_2 + w_2]
            cv2.imwrite('static/char_seg/line'+str(i)+'char'+str(k)+'.png',roi_2)
            
            
                # #   show ROI
            resized = cv2.resize(roi_2,(15,15), interpolation = cv2.INTER_AREA)
            ret,thresh2 = cv2.threshold(resized,0,255,cv2.THRESH_OTSU)
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
            
            cv2.imwrite('static/normal/line'+str(i)+'char'+str(k)+'.png', skel)
            #cv2.imwrite('image/segmentasi/line'+str(i)+'word'++'char'+str(k)+'.jpg', roi_2)
                
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            
            outputString = Template_Matching (roi_2,skel,l)
            l = l + 1
            #PredBenar = Template_Matching (roi_2,skel,l)
            #Benar = PredBenar + Benar
            #print('Benar')
            #print (Benar)
            hasil += outputString
            
    
        hasil += "\n"   
    print (hasil+ "\n")
    Accuracy = (Benar/l) * 100
    print (Accuracy) 
    return hasil
        
def Template_Matching (roi_2,skel,l):
    labels = ['A', 'B', 'D', 'E', 'G', 'H', 'I', 'K', 'Kh', 'L', 'M', 'N', 
           'O', 'P', 'Ph', 'R', 'S', 'T', 'Th', 'U', 'W', 'X', 'Z', 'Psi']
    maxValue = 0.1
    
    Prediksi_text = ""
    Prediksi_Benar = 0
    for label in labels:
        files1= glob.glob("../Dataset (trained)/"+label+"/*.png")
        test_image=skel
        template_data=[]
        
        for myfile in files1:
            image = cv2.imread(myfile,0)
            template_data.append(image)
            
        for tmp in template_data:
            w, h = tmp.shape[::-1]
            
            result = cv2.matchTemplate(test_image, tmp, cv2.TM_CCORR_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            threshold = 0.2
            if max_val > threshold :
                if max_val > maxValue:
                        print ('Nilai Korelasi = ',maxValue)
                        print ('character_ke : '+str(l))
                        print ('Huruf = '+label)    
                        maxValue = max_val
                        
                        letter = label
               
    Prediksi_text += letter
   
    
                
            
                
            
                    
            #loc = np.where( result >= threshold)
           # for pt in zip(*loc[::-1]):
                #cv2.rectangle(test_image, pt, (pt[0] + w , pt[1] + h), (255,0,0), 2)
                #Prediksi_Benar = Prediksi_Benar + 1
                #print ('Prediksi_Benar')
                #print (Prediksi_Benar)
                #Prediksi_text += label
                #print ('character_ke : '+str(l))
                #print ('Huruf = '+label)
    
    #cv2.imshow('Result',test_image)    
    return Prediksi_text
    


grayscale = grayscalling(img)
lpf = proses_lpf(grayscale)
treshold = thresholding(lpf)
img_dilation = dilation (treshold)  
segmen = segmentasi(treshold, img_dilation)


def delete():
    for z in range (100):
        for x in range (100):
            if os.path.exists('static/normal/line'+str(z)+'char'+str(x)+'.png'):
                os.remove('static/normal/line'+str(z)+'char'+str(x)+'.png')
            if os.path.exists('static/char_seg/line'+str(z)+'char'+str(x)+'.png'):
                os.remove('static/char_seg/line'+str(z)+'char'+str(x)+'.png')
            if os.path.exists('static/line_seg/line'+str(z)+'.png'):
                os.remove('static/line_seg/line'+str(z)+'.png')
            
           
#grayscale = grayscalling()
#treshold = thresholding(grayscale)
#img_dilation = dilation (treshold)  
#segmen = segmentasi(treshold, img_dilation)
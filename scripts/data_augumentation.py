import cv2
import numpy as np
import os
import random
from data_aug import *
from bbox_util import *
import random
import hashlib 
import sys

def downsample(img,size=0):
    width, height, _ = img.shape
    if(size == 0):
        size = height/5
    r = size / img.shape[1]
    dim = (int(img.shape[0] * r), int(img.shape[1] * r))
    w, h = (int(width/size), int(height/size))

    temp = cv2.resize(img, dim, interpolation=cv2.INTER_LINEAR)

    output = cv2.resize(temp, (height,width), interpolation=cv2.INTER_NEAREST)
    return output


def noisy(image,noise_typ="s&p"):
    if noise_typ == "s&p":
        row,col,ch = image.shape
        s_vs_p = 0.5
        amount = 0.1
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
        out[tuple(coords)] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
        out[tuple(coords)] = 0
        return out

    elif noise_typ =="speckle":
        row,col,ch = image.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)        
        noisy = image + image * gauss
        return noisy

def readFile(adress):
    img = cv2.imread("gun/"+adress+".JPEG")
    file = open("Labels/"+adress+".txt","r")
    lines = file.readlines()
    file.close()

    for i in range(len(lines)):
        lines[i] = lines[i].strip("\n")

    bboxes = np.zeros(shape=(int(lines[0]),4))

    for i in range(1,len(lines)):
        b = np.fromstring(lines[i],sep=" ")
        bboxes[i-1] = b
    
    return img,bboxes


def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[2])/2.0
    y = (box[1] + box[3])/2.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)


def save_labels(name,bboxes,width=0,height=0):
    arq = open(name+".txt","w")
    if width != 0 or height != 0:
        for i in bboxes:
            x,y,w,h = convert([width,height],i)
            if x < 0:
                x = 0
            if y < 0:
                y = 0
            if w > width:
                w = 1
            if h > height:
                h = 1

            s = ("0 %f %f %f %f" % (x,y,w,h))
            arq.write(s)
            arq.write("\n")
    arq.close()

def augment(names):
    img,bboxes = readFile(names[:-5])
    h,w,_ = img.shape
    img_d = downsample(img)
    img_p = noisy(img,noise_typ="s&p")
    img_k = noisy(img,noise_typ="speckle")

    cv2.imwrite("Gun_aug/"+names[:-5]+"_a.JPEG",img)
    cv2.imwrite("Gun_aug/"+names[:-5]+"_b.JPEG",img_d)
    cv2.imwrite("Gun_aug/"+names[:-5]+"_c.JPEG",img_p)
    cv2.imwrite("Gun_aug/"+names[:-5]+"_d.JPEG",img_k)
    save_labels("Gun_aug_Labels/"+names[:-5]+"_a",bboxes,w,h)
    save_labels("Gun_aug_Labels/"+names[:-5]+"_b",bboxes,w,h)
    save_labels("Gun_aug_Labels/"+names[:-5]+"_c",bboxes,w,h)
    save_labels("Gun_aug_Labels/"+names[:-5]+"_d",bboxes,w,h)

    img_flip,bboxes_flip = RandomHorizontalFlip(1)(img,bboxes)
    img_d = downsample(img_flip)
    img_p = noisy(img_flip,noise_typ="s&p")
    img_k = noisy(img_flip,noise_typ="speckle")
    
    cv2.imwrite("Gun_aug/"+names[:-5]+"_e.JPEG",img_flip)
    cv2.imwrite("Gun_aug/"+names[:-5]+"_f.JPEG",img_d)
    cv2.imwrite("Gun_aug/"+names[:-5]+"_g.JPEG",img_p)
    cv2.imwrite("Gun_aug/"+names[:-5]+"_h.JPEG",img_k)
    save_labels("Gun_aug_Labels/"+names[:-5]+"_e",bboxes_flip,w,h)
    save_labels("Gun_aug_Labels/"+names[:-5]+"_f",bboxes_flip,w,h)
    save_labels("Gun_aug_Labels/"+names[:-5]+"_g",bboxes_flip,w,h)
    save_labels("Gun_aug_Labels/"+names[:-5]+"_h",bboxes_flip,w,h)

def augment_neg(names):
    bboxes = []

    img = cv2.imread("VOCdevkit/VOC2012/JPEGImages/"+names)
    try:
        img_name = getHash(img)
    except:
        print(names)
    

    img_d = downsample(img)
    img_p = noisy(img,noise_typ="s&p")
    img_k = noisy(img,noise_typ="speckle")

    cv2.imwrite("negative_aug/"+img_name+"_a.JPEG",img)
    cv2.imwrite("negative_aug/"+img_name+"_b.JPEG",img_d)
    cv2.imwrite("negative_aug/"+img_name+"_c.JPEG",img_p)
    cv2.imwrite("negative_aug/"+img_name+"_d.JPEG",img_k)
    save_labels("negative_aug_Labels/"+img_name+"_a",bboxes)
    save_labels("negative_aug_Labels/"+img_name+"_b",bboxes)
    save_labels("negative_aug_Labels/"+img_name+"_c",bboxes)
    save_labels("negative_aug_Labels/"+img_name+"_d",bboxes)


    img_flip = cv2.flip(img, 1)
    img_d = downsample(img_flip)
    img_p = noisy(img_flip,noise_typ="s&p")
    img_k = noisy(img_flip,noise_typ="speckle")
    cv2.imwrite("negative_aug/"+img_name+"_e.JPEG",img_flip)
    cv2.imwrite("negative_aug/"+img_name+"_f.JPEG",img_d)
    cv2.imwrite("negative_aug/"+img_name+"_g.JPEG",img_p)
    cv2.imwrite("negative_aug/"+img_name+"_h.JPEG",img_k)
    save_labels("negative_aug_Labels/"+img_name+"_e",bboxes)
    save_labels("negative_aug_Labels/"+img_name+"_f",bboxes)
    save_labels("negative_aug_Labels/"+img_name+"_g",bboxes)
    save_labels("negative_aug_Labels/"+img_name+"_h",bboxes)

def getHash(img):
    imageHash = hashlib.sha1(img.tostring())
    return imageHash.hexdigest()


def augment_neg_gun(names):
    bboxes = []
    img = cv2.imread("negative/"+names)
    try:
        img_name = getHash(img)
    except:
        print(names)
        #pass

    
    img_d = downsample(img)
    img_p = noisy(img,noise_typ="s&p")
    img_k = noisy(img,noise_typ="speckle")

    cv2.imwrite("negative_gun_aug/"+img_name+"_a.JPEG",img)
    cv2.imwrite("negative_gun_aug/"+img_name+"_b.JPEG",img_d)
    cv2.imwrite("negative_gun_aug/"+img_name+"_c.JPEG",img_p)
    cv2.imwrite("negative_gun_aug/"+img_name+"_d.JPEG",img_k)
    save_labels("negative_gun_aug_Labels/"+img_name+"_a",bboxes)
    save_labels("negative_gun_aug_Labels/"+img_name+"_b",bboxes)
    save_labels("negative_gun_aug_Labels/"+img_name+"_c",bboxes)
    save_labels("negative_gun_aug_Labels/"+img_name+"_d",bboxes)


    img_flip = cv2.flip(img, 1)
    img_d = downsample(img_flip)
    img_p = noisy(img_flip,noise_typ="s&p")
    img_k = noisy(img_flip,noise_typ="speckle")
    cv2.imwrite("negative_gun_aug/"+img_name+"_e.JPEG",img_flip)
    cv2.imwrite("negative_gun_aug/"+img_name+"_f.JPEG",img_d)
    cv2.imwrite("negative_gun_aug/"+img_name+"_g.JPEG",img_p)
    cv2.imwrite("negative_gun_aug/"+img_name+"_h.JPEG",img_k)
    save_labels("negative_gun_aug_Labels/"+img_name+"_e",bboxes)
    save_labels("negative_gun_aug_Labels/"+img_name+"_f",bboxes)
    save_labels("negative_gun_aug_Labels/"+img_name+"_g",bboxes)
    save_labels("negative_gun_aug_Labels/"+img_name+"_h",bboxes)


def gen_labels(names):
    bboxes = []

    img = cv2.imread("VOCdevkit/VOC2012/JPEGImages/"+names)
    try:
        img_name = getHash(img)
    except:
        print(names)

    save_labels("negative_aug_Labels/"+img_name+"_a",bboxes)
    save_labels("negative_aug_Labels/"+img_name+"_b",bboxes)
    save_labels("negative_aug_Labels/"+img_name+"_c",bboxes)
    save_labels("negative_aug_Labels/"+img_name+"_d",bboxes)
    save_labels("negative_aug_Labels/"+img_name+"_e",bboxes)
    save_labels("negative_aug_Labels/"+img_name+"_f",bboxes)
    save_labels("negative_aug_Labels/"+img_name+"_g",bboxes)
    save_labels("negative_aug_Labels/"+img_name+"_h",bboxes)


def main(args):
    args = args.lower()
    
    if args == "gun":
    
        guns = os.listdir("gun")
        count = 1

        print("gun")
        for i in guns:
            print("%d/%d" % (count,len(guns)), end="\r")    
            augment(i)
            count += 1
        print("%d/%d" % (count,len(guns)))

    if args == "negative":
        count = 1
        negative = os.listdir("VOCdevkit/VOC2012/JPEGImages")
        
        print("negative")
        for i in negative:
            print("%d/%d" % (count,len(negative)), end="\r")
            augment_neg(i)
            count += 1
        print("%d/%d" % (count,len(negative)))

    if args == "gen_labels":
        count = 1
        negative = os.listdir("VOCdevkit/VOC2012/JPEGImages")
        
        print("negative")
        for i in negative:
            print("%d/%d" % (count,len(negative)), end="\r")
            #augment_neg(i)
            gen_labels(i)
            count += 1
        print("%d/%d" % (count,len(negative)))        

    if args == "negative_gun":
        count = 1
        negative_gun = os.listdir("negative/")
        
        print("negative gun")
        for i in negative_gun:
            print("%d/%d" % (count,len(negative_gun)), end="\r")
            augment_neg_gun(i)
            count += 1
        print("%d/%d" % (count,len(negative_gun)))

main(sys.argv[1])
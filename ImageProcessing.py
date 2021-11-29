from cv2 import COLOR_BGR2HSV, inRange, getStructuringElement, MORPH_ELLIPSE, erode, bitwise_and, drawContours, THRESH_BINARY_INV, threshold, Canny, dilate, findContours, RETR_TREE, contourArea, CHAIN_APPROX_SIMPLE, cvtColor, COLOR_GRAY2RGB, COLOR_BGR2GRAY, imread, equalizeHist, add, INTER_CUBIC, GaussianBlur, subtract, filter2D, flip, getRotationMatrix2D, warpAffine, imshow, waitKey, destroyAllWindows, resize
from numpy import array, ones, float64, mean, zeros 

def Rotate(image,rotation=90,steps=15):
    images=[]
    for i in range(-rotation,rotation+1,steps):
        rotation_matrix=getRotationMatrix2D(center=(image.shape[1]/2,image.shape[2]/2),
        angle=i, scale=1)
        rotated_image=warpAffine(src=image, M=rotation_matrix, 
        dsize=(image.shape[1], image.shape[0]))
        images.append(rotated_image)
    return array(images)

def AdjustBrightness(image):
    images=[]
    mask=ones(image.shape,dtype='uint8') * 70
    images.append(add(image,mask))
    images.append(subtract(image,mask))
    return images

def FlipImage(image):
    images=[]
    modes=[-1,0,1]
    for i in modes:
        images.append(flip(image,i))
    return images

def Sharpening(image):
    kernel=array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])  # Laplacian Kernel
    image=filter2D(src=image,ddepth=-1,kernel=kernel)
    return image

def Smoothing(image):
    image=GaussianBlur(image,(3,3),0)
    return image

def Resize(image,dimension=(28,28)):
    image=resize(image,dimension,interpolation=INTER_CUBIC)
    return image

    
def Segmentation(image):
    image_gray=cvtColor(image, COLOR_BGR2GRAY)
    _,thresh=threshold(image_gray,mean(image_gray),255,THRESH_BINARY_INV)
    edges=dilate(Canny(thresh,0,255),None)
    cnt=sorted(findContours(edges,RETR_TREE,CHAIN_APPROX_SIMPLE)[-2],key=contourArea)[-1]
    image=drawContours(image,[cnt],-1,(255,255,0),-1)
    return image
 
def SkinDetection(image):
    lower=array([0,40,80],dtype='uint8')
    upper=array([20,255,255],dtype='uint8')
    #image=Resize(image,dimension=(256,256))
    image_hsv=cvtColor(image,COLOR_BGR2HSV)
    skin_mask=inRange(image_hsv,lower,upper)
    kernel=getStructuringElement(MORPH_ELLIPSE,(11,11))
    skin_mask=erode(skin_mask,kernel,iterations=2)
    skin_mask=dilate(skin_mask,kernel,iterations=2)
    skin_mask=Smoothing(skin_mask)
    skin=bitwise_and(image,image,mask=skin_mask)
    return skin

def show_image(image,window_name):
    imshow(window_name, image)
    waitKey(0) 
    destroyAllWindows()

def read_image(path):
    return imread(path,1)

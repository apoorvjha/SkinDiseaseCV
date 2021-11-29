import ImageProcessing as driver
from os import listdir
from random import randint

rotation_range=90
steps=15
dimension=(256,256)

def Augment(source_path):
    images=[]
    for i in listdir(source_path):
        image=driver.read_image(source_path+i)
        image=driver.Resize(image,dimension=dimension)
        skin=driver.SkinDetection(image)
        rotated_images=driver.Rotate(skin,rotation=rotation_range,steps=steps)
        brightness_adjusted_images=driver.AdjustBrightness(skin)
        flipped_images=driver.FlipImage(skin)
        sharp_image=driver.Sharpening(skin)
        smooth_image=driver.Smoothing(skin)
        for j in rotated_images:
            images.append(j)
        for j in brightness_adjusted_images:
            images.append(j)
        for j in flipped_images:
            images.append(j)
        images.append(sharp_image)
        images.append(smooth_image)
    return driver.array(images)

images=Augment("./static/Dataset/Acne/")
driver.show_image(images[randint(0,images.shape[0])], "")





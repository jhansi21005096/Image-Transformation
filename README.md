# Image-Transformation
## Aim
To perform image transformation such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping using OpenCV and Python.

## Software Required:
Anaconda - Python 3.7

## Algorithm:

### Step1:
Import the required libraries and read the original image.

### Step2:
Translate the image.

### Step3:
Scale the image.

### Step4:
Shear the image.

### Step5:
Find reflection of image.

### Step 6:

Rotate the image.

### Step 7:

Crop the image.

### Step 8:

Display all the Transformed images.



## Program:
```python
Developed By:K.Jhansi

Register Number:212221230045

##i)Image Translation

import numpy as np

import cv2

import matplotlib.pyplot as plt

image = cv2.imread('image01.jpg')

image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

plt.axis('off')

plt.imshow(image)

plt.show()

rows,cols,dim = image.shape

M =np.float32([[1, 0, 100],

               [0, 1, 300],
               
               [0, 0, 1]])
               
translated_image = cv2.warpPerspective(image,M,(cols,rows))

plt.axis('off')

plt.imshow(translated_image)

plt.show()



## ii) Image Scaling

import numpy as np

import cv2

import matplotlib.pyplot as plt

image = cv2.imread('image01.jpg')

image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

plt.axis('off')

plt.imshow(image)

plt.show()

rows,cols,dim = image.shape

M =np.float32([[1.5, 0, 0],

               [0, 1.8, 0],
               
               [0, 0, 1]])
               
scaled_image = cv2.warpPerspective(image,M,(cols*2,rows*2))

plt.axis('off')

plt.imshow(scaled_image)

plt.show()



## iii)Image shearing

import numpy as np

import cv2

import matplotlib.pyplot as plt

image = cv2.imread('image01.jpg')

image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

plt.axis('off')

plt.imshow(image)

plt.show()

rows,cols,dim = image.shape

Mx =np.float32([[1, 0.5, 0],

               [0, 1, 0],
               
               [0, 0, 1]])

My =np.float32([[1, 0, 0],

               [0.5, 1, 0],
               
               [0, 0, 1]])
               
shearedx_image = cv2.warpPerspective(image,Mx,(int(cols*1.5),int(rows*1.5)))

shearedy_image = cv2.warpPerspective(image,My,(int(cols*1.5),int(rows*1.5)))

plt.axis('off')

plt.imshow(shearedx_image)

plt.show()

plt.imshow(shearedy_image)

plt.show()



## iv)Image Reflection

import numpy as np

import cv2

import matplotlib.pyplot as plt

image = cv2.imread('image01.jpg')

image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

plt.axis('off')

plt.imshow(image)

plt.show()

rows,cols,dim = image.shape

Mx =np.float32([[1, 0, 0],

               [0, -1, rows],
               
               [0, 0, 1]])

My =np.float32([[-1, 0, cols],

               [0, 1, 0],
               
               [0, 0, 1]])
               
reflectedx_image = cv2.warpPerspective(image,Mx,((cols),(rows)))

reflectedy_image = cv2.warpPerspective(image,My,((cols),(rows)))'

plt.axis('off')

plt.imshow(reflectedx_image)

plt.show()

plt.imshow(reflectedy_image)

plt.show()



## v)Image Rotation

import numpy as np

import cv2

import matplotlib.pyplot as plt

image = cv2.imread('image01.jpg')

image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

plt.axis('off')

plt.imshow(image)

plt.show()

rows,cols,dim = image.shape

angle = np.radians(50)

M =np.float32([[np.cos(angle), -(np.sin(angle)), 0],

               [np.sin(angle), np.cos(angle), 0],
               
               [0, 0, 1]])
               
rotated_image = cv2.warpPerspective(image,M,(int(cols),int(rows)))

plt.axis('off')

plt.imshow(rotated_image)

plt.show()



## vi)Image Cropping

import numpy as np

import cv2

import matplotlib.pyplot as plt

image = cv2.imread('image01.jpg')

image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

plt.axis('off')

cropped_image = image[100:400,100:400]

plt.imshow(cropped_image)

plt.show()




```
## Output:
### i)Image Translation

![output](https://github.com/jhansi21005096/Image-Transformation/blob/main/output1.png)

### ii) Image Scaling

![output](https://github.com/jhansi21005096/Image-Transformation/blob/main/output2.png)


### iii)Image shearing

![output](https://github.com/jhansi21005096/Image-Transformation/blob/main/output3.png)


### iv)Image Reflection

![output](https://github.com/jhansi21005096/Image-Transformation/blob/main/output4.png)

### v)Image Rotation

![output](https://github.com/jhansi21005096/Image-Transformation/blob/main/output5.png)


### vi)Image Cropping

![output](https://github.com/jhansi21005096/Image-Transformation/blob/main/output6.png)

## Result: 

Thus the different image transformations such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping are done using OpenCV and python programming.

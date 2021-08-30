# HOG-BGR-with-visualization
HOG feature descriptor, a kind of feature transform before we put our image into SVM or NN.  
  
In this API, I split hog feature into BGR channel respectively, it's a little different from hog in skimage, but this API still support grayscale image.  
This API divides orients of gradient into 9 bins and each bin represent 0 20 40 60 ... 160.  
  
Furthermore, due to hog's performance has some relationship with tuning parameters, so this repository provides hog visualization both before and after doing block normalization. 
  
## Environment
Python>=3.9.1  
numpy>=1.20.0  
opencv-python>=4.5.1.48  

## Installation  
```
pip install hogBGR  
```

## Demo
Original Picture  
<img src="https://github.com/ycc789741ycc/HOG-BGR-with-visualization/blob/master/hogBGR/example/rose.png" alt="Cover" width="50%"/>  

```
import cv2
from hogBGR import HOG
from hogBGR import HOGvisualized
from hogBGR import util
```

### HOG  
> Let's create a 16x16 (pixels) cell, 2x2 (cells) block hog image.  
```
img = cv2.imread(r'hogBGR/example/rose.png')
cell_hogfeature = HOG.Cell_HOG(img,cell_size=(16,16))
hog = HOG.BlockNorm_HOG(cell_hogfeature,block_size=(2,2))
```
or
```
img = cv2.imread(r'hogBGR/example/rose.png')
hog = HOG.HOGfeature(img,cell_size=(16,16),block_size=(2,2),flatten=False)
```
> Take a visualization
```
res = HOGvisualized.visualize_HOG(hog[:,:,:,0,0,:],15)
cv2.imshow("HOG feature with block normaliztion",res)
```
or  
```
util.BlockNormaliztionHOG_Show(img,one_cell=True)
```
> Below shows hog image in each cell of blocks. (First method would only show first cell of blocks)  
<img src=https://github.com/ycc789741ycc/HOG-BGR-with-visualization/blob/master/READMEpics/blocknorm.png alt="Cover" width="200%"/>  

### HOG without block normalization
> Let's create a 16x16 (pixels) cell hog image.  
```
img = cv2.imread(r'hogBGR/example/rose.png')
cell_hogfeature = HOG.Cell_HOG(img,cell_size=(16,16))
```
> Take a visualization
```
res = HOGvisualized.visualize_HOG(cell_hogfeature,15)
cv2.imshow("part1 - HOG feature without block normaliztion",res)
```
or
```
util.NonBlockNormaliztionHOG_Show(img)
```
> Below shows hog image before block normalization
<img src=https://github.com/ycc789741ycc/HOG-BGR-with-visualization/blob/master/READMEpics/non-blocknorm.png alt="Cover" width="50%"/>  

### Gray scale input image is also supported
Orginal Picture  
<img src=https://github.com/ycc789741ycc/HOG-BGR-with-visualization/blob/master/READMEpics/grayrose.png alt="Cover" width="50%"/>  
  
Visualization  
<img src=https://github.com/ycc789741ycc/HOG-BGR-with-visualization/blob/master/READMEpics/grayhog.png alt="Cover" width="100%"/>  
  
### Simplified visualization
> You can try the result with yourself if you want a faster visualization.
```
res = HOGvisualized.Fast_visualize_HOG(hogfeature,7)
```


import cv2
from hogBGR import HOG
from hogBGR import HOGvisualized
from hogBGR import util

def main():
    img = cv2.imread(r'hogBGR/example/rose.png')
    #img = cv2.imread(r'hogBGR/example/coins.png')
    cv2.imshow("Origin",img)
    '''
    util.BlockNormaliztionHOG_Show(img,one_cell=False)
    util.NonBlockNormaliztionHOG_Show(img)
    '''
    cell_hogfeature = HOG.Cell_HOG(img,cell_size=(16,16))
    #res = HOGvisualized.Fast_visualize_HOG(cell_hogfeature,7)
    res = HOGvisualized.visualize_HOG(cell_hogfeature,15)
    cv2.imshow("part1 - HOG feature without block normaliztion",res)

    hog = HOG.BlockNorm_HOG(cell_hogfeature,block_size=(2,2))
    res = HOGvisualized.visualize_HOG(hog[:,:,:,0,0,:],15)
    cv2.imshow("part2 - HOG feature",res)
    
    hog = HOG.HOGfeature(img,cell_size=(16,16),block_size=(2,2),flatten=False)
    res = HOGvisualized.visualize_HOG(hog[:,:,:,0,0,:],15)
    cv2.imshow("part3 - HOG feature (same result with part2)",res)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__" :
    main()

    
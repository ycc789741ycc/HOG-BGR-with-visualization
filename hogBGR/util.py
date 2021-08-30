import cv2
from hogBGR import HOG
from hogBGR import HOGvisualized

def NonBlockNormaliztionHOG_Show(img,cell_size=(8,8)):
    '''
    Parameters
    ----------
    - cell_hist: HOG which returned from Cell_HOG or BlockNorm_HOG
    - cell_size: int
    '''
    cell_hogfeature = HOG.Cell_HOG(img,cell_size)
    Img = HOGvisualized.visualize_HOG(cell_hogfeature,cell_size[0])
    cv2.imshow("No Block Normaliztion HOG",Img)
    cv2.waitKey(0)
    cv2.destroyWindow("No Block Normaliztion HOG")

def BlockNormaliztionHOG_Show(img,cell_size=(8,8),block_size=(2,2),one_cell=True):
    '''
    Parameters
    ----------
    - cell_hist: HOG which returned from Cell_HOG or BlockNorm_HOG
    - cell_size: int

    Return
    ----------
    - hog_image : (M,N,C) ndarray

    '''
    cell_hogfeature = HOG.Cell_HOG(img,cell_size)
    blocknormHOG = HOG.BlockNorm_HOG(cell_hogfeature,block_size)
    if(one_cell):
        Img = HOGvisualized.visualize_HOG(blocknormHOG[:,:,:,0,0,:],cell_size[0])
        cv2.imshow("HOG, first cell in each block",Img)
    else:
        Img=[]
        for i in range(block_size[0]):
            for j in range(block_size[1]):
                Img.append(HOGvisualized.visualize_HOG(blocknormHOG[:,:,:,i,j,:],cell_size[0]))
        for i in range(block_size[0]*block_size[0]):
            cv2.imshow("HOG, cell_{} in each block".format(i),Img[i])
    cv2.waitKey(0)
    cv2.destroyAllWindows()


import cv2
import numpy as np

def _UnsignedOrientedGradients(img):
    dx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=1)
    dy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=1)
    mag, angle = cv2.cartToPolar(dx, -dy, angleInDegrees=True)

    angle = angle%180
    angle_map = None
    if(len(img.shape)==3):
        h,w,c = img.shape
        angle_map = np.zeros((h,w,3,9))
    else:
        h,w = img.shape
        angle_map = np.zeros((h,w,1,9))
        mag = mag[:,:,None]
        angle = angle[:,:,None]
    
    mask = angle>160
    angle_map[:,:,:,0] += (mag*mask)*(1-(180-mask*angle)/20)
    mask = angle<=20
    angle_map[:,:,:,0] += (mag*mask)*(1-mask*angle/20)
    for i in range(1,9):
        mask = np.bitwise_and(angle>(i-1)*20,angle<=(i+1)*20)
        angle_map[:,:,:,i] += (mag*mask)*(1-np.abs((i*20-mask*angle)/20))
    return angle_map

def _HOGinCell(angle_map,cell_size=(8,8)):
    h,w,c,_ = angle_map.shape
    h2 = int(h/cell_size[0])
    w2 = int(w/cell_size[1])
    cell_hist = np.zeros((h2,w2,c,9))
    tmp = np.zeros((h2,w,c,9))
    for i in range(h2):
        tmp[i] = np.sum(angle_map[i*cell_size[0]:(i+1)*cell_size[0]],axis=0)
    for i in range(w2):
        cell_hist[:,i] = np.sum(tmp[:,i*cell_size[1]:(i+1)*cell_size[1]],axis=1)
    return cell_hist

def Cell_HOG(img,cell_size=(8,8)):
    '''
    Parameters
    ----------
    - img: colorful or Gray scale (M,N,C) ndarry
    - cell_size: (int,int)

    Return
    ----------
    - cell_hist: (M/cell_size,N/cell_size,C,9) ndarry

    '''
    angle_map = _UnsignedOrientedGradients(img)
    cell_hist = _HOGinCell(angle_map,cell_size)
    return cell_hist

def BlockNorm_HOG(cell_hist,block_size=(2,2),flatten=False):
    '''
    Perform L2-norm block normalization

    Parameters
    ----------
    - cell_hist: HOG which returned from Cell_HOG or BlockNorm_HOG
    - block_size: (int,int)
    - flaten: boolean, ravel the out to a 1xn vector

    Return
    ------
    out: (n_blocks_row, n_blocks_col, channel, n_cells_row, n_cells_col, n_orient) ndarray
    '''
    eps=1e-5
    h,w,c,b = cell_hist.shape
    h2 = h-block_size[0]+1
    w2 = w-block_size[1]+1
    blockHOG = np.zeros((h2,w2,c,block_size[0],block_size[1],b))
    tmp = np.zeros((h,w2,c,block_size[1],b))
    for i in range(w2):
        for j in range(block_size[1]):
            tmp[:,i,:,j] = cell_hist[:,i+j]
    for i in range(h2):
        for j in range(block_size[0]):
            blockHOG[i,:,:,j] = tmp[i+j]
    blockHOG = blockHOG.reshape((h2,w2,c,-1))
    blocknormHOG = blockHOG/(np.sqrt(np.sum(blockHOG**2,axis=3)+eps**2)[:,:,:,None])
    blocknormHOG = blocknormHOG.reshape((h2,w2,c,block_size[0],block_size[1],b))
    if(flatten==False):      
        return blocknormHOG
    else:
        return np.ravel(blocknormHOG)

def HOGfeature(img,cell_size=(8,8),block_size=(2,2),flatten=False):
    '''
    Perform HOG in 9 orientations

    Parameters
    ----------
    - img: colorful or Gray scale (M,N,C) ndarry
    - cell_size: (int,int)
    - block_size: (int,int)
    - flaten: boolean, ravel the out to a 1xn vector

    Return
    ------
    out: (n_blocks_row, n_blocks_col, channel, n_cells_row, n_cells_col, n_orient) ndarray
    '''
    cell_hist = Cell_HOG(img,cell_size)
    blocknormHOG = BlockNorm_HOG(cell_hist,block_size,flatten)
    return blocknormHOG

import cv2
import numpy as np

def _DrawCell(bin_idx,bin_len=1,cell_size=(7,7)):
    h,w = cell_size
    cell = np.zeros((h,w))
    angle = np.pi*((bin_idx*20+90))/180
    h_len=  bin_len*h*np.sin(angle)
    w_len = bin_len*w*np.cos(angle)
    y1 = round((h-h_len)/2)
    y2 = round((h+h_len)/2)
    x1 = round((w+w_len)/2)
    x2 = round((w-w_len)/2)
    cv2.line(cell,(x1,y1),(x2,y2),1)
    return cell

def visualize_HOG(hist,cell_size=8):
    '''
    Parameters
    ----------
    - hist: HOG which returned from Cell_HOG or BlockNorm_HOG
    - cell_size: int

    Return
    ----------
    - hog_image : (M,N,C) ndarray

    '''
    h,w,c,b = hist.shape
    norm_hist = (255/np.max(hist))*hist
    canvas = None
    canvas = np.zeros((h*cell_size,w*cell_size,c))
    for i in range(h):
        for j in range(w):
            for k in range(b):
                for l in range(c):
                    canvas[i*cell_size:(i+1)*cell_size,j*cell_size:(j+1)*cell_size,l]+=norm_hist[i,j,l,k]*_DrawCell(k,norm_hist[i,j,l,k],(cell_size,cell_size))
    canvas = np.clip(canvas,0,255)         
    canvas = canvas.astype(np.uint8)
    if(c==1):
        canvas = np.reshape(canvas,(cell_size*h,cell_size*w))
    return canvas
    
def Fast_visualize_HOG(hist,cell_size=8):
    '''
    Parameters
    ----------
    - hist: HOG which returned from Cell_HOG or BlockNorm_HOG
    - cell_size: int

    Return
    ----------
    - hog_image : (M,N,C) ndarray

    '''
    h,w,c,b = hist.shape
    cell_ori = np.argmax(hist,axis=3)
    hist = np.max(hist,axis=3)
    norm_hist = (255/np.max(hist))*hist
    canvas = np.zeros((h*cell_size,w*cell_size,c))
    for i in range(h):
        for j in range(w):
            for l in range(c):
                canvas[i*cell_size:(i+1)*cell_size,j*cell_size:(j+1)*cell_size,l] = norm_hist[i,j,l]*_DrawCell(cell_ori[i,j,l],norm_hist[i,j,l],(cell_size,cell_size))
    canvas = np.clip(canvas,0,255)             
    canvas = canvas.astype(np.uint8)
    if(c==1):
        canvas = np.reshape(canvas,(cell_size*h,cell_size*w))
    return canvas


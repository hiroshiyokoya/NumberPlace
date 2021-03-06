import numpy as np
import cv2

def directs(pt):
        return -pt[0]-pt[1],-pt[0]+pt[1],pt[0]-pt[1],pt[0]+pt[1]
    
def get_argmax_directs(pix_arr,i):
    return np.argmax(np.array([ directs(pix) for pix in pix_arr ])[:,i])

def get_corners(pix_arr):
    corners = np.array([ pix_arr[get_argmax_directs(pix_arr,i)] for i in range(4) ])
    return corners

def get_cell(output,i, width, height, bin_threshold=150):
    j, k = i//9, i%9
    cell = output[height*j:height*(j+1),width*k:width*(k+1)]
    cell = 255 - (cell - np.min(cell))/(np.max(cell) - np.min(cell)) * 255
    ret, cell = cv2.threshold(cell, bin_threshold, 255, cv2.THRESH_BINARY)
#     cell = cv2.adaptiveThreshold(cell.astype(np.uint8),255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
#             cv2.THRESH_BINARY,11,2)
#     cell = cv2.morphologyEx(cell, cv2.MORPH_ERODE, (3,3), 1)
    cell = cv2.morphologyEx(cell, cv2.MORPH_OPEN, (5,5), 1)
#     cell = cv2.morphologyEx(cell, cv2.MORPH_CLOSE, (5,5), 1)
    cell = cv2.morphologyEx(cell, cv2.MORPH_DILATE, (3,3), 1)
    return cell

def capture_image(path, scale=0.3, width=28, height=28, bin_threshold=150):
    board_width = width*9
    board_height = height*9

    image = cv2.imread(path,0)
    image_resized = cv2.resize(image, None, fx=scale, fy=scale)

    # Laplacian transform 
    image_lap = cv2.Laplacian(image_resized, cv2.CV_32F, ksize=29)
    lap_med, lap_max = np.median(image_lap.ravel()), np.max(image_lap.ravel())
    image_lap_norm = (image_lap - lap_med)/(lap_max-lap_med)
    image_lap_norm[image_lap_norm<0] = 0
    image_lap_pow = pow(image_lap_norm, 1/2)
    image_lap_pow = (image_lap_pow*255).astype(np.uint8)

    # connected components
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (image_lap_pow>50).astype(np.uint8)
    )
    border_labels = list(
        set(np.unique(labels[0,:])) | \
        set(np.unique(labels[:,0])) | \
        set(np.unique(labels[-1,:])) | \
        set(np.unique(labels[:,-1]))
    )
    full_labels = [ label for label in range(len(stats)) if label not in border_labels ]
    full_label_image = labels.copy()
    for border in border_labels:
        full_label_image[full_label_image==border] = 0

    # detect outer frame
    candidate_idx = np.argmax(stats[full_labels][:,-1])
    candidate = stats[full_labels][candidate_idx]
    candidate_original_idx = np.array(list(range(len(stats))))[full_labels][candidate_idx]
    frame = (full_label_image==candidate_original_idx).astype(np.uint8)*255

    # detect corners of the outer frame
    pix_arr = np.array(np.where(frame>0)).T
    corners = np.array(get_corners(pix_arr))
    
    # ????????????
    ## ????????????4???
    src = np.float32(corners)
    ## ????????????4???
    dst = np.float32([[0,0],[0,board_width],[board_height,0],[board_height,board_width]])
    ## ????????????
    M = cv2.getPerspectiveTransform(src[:,::-1], dst[:,::-1])
    ## ?????????????????????????????????
    output = cv2.warpPerspective(image_resized,M,(board_width, board_height))

    cells = np.array([ get_cell(output,i,width,height,bin_threshold=bin_threshold) for i in range(81) ])
    return cells

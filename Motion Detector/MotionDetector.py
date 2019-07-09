import numpy as np
import cv2 as cv
import math

cam = cv.VideoCapture(0)
ret, prev = cam.read()
prevgray = cv.cvtColor(prev, cv.COLOR_BGR2GRAY)

# As you decrease the step size the number of green points on screen increases
def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    print(lines.shape)
    # blue lines join green to red points
    # cv.polylines(vis, lines, 0, (255, 0, 0))
    # for (x1, y1), (_x2, _y2) in lines:
    #     # Red circles are the destination
    #     cv.circle(vis, (_x2, _y2), 1, (0, 0, 255), -1)
    #     # Green circles are the beginning point
    #     cv.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    (x1, y1), (_x2, _y2) = lines[500]
    # dist = math.sqrt((x1 - _x2)**2 + (y1 - _y2)**2)
    dist_x = x1 - _x2
    dist_y = y1 - _y2
    # if dist > 10:
    #     print("Motion")
    # print(dist)
    if dist_x > 10:
        print("Going left")
        print("motion at x coordinate {}".format(x1))
    if dist_x < -10:
        print("Going right")
        print("motion at x coordinate {}".format(x1))
    cv.circle(vis, (_x2, _y2), 1, (0, 0, 255), -1)
    # Green circles are the beginning point
    cv.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis

def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv.remap(img, flow, None, cv.INTER_LINEAR)
    return res

    
while True:
        ret, img = cam.read()
        img = cv.flip(img , 1)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        flow = cv.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        prevgray = gray
        cv.imshow('flow', draw_flow(gray, flow))
        ch = cv.waitKey(5)
        if ch == 27:
            break

cv.destroyAllWindows()
cam.release()

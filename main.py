import numpy as np
import cv2
from Frame import Frame, matchFrames
from Cloud import Cloud
from Point import Point
import time
import os
from scipy.spatial import distance

np.set_printoptions(suppress=True)

# pixels are int
def ipair(fpair):
    return (int(round(fpair[0])), int(round(fpair[1])))
                
# world to screen
def denormalize(fpair): # (2)
    xyz = np.array([fpair[0], fpair[1], 1])
    uv = np.dot(K, xyz)[:2]
    uv = np.round(uv)
    return ipair(uv)

def triangulate(pose1, pose2, pts1, pts2, mirror):
#    return cv2.triangulatePoints(pose1[:3], pose2[:3], pts1.T, pts2.T).T
    pts4d = np.zeros((pts1.shape[0], 4))
    pose1 = np.linalg.inv(pose1)
    pose2 = np.linalg.inv(pose2)
#    print(pose1)
    for i, (p1, p2) in enumerate(zip(pts1, pts2)):
        A = np.zeros((4,4))
        A[0] = p1[0] * pose1[2] - pose1[0]
        A[1] = p1[1] * pose1[2] - pose1[1]
        A[2] = p2[0] * pose2[2] - pose2[0]
        A[3] = p2[1] * pose2[2] - pose2[1]
        _, _, vt = np.linalg.svd(A)
        # these can all be sign-swapped at x,y,z or w
        pts4d[i] = vt[3]
        # scale w to 1
        pts4d[i] /= pts4d[i][3]
        # try to fix -z instead of z
        if pts4d[i][2] < 0:
            if mirror:
                pts4d[i][:3] *= -1
            else:
                pts4d[i][3] = 0
    return pts4d 

def hamming_distance(a, b):
    r = (1 << np.arange(8))[:,None]
    return np.count_nonzero((np.bitwise_xor(a,b) & r) != 0)

if __name__ == '__main__':
#    cap = cv2.VideoCapture('../videos/1.MP4')
#    cap = cv2.VideoCapture('../videos/5.MP4')
#    cap = cv2.VideoCapture('../videos/test_ohio.mp4')
#    cap = cv2.VideoCapture('../videos/test_drone.mp4')
#    cap = cv2.VideoCapture('../videos/test_countryroad.mp4')
#    cap = cv2.VideoCapture('../videos/test_countryroad_reverse.mp4')
#    cap = cv2.VideoCapture('../videos/test_kitti984.mp4')
#    cap = cv2.VideoCapture('../videos/test_kitti984_reverse.mp4')
    cap = cv2.VideoCapture('../videos/test_freiburgxyz525.mp4')

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    F = 300 # focal distance of camera
    MIRROR = not True # try to recover points mis-triangulated behind the camera
    REVERSE = False # hack for camera moving backwards
    SBP_PX = 5 # max euclidian distance to search by projection
    
    K = np.array([[F, 0, W/2],
                  [0, F, H/2],
                  [0, 0, 1  ]])
    
    c = Cloud()
    while(cap.isOpened()):
        _, img = cap.read()
        
        f_new = Frame(img, K)
        c.addFrame(f_new)
        
        if f_new.id == 0:
            continue
        
        print(f"\n*** frame {f_new.id} ***")
        f_old = c.frames[-2]

        # old frame kp indices, new frame kp indices, pose delta
        idx_old, idx_new, Rt = matchFrames(f_old, f_new, REVERSE)
        f_new.pose = np.dot(Rt, f_old.pose)
        
        for m,i_old in enumerate(idx_old):
            p = f_old.pts[i_old]
            if p is not None:
                i_new = idx_new[m]
                p.addObservation(f_new, i_new)
  
        pts_old = f_old.kps[idx_old]
        pts_new = f_new.kps[idx_new]
        
        count_sbp = 0
        projs = None
        if f_new.id >= 2:
            # optimize pose only
            c.optimize(K, frames=1, fix_points=True)
            Rt = np.dot(f_new.pose, np.linalg.inv(f_old.pose))
        
            cloud_points = np.array([p.pt4d for p in c.points])
            KP = np.dot(K, np.linalg.inv(f_new.pose)[:3])
            projs = np.dot(KP, cloud_points.T).T
            projs = projs[:,:2] / projs[:,2:]

#without SbP f73 10k
#Points [8627 1115  221  110   25   18    7    2    3    1]
#Frames [ 2.   4.6  7.2  9.8 12.4 15.  17.6 20.2 22.8 25.4 28. ]
#with SbP f73 5k
#Points [3997  583  215   82   45   33   16   12    7    3]
#Frames [ 2.   9.2 16.4 23.6 30.8 38.  45.2 52.4 59.6 66.8 74. ]
            # search by projection
#            for i,p in enumerate(c.points):
#                proj = projs[i]
#                q = f_new.kd.query_ball_point(proj, 5) # tree of kpus
#                for m_id in q:
#                    if f_new.pts[m_id] is None:
#                        dist = cv2.norm(p.orb(), f_new.des[m_id], cv2.NORM_HAMMING)
#                        if dist < 32:
#                            p.addObservation(f_new, m_id)
#                            count_sbp += 1
        
        # local
        pts4d = triangulate(np.eye(4), Rt, pts_old, pts_new, MIRROR)
        if len(pts4d) == 0:
            print(Rt, pts_old, pts_new)
        # to world
        pts4d = np.dot(f_old.pose, pts4d.T).T
        # only create new Point if seen for the first time
        good_pts4d = np.array([f_new.pts[i] is None for i in idx_new])    
        
        good_pts4d &= (pts4d[:,3] != 0)

        count = 0
        for i,p in enumerate(pts4d):
            if not good_pts4d[i]:
                continue

            u, v = ipair(f_new.kpus[idx_new[i]])
            color = np.flip(np.array(img[v][u], dtype=np.float))
            pt = Point(c, p, color)
            pt.addObservation(f_old, idx_old[i])
            pt.addObservation(f_new, idx_new[i])
            count += 1
            
        print(f"Adding: {sum(good_pts4d)}, By projection: {count_sbp}")
        
        if f_new.id >= 4:
            c.optimize(K, frames=10)
#            exit(0)

        frames_per_point = [len(p.frames) for p in c.points]
        hist, bins = np.histogram(frames_per_point, 10)
            
        print("Stats")
        print("Points", hist)
        print("Frames", bins)
        c.show()
        
        for i_old, i_new in zip(idx_old, idx_new):
            kpu_old = f_old.kpus[i_old]
            kpu_new = f_new.kpus[i_new]
            if f_new.pts[i_new] is not None:
                cv2.circle(img, ipair(kpu_new), color=(0,255,0), radius=5)
                cv2.line(img, ipair(kpu_new), ipair(kpu_old), color=(0,255,0))
            else:
                cv2.circle(img, ipair(kpu_new), color=(0,0,255), radius=5)
#        if projs is not None:
#            for proj in projs:
#                cv2.circle(img, ipair(proj), color=(255,0,0), radius=3)
         
        cv2.imshow('cv2', img)
#        time.sleep(3.5)
        
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            c.close()
            break
    
    cap.release()
    cv2.destroyAllWindows()


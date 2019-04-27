import numpy as np
import cv2
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform
from skimage.transform import EssentialMatrixTransform
from scipy.spatial import KDTree

# screen to world
def normalize(pts, Kinv): # (400, 2)
    count = pts.shape[0]   
    uv1 = np.ones((3, count)) # (3, 400)
    uv1[:-1,:] = pts.T
    xyz = np.dot(Kinv, uv1) # (3, 400)
    xy = xyz.T[:, :2] # (400, 2)
    return xy

def extractRt(E, reverse):  
    U,D,Vt = np.linalg.svd(E)

    if np.linalg.det(U) < 0:
        U *= -1.0
    if np.linalg.det(Vt) < 0:
        Vt *= -1.0

    #Find R and T from Hartley & Zisserman
    W = np.mat([[0,-1,0],[1,0,0],[0,0,1]],dtype=float)
    
    R = np.mat(U) * W * np.mat(Vt)
    if (np.sum(R.diagonal()) < 0):
        R = np.mat(U) * W.T * np.mat(Vt)
    
    t = U[:,2]
    if reverse:
        t *= -1
    
    Rt = np.eye(4,4)
    Rt[:3,:3] = R
    Rt[:3, 3] = t

    return Rt

def matchFrames(old, new, reverse=False):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(new.des, old.des, k=2)
    
    idx_old = []
    idx_new = []
    pts_old = []
    pts_new = []
    for m,n in matches: # 2 top matches
        if m.distance < 0.75*n.distance: # only if best match is a lot better
            point_old = old.kps[m.trainIdx]
            point_new = new.kps[m.queryIdx]
            if m.distance < 32 and np.linalg.norm(point_new-point_old) < 0.1: # 10% of screen
                if (m.trainIdx) not in idx_old and (m.queryIdx) not in idx_new:
                    idx_old.append(m.trainIdx)
                    idx_new.append(m.queryIdx)
                    pts_old.append(point_old)
                    pts_new.append(point_new)
    
    model, inliers = ransac((np.array(pts_new), np.array(pts_old)),
#                           FundamentalMatrixTransform,
                            EssentialMatrixTransform,
                            min_samples=8,
                            residual_threshold=0.001,
                            max_trials=50)
    
    print(f"descriptors {len(new.des)} -> matches {len(idx_old)} -> inliers {sum(inliers)}")
    
    # match id     0     1     2  ...
    # idx_old    [123 , 666 ,      ]
    # idx_new    [127 , 11 ,       ] 
    idx_old = np.array(idx_old)
    idx_new = np.array(idx_new)
    return idx_old[inliers], idx_new[inliers], extractRt(model.params, reverse)

class Frame():
    def __init__(self, img, K):
#        self.diag = np.linalg.norm(img.shape[:2])
        self.pose = np.eye(4)
#        self.pose[2,3] = 100
        self.orb = cv2.ORB_create()
        
        # detection
        img_gray = np.mean(img, axis=2).astype(np.uint8)
        feats = cv2.goodFeaturesToTrack(img_gray, maxCorners=1000, qualityLevel=0.01, minDistance=7)
        
        # extraction
        self.kpus = [cv2.KeyPoint(x=f[0][0], y=f[0][1], _size=20) for f in feats]
        self.kpus, self.des = self.orb.compute(img, self.kpus)
        
        # screen points
        self.kpus = np.array([kp.pt for kp in self.kpus])
        
        # for nearest kpu lookup
        self.kd = KDTree(self.kpus)

        # normalized points
        Kinv = np.linalg.inv(K)
        self.kps = normalize(self.kpus, Kinv)
        
        # array of Point
        self.pts = [None] * len(self.kpus)
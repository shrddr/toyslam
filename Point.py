import numpy as np

class Point():
    def __init__(self, cloud, pt4d, color):
        self.pt4d = pt4d
        self.pt3d = pt4d[:3]
        self.frames = []
        self.color = color
        
        self.id = cloud.points_newid
        cloud.points_newid += 1
            
        cloud.points.append(self)
        
    def orb(self):
        f = self.frames[-1]
        return f.des[f.pts.index(self)]
        
    def delete(self):
        for f in self.frames:
            i = f.pts.index(self)
            f.pts[i] = None
        del self

    def addObservation(self, frame, index):
        if frame.pts[index] is None:
            self.frames.append(frame)
            frame.pts[index] = self
            self.orb()
        else:
            print(f"frame {frame.id} kp {index} has P{frame.pts[index].id} - new P{self.id}")
        
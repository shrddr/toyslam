import OpenGL.GL as gl
import pangolin

import numpy as np

from multiprocessing import Process, Queue

import g2o

def draw_axes():
    gl.glPointSize(1)
    points = np.zeros((300, 3))
    points[:100, 0] = np.arange(100)
    points[100:200, 1] = np.arange(100)
    points[200:300, 2] = np.arange(100)
    colors = np.zeros((300, 3)) 
    colors[:100, 0] = 1
    colors[100:200, 1] = 1
    colors[200:300, 2] = 1
    pangolin.DrawPoints(points, colors)    

def work(q, qclose, w=960, h=540):
    pangolin.CreateWindowAndBind('pangolin', w, h)
    gl.glEnable(gl.GL_DEPTH_TEST)

    # Define Projection and initial ModelView matrix
    scam = pangolin.OpenGlRenderState(
        pangolin.ProjectionMatrix(w, h, 420, 420, w//2, h//2, 0.2, 10000),
        pangolin.ModelViewLookAt(-2, -2, -8, 0, 0, 0, pangolin.AxisDirection.AxisNegY))
    handler = pangolin.Handler3D(scam)

    # Create Interactive View in window
    dcam = pangolin.CreateDisplay()
    dcam.SetBounds(0.0, 1.0, 0.0, 1.0, -w/h)
    dcam.SetHandler(handler)
    
    pose = np.eye(4)
    opath = np.zeros((0,3))
    opts = np.zeros((2,3))
    colors = np.zeros((2,3))
    
    while not pangolin.ShouldQuit():
        if not qclose.empty():
            if qclose.get():
                pangolin.Quit()
        
        if not q.empty():
            pose, opath, opts, colors = q.get()
            colors /= 256.0
            
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(0.0, 0.0, 0.0, 1.0)
        dcam.Activate(scam)
    
#        draw_axes()    
        
        # Draw optimized cloud
        gl.glPointSize(2)
        gl.glColor3f(0.5, 0.8, 0.5)
        pangolin.DrawPoints(opts, colors)
        
        # Draw camera
        gl.glLineWidth(1)
        gl.glColor3f(0.4, 0.4, 0.4)
        pangolin.DrawCamera(pose, 10, 1, 1)
    
        # Optimized path
        if len(opath)>2:
            gl.glLineWidth(1)
            gl.glColor3f(0.4, 0.4, 0.4)
            pangolin.DrawLine(np.array(opath))

        pangolin.FinishFrame()
            
class Cloud():
    def __init__(self):
        # data
        self.frames = []
        self.points = []
        self.points_newid = 0
        self.opath = []
        
        # viewer
        self.q = Queue()
        self.qclose = Queue()
        self.p = Process(target=work, args=(self.q, self.qclose), daemon=True)
        self.p.start()
        
    def addFrame(self, f):
        f.id = len(self.frames)
        self.frames.append(f)

    def show(self):
        points = np.array([p.pt3d for p in self.points])
        colors = np.array([p.color for p in self.points])
        self.q.put((self.frames[-1].pose, self.opath, points, colors))
        
    def close(self):
        self.qclose.put(True)
        pass
    
    def optimize(self, K, frames, fix_points=False):
        PT_OFFSET = 10000
        opt = g2o.SparseOptimizer()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        opt.set_algorithm(solver)
        
        rk = g2o.RobustKernelHuber(np.sqrt(5.991))
        
        # only optimize these poses and their Points
        local_frames = self.frames[-frames:]
        
        # a vertex for each camera pose
        for f in self.frames:
            assert(f.id<PT_OFFSET)
            quat = g2o.SE3Quat(f.pose[:3,:3], f.pose[:3,3])
            sbacam = g2o.SBACam(quat)
            sbacam.set_cam(K[0,0], K[1,1], K[0,2], K[1,2], 1.0) # for pixel input
            v = g2o.VertexCam()

            v.set_id(f.id)
            v.set_estimate(sbacam)
            v.set_fixed(f.id < 2 or f not in local_frames)
            opt.add_vertex(v)
        
        # a vertex for each point in cloud
        for p in self.points:
#            if p.frames[-1] not in local_frames:
            if not any([f in local_frames for f in p.frames]):
                continue
            vp = g2o.VertexSBAPointXYZ()
            vp.set_id(PT_OFFSET + p.id)
            vp.set_estimate(p.pt3d)
            vp.set_marginalized(True)
            vp.set_fixed(fix_points)
            opt.add_vertex(vp)
               
            # edge connects every camera with every point
            for f in p.frames:
                edge = g2o.EdgeProjectP2MC()
#                edge = g2o.EdgeSE3ProjectXYZ()
                edge.set_vertex(0, vp)
                edge.set_vertex(1, opt.vertex(f.id))
                idx = f.pts.index(p)
                edge.set_measurement(f.kpus[idx])
                edge.set_information(np.eye(2))
                edge.set_robust_kernel(rk)
                opt.add_edge(edge)
        
#        opt.set_verbose(True)
        opt.initialize_optimization()
        opt.optimize(20)
#        print("chi2", opt.active_chi2())
        
        # get back optimized poses
        self.opath = []
        for f in self.frames:
            f_est = opt.vertex(f.id).estimate()
            R = f_est.rotation().matrix()
            t = f_est.translation()
            Rt = np.eye(4)
            Rt[:3,:3] = R
            Rt[:3, 3] = t.T
#            print(Rt)
            self.opath.append(t)
            f.pose = Rt

        if fix_points:
            return
        
        new_points = []
        # add back or cull
        for p in self.points:
            vp = opt.vertex(PT_OFFSET + p.id)
            if vp is None: # not updated
                new_points.append(p)
                continue
            vp = opt.vertex(PT_OFFSET + p.id)
            p_est = vp.estimate()
            
#            if (len(p.frames) <= 5) and (p.frames[-1] not in local_frames):
#                p.delete()
#                continue
            
            errors = []
            for f in p.frames:
                f_est = opt.vertex(f.id).estimate()
                measured = f.kpus[f.pts.index(p)]
                projected = f_est.w2i.dot(np.hstack([p_est, [1]])) # only works because is VertexCam
                projected = projected[:2] / projected[2] # don't forget to homo
                errors.append(np.linalg.norm(measured-projected))     
            error = np.mean(errors) # mean of squares - over all frames
            if error > 1.0: # px
#                print(f"error {error}, dropping")
                p.delete()
            else:
                p.pt3d = np.array(p_est)
                new_points.append(p)
                
        print("Dropping:", len(self.points)-len(new_points), ", Remaining:", len(new_points))
        self.points = new_points
        
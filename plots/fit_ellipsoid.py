import numpy as np
import os

class Ellipsoid(object):

    def __init__(self, a, b):

        self._a = a
        self._b = b

    @property
    def a(self):
        return self._a

    @property
    def b(self):
        return self._b

    def r(self, theta):
        return (self.a * self.b) / np.sqrt((self.a*np.sin(theta))**2 + (self.b*np.cos(theta))**2)

    def __repr__(self):
        s = f"{self.a}\n"
        s = s + f"{self.b}"
        return s

def initialize_args():
    import argparse
    parser =  argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        dest="infile",
        type=str,
        required=True
    )
    parser.add_argument(
        "-o",
        dest="outfile",
        type=str,
        required=True
    )
    parser.add_argument(
        "-s",
        dest="start",
        type=int,
        required=True
    )
    args = parser.parse_args()
    return args

def cost(pts, rotation, ellipsoid):
    pts = pts.transpose() - np.mean(pts, axis=1)
    pts = np.matmul(rotation.as_matrix(), pts.transpose())
    norms = np.linalg.norm(pts, axis=0)
    thetas = np.arccos(pts[2] / norms)
    rs = ellipsoid.r(thetas)
    return np.mean(np.abs(np.unique(rs - norms)))

def f(params, pts, return_objs=False):
    # from ellipsoid import Ellipsoid
    from scipy.spatial.transform import Rotation as R
    a, b, theta, phi, psi = params
    rotation = R.from_euler("zyx", [theta, phi, psi])
    ellipsoid = Ellipsoid(a, b)
    if return_objs:
        return cost(pts, rotation, ellipsoid), rotation, ellipsoid
    else:
        return cost(pts, rotation, ellipsoid)

def fit_ellipsoid(event, nhit=8, field="filtered"):
    if len(event.total.t) <= nhit:
        return None
    else:
        from scipy.optimize import differential_evolution

        total_x = event[field, "sensor_pos_x"].to_numpy()
        total_y = event[field, "sensor_pos_y"].to_numpy()
        total_z = event[field, "sensor_pos_z"].to_numpy()

        pts = np.array([
            total_x,
            total_y,
            total_z
        ])

        pts = np.unique(pts, axis=1)
        # pts = (pts.transpose() - np.mean(pts, axis=1)).transpose()
        # reduced_f = lambda p: f(p, pts)
        res = differential_evolution(f, [(0, 500),(0, 500),(0, 2*np.pi),(0, 2*np.pi),(0, 2*np.pi),], args=(pts,), workers=len(os.sched_getaffinity(0)), updating='deferred')
        rat = res.x[0] / res.x[1]
        if rat < 1:
            rat = 1 / rat
        return res

if __name__=="__main__":

    args = initialize_args()
    infile = args.infile
    outfile = args.outfile
    start = args.start
    end = args.start + 1000

    import awkward as ak
    events = ak.from_parquet(infile)[start:end]
    rats = []
    for event in events:
        res = fit_ellipsoid(event, nhit=0)
        if res is not None:
            rat = res.x[0] / res.x[1]
            if rat < 1:
                rat = 1 / rat
            rats.append(rat)
    np.save(outfile, rats)

# -*- coding: utf-8 -*-
# vispy: gallery 10
# Copyright (c) Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

""" Demonstrates use of visual.Markers to create a point cloud with a
standard turntable camera to fly around with and a centered 3D Axis.

problem: color must be constant for all points
TODO use threshold to hide low-amp points
"""

import numpy as np
import vispy.scene
from vispy.scene import visuals
# local
import util

# generate data
pos = np.random.normal(size=(100000, 3), scale=0.2)
# one could stop here for the data generation, the rest is just to make the
# data look more interesting. Copied over from magnify.py
centers = np.random.normal(size=(50, 3))
indexes = np.random.normal(size=100000, loc=centers.shape[0] / 2.,
                           scale=centers.shape[0] / 3.)
indexes = np.clip(indexes, 0, centers.shape[0] - 1).astype(int)
scales = 10**(np.linspace(-2, 0.5, centers.shape[0]))[indexes][:, np.newaxis]
pos *= scales
pos += centers[indexes]

print(pos.shape)
print(pos.mean(), pos.min(), pos.max())

dir = '../tmp'
fn = 'out.zip'
data = util.parse_file(dir, fn, 'out.txt')
n = 100000
# pos = np.vstack([data['w'][i][:n] for i in range(len(data['w']))])

a = np.vstack([z[:n, 0] for z in data['z']])
mu = a.mean()
std = a.std()
print(mu, std)
mu = ((a > mu) * a).mean()
print(mu, std)
selection = []
for i, w in enumerate(data['w']):
    indices = data['z'][i][:n, 0] > mu
    # print(data['z'][i][:n, 0].mean())
    # assert(indices.nonzero()[0].size > 0)
    selected = w[:n][indices]
    if selected.size > 0:
        selection.append(selected)

print(len(selection))
assert len(selection) > 0

pos = np.vstack(selection)
# pos = np.vstack((w[:n] for w in data['w']))
for i in range(2):
    pos[:, i] /= pos[:, i].max()
# TODO center around 0,0,0 for intuitive rotation?

#
# Make a canvas and add simple view
#
canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
view = canvas.central_widget.add_view()

# create scatter object and fill in the data
scatter = visuals.Markers()
scatter.set_data(pos, edge_color=None, face_color=(1, 1, 1, .5), size=3,
                 scaling=False)

view.add(scatter)

# view.camera = 'arcball'  # or try 'turntable' 'arcball'
view.camera = 'turntable'  # or try 'turntable' 'arcball'

# add a colored 3D axis for orientation
axis = visuals.XYZAxis(parent=view.scene)

if __name__ == '__main__':
    import sys
    if sys.flags.interactive != 1:
        vispy.app.run()

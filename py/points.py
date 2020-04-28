#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vispy: testskip

import sys
import numpy as np
import zipfile
import os
from vispy import app, gloo
from vispy.visuals.collections import PointCollection
from vispy.visuals.transforms import PanZoomTransform
# local
import util

data = {}
fn = 'tmp/out.zip'
size = os.path.getsize(fn)
print(f'Input file size: {size * 1e-3:0.5f} kB')
if size > 1e6:
    print(f'WARNING, file too large: {size*1e-6:0.4f} MB')

with zipfile.ZipFile(fn) as z:
    # with open(fn, 'rb') as f:
    with z.open('tmp/out.txt', 'r') as f:
        for line in f:
            k, content = line.decode().split(':')
            util.parse_line(data, k, content)

canvas = app.Canvas(size=(800, 600), show=True, keys='interactive')
gloo.set_viewport(0, 0, canvas.size[0], canvas.size[1])
gloo.set_state("translucent", depth_test=False)

panzoom = PanZoomTransform(canvas)

points = PointCollection("agg", color="shared", transform=panzoom)
# points.append(np.random.normal(0.0, 0.5, (10000, 3)), itemsize=5000)
pos = data['v']  # / data['v'][:3].max()
for i in range(3):
    pos[:, i] *= 1 / pos[:, i].max() * 10
    pos[:, i] -= pos[:, i].mean()

# data[:, 0] = 0
# data[:, 1] = 0
pos[:, 2] = 0
print(pos[:3])
colors = data['y'][:, 0]
print(colors.shape)
points.append(pos, colors=colors, itemsize=1)
# points["color"] = (1, 0, 0, 1), (0, 0, 1, 1)
points["color"] = (1, 0, 0, 1)
points.update.connect(canvas.update)


@canvas.connect
def on_draw(event):
    gloo.clear('white')
    points.draw()


@canvas.connect
def on_resize(event):
    width, height = event.size
    gloo.set_viewport(0, 0, width, height)


if __name__ == '__main__' and sys.flags.interactive == 0:
    app.run()

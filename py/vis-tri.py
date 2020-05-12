#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vispy: gallery 50
"""
This example shows how to display 3D objects.
You should see a colored outlined spinning cube.
"""
import zipfile
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from vispy import gloo, app, geometry
from vispy.util.transforms import perspective, translate, rotate
# local
import util


vert = """
// Uniforms
// ------------------------------------
uniform   mat4 u_model;
uniform   mat4 u_view;
uniform   mat4 u_projection;
uniform   vec4 u_color;

// Attributes
// ------------------------------------
attribute vec3 a_position;
attribute vec4 a_color_old;
attribute float a_color;
attribute vec3 a_normal;

// Varying
// ------------------------------------
varying vec4 v_color;

void main()
{
    // r,g,b,alpha
    v_color = vec4(0, a_color, 0, 1) * u_color;
    gl_Position = u_projection * u_view * u_model * vec4(a_position, 1.0);
}
"""


frag = """
// Varying
// ------------------------------------
varying vec4 v_color;

void main()
{
    gl_FragColor = v_color;
}
"""


# -----------------------------------------------------------------------------
def cube(pos=None, c=None, colors=None):
    """
    Build vertices for a colored cube.

    V  is the vertices
    I1 is the indices for a filled cube (use with GL_TRIANGLES)
    I2 is the indices for an outline cube (use with GL_LINES)
    """
    vtype = [('a_position', np.float32, 3),
             ('a_normal', np.float32, 3),
             ('a_color', np.float32, 1),
             ('a_color_old', np.float32, 4)]
    # Vertices positions
    if pos is None:
        pos = np.array([[1, 1, 1], [-1, 1, 1], [-1, -1, 1], [1, -1, 1],
                        [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, -1]])

    # ignore third dim
    delaunay = Delaunay(pos[:, :2], furthest_site=0)
    triangles = delaunay.simplices.flatten().astype(np.uint32)
    print('pos', pos.shape)
    print('tri', triangles.shape, triangles.size / 3)

    # Face Normals
    n = [[0, 0, 1], [1, 0, 0], [0, 1, 0],
         [-1, 0, 1], [0, -1, 0], [0, 0, -1]]
    # Vertice colors
    if colors is None:
        colors = [[0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 1], [0, 1, 0, 1],
                  [1, 1, 0, 1], [1, 1, 1, 1], [1, 0, 1, 1], [1, 0, 0, 1]]

    V = np.array([(pos[i], n[i % len(n)], c[i], colors[i % len(colors)])
                  for i in range(pos.shape[0])], dtype=vtype)

    # I1 is the indices for a filled cube (use with GL_TRIANGLES)
    print("I1 tri faces")
    I1 = np.array([0, 1, 2], dtype=np.uint32)
    I1 = triangles

    I2 = I1

    return V, I1, I2


# -----------------------------------------------------------------------------
class Canvas(app.Canvas):

    def __init__(self, pos=None, colors=None):
        app.Canvas.__init__(self, keys='interactive', size=(600, 600))

        self.vertices, self.filled, self.outline = cube(pos, colors)
        self.filled_buf = gloo.IndexBuffer(self.filled)
        self.outline_buf = gloo.IndexBuffer(self.outline)

        self.program = gloo.Program(vert, frag)
        self.program.bind(gloo.VertexBuffer(self.vertices))

        self.view = translate((0, 0, -5))
        self.model = np.eye(4, dtype=np.float32)

        gloo.set_viewport(0, 0, self.physical_size[0], self.physical_size[1])
        self.projection = perspective(45.0, self.size[0] /
                                      float(self.size[1]), 2.0, 10.0)

        self.program['u_projection'] = self.projection

        self.program['u_model'] = self.model
        self.program['u_view'] = self.view

        self.theta = 0
        self.phi = 0

        gloo.set_clear_color('white')
        gloo.set_state('opaque')
        gloo.set_polygon_offset(1, 1)

        self.on_timer
        self._timer = app.Timer('auto', connect=self.on_timer, start=True)
        self.show()

    # ---------------------------------
    def on_timer(self, event):
        # self.theta += .5
        # self.phi += .5
        self.model = np.dot(rotate(self.theta, (0, 1, 0)),
                            rotate(self.phi, (0, 0, 1)))
        self.program['u_model'] = self.model
        self.update()

    # ---------------------------------
    def on_resize(self, event):
        gloo.set_viewport(0, 0, event.physical_size[0], event.physical_size[1])
        self.projection = perspective(45.0, event.size[0] /
                                      float(event.size[1]), 2.0, 10.0)
        self.program['u_projection'] = self.projection

    # ---------------------------------
    def on_draw(self, event):
        gloo.clear()

        # Filled cube

        gloo.set_state(blend=False, depth_test=True, polygon_offset_fill=True)
        self.program['u_color'] = 1, 1, 1, 1
        self.program.draw('triangles', self.filled_buf)

        # Outline
        gloo.set_state(blend=True, depth_test=True, polygon_offset_fill=False)
        gloo.set_depth_mask(False)
        self.program['u_color'] = 0, 0, 0, 1
        self.program.draw('lines', self.outline_buf)
        gloo.set_depth_mask(True)


# -----------------------------------------------------------------------------
if __name__ == '__main__':
    dir = '../tmp'
    fn = 'out.zip'
    data = util.parse_file(dir, fn, 'out.txt')

    # float64 is not allowed for texture
    pos = data['v'].astype('float32')
    a = data['y'][:, 0].astype('float32')
    phi = data['y'][:, 1].astype('float32')
    # pos[:, 2] = 0
    for i in range(3):
        pos[:, i] *= 1.2 / pos[:, i].max()

    # pos = np.random.normal(0, 0.5, size=(1000, 3)).astype('float32')
    # pos[:, 2] *= pos[:, 2] / 2
    # phi = np.random.normal(0, 0.5, size=(1000, 1)).astype('float32') ** 2
    # print(a.min(), a.max())
    # data is standardized but also normalize (data is range [0,1))
    a = (a - a.min()) / (a.max() - a.min())
    # a -= a.min()
    print('phi pre', phi.min(), phi.max())
    phi = (phi + np.pi) / (2 * np.pi)
    # cyclic coloring
    phi -= np.clip(phi - 0.5, 0, None)

    n = 10000
    indices = np.random.randint(0, pos.shape[0], n)
    # c = Canvas(pos[indices], a[indices])
    c = Canvas(pos[indices], phi[indices])
    # c = Canvas()
    app.run()

    # TODO combine nearby bins?

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vispy: gallery 20

"""
Example demonstrating simulation of fireworks using point sprites.
(adapted from the "OpenGL ES 2.0 Programming Guide")

This example demonstrates a series of explosions that last one second. The
visualization during the explosion is highly optimized using a Vertex Buffer
Object (VBO). After each explosion, vertex data for the next explosion are
calculated, such that each explostion is unique.
"""

import zipfile
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from vispy import gloo, app, geometry
# local
import util

# import vispy
# vispy.use('pyside', 'es2')

VERT_SHADER = """
// attribute vec3 a_startPosition;
attribute vec3 a_position;
// attribute float32 a_color;

void main() {
    gl_Position = vec4(a_position, 1.0);
    // gl_Position.xyz = a_position;
    // gl_Position.w = 1.0;
    gl_PointSize = 90.0;
}
"""

# Deliberately add precision qualifiers to test automatic GLSL code conversion
FRAG_SHADER = """
# version 120
precision highp float;
uniform sampler2D texture1;
uniform vec4 u_color;

void main()
{
    highp vec4 texColor;
    // texColor = texture2D(s_texture, gl_PointCoord);
    // gl_FragColor = vec4(u_color) * texColor;
    // gl_FragColor = vec4(u_color);
    // gl_FragColor.r = a_color;
    gl_FragColor.a *= 1;
}
"""


class Canvas(app.Canvas):

    def __init__(self, data, pos, color):
        app.Canvas.__init__(self, keys='interactive', size=(800, 600))
        # mesh = geometry.MeshData(pos, color)
        # print(mesh)
        # Constrained delaunay triangulation
        # _, self.triangles = geometry.triangulate(pos)
        # self.triangles = self.triangles.astype('int32')
        print(pos.shape)
        delaunay = Delaunay(pos[:, :2])
        self.triangles = delaunay.simplices.T.astype(np.uint32)
        # tri = matplotlib.tri.Triangulation(v[indices, 0], v[indices, 1]) # Delaunay triangulation
        print('tri', self.triangles.shape)
        delaunay.close()
        # self.vertices = pos
        vtype = [('a_position', np.float32, 3)
                 # , ('a_color', np.float32, 1)
                 ]
        # self.vertices = np.array(pos[:, :2].reshape((1, -1)), dtype=vtype)
        self.vertices = np.array([(v,) for v in pos], dtype=vtype)
        print(self.vertices.shape, type(self.vertices))

        # Create program
        self.program = gloo.Program(VERT_SHADER, FRAG_SHADER)
        print(self.vertices)
        self.program.bind(gloo.VertexBuffer(self.vertices))
        self.triangle_buffer = gloo.IndexBuffer(self.triangles)

        # self.faces = gloo.IndexBuffer(self.triangles)
        # self.program.bind(self.faces)
        # self.program.bind(gloo.VertexBuffer(data))
        # self.program['s_texture'] = gloo.Texture2D(pos)

        # Create first explosion
        # self._update_frame(pos, a)

        # gloo.set_clear_color('white')
        # gloo.set_state('opaque')
        # gloo.set_polygon_offset(1, 1)

        # Enable blending
        gloo.set_state(blend=True, clear_color='black',
                       blend_func=('src_alpha', 'one'))

        gloo.set_viewport(0, 0, self.physical_size[0], self.physical_size[1])

        # self._timer = app.Timer('auto', connect=self.update, start=True)

        self.show()
        # time.sleep(1)
        # im2 = self.render()
        # print(im2.shape)
        # io.imsave('vis.png', im2)
        # io.write_png('vis-.png', im2)

    def on_resize(self, event):
        width, height = event.physical_size
        gloo.set_viewport(0, 0, width, height)

    def on_draw(self, event):

        # Clear
        gloo.clear()

        # Draw
        gloo.set_state(blend=False, depth_test=True, polygon_offset_fill=True)
        self.program['u_color'] = 1, 1, 1, 1
        self.program.draw('triangles', self.triangle_buffer)
        # self.program['u_time'] = time.time() - self._starttime
        # self.program.draw('points')
        # self.program.draw('triangles', self.faces)

        # Outline
        # gloo.set_state(blend=True, depth_test=True, polygon_offset_fill=False)
        # gloo.set_depth_mask(False)

        # # New explosion?
        # if 0 and time.time() - self._starttime > 1.5:
        #     self._new_explosion()

    def _update_frame(self, im):
        # # New centerpos
        # centerpos = np.random.uniform(-0.5, 0.5, (3,))
        # self._program['u_centerPosition'] = centerpos

        # New color, scale alpha with N
        alpha = 1.0 / N ** 0.08
        color = np.random.uniform(0.1, 0.9, (3,))

        self._program['u_color'] = tuple(color) + (alpha,)

        # Create new vertex data
        # data['a_lifetime'] = np.random.normal(2.0, 0.5, (N,))
        # data['a_startPosition'] = np.random.normal(0.0, 0.2, (N, 3))
        data['a_startPosition'] = im
        # data['a_pos'] = im
        data['a_color'] = a
        # data['a_endPosition'] = np.random.normal(0.0, 1.2, (N, 3))

        # # Set time to zero
        # self._starttime = time.time()


if __name__ == '__main__':
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
    # print(data['v'].shape)

    # # Create a texture
    # radius = 32
    # im = np.random.normal(
    #     0.8, 0.3, (radius * 2 + 1, radius * 2 + 1)).astype(np.float32)
    #
    # # Mask it with a disk
    # L = np.linspace(-radius, radius, 2 * radius + 1)
    # (X, Y) = np.meshgrid(L, L)
    # im *= np.array((X ** 2 + Y ** 2) <= radius * radius, dtype='float32')
    # print(im.shape)
    #

    # float64 is not allowed for texture
    pos = data['v'].astype('float32')
    a = data['y'][:, 0].astype('float32')
    phi = data['y'][:, 1].astype('float32')

    # Set number of particles, you should be able to scale this to 100000
    N = a.size
    # Create vertex data container
    data = np.zeros(N, [('a_startPosition', np.float32, 3),
                        ('a_color', np.float32)])

    # c = Canvas(data, pos, a)
    n = 10
    c = Canvas(data, np.random.normal(0, 1, size=(n, 3)).astype('float32'),
               np.linspace(0, 1, n))
    im = c.render()
    c.close()
    plt.imshow(im, origin='left')
    n = 9
    print(im.shape)
    plt.xticks(np.linspace(0, im.shape[1], n),
               np.linspace(-0.1, 0.1, n).round(2))
    plt.yticks(np.linspace(0, im.shape[0], n),
               np.linspace(-0.1, 0.1, n).round(2))
    # TODO sci labels
    plt.show()

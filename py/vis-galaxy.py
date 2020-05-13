#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vispy: gallery 30
# Copyright (c) Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

import numpy as np
import itertools
from vispy import gloo
from vispy import app
import vispy.scene
from vispy.util.transforms import perspective, translate, rotate
# local
import util

VERT_SHADER = """
#version 120
// Uniforms
// ------------------------------------
uniform mat4  u_model;
uniform mat4  u_view;
uniform mat4  u_projection;
uniform float u_size;


// Attributes
// ------------------------------------
attribute vec3  a_position;
attribute float a_dist; // color

// Varyings
// ------------------------------------
varying float v_dist;

void main (void) {
    v_dist  = a_dist;
    gl_Position = u_projection * u_view * u_model * vec4(a_position,1.0);
    gl_PointSize = u_size;
}
"""

FRAG_SHADER = """
#version 120
// Uniforms
// ------------------------------------
uniform sampler2D u_colormap;

// Varyings
// ------------------------------------
varying float v_dist;

// Main
// ------------------------------------
void main()
{
    gl_FragColor = vec4(v_dist, 1-v_dist, 0, 1);
}
"""


class MyCanvas(app.Canvas):
    # class MyCanvas(vispy.scene.SceneCanvas):

    def __init__(self, pos=None, colors=None):
        """
        pos = np.ndarray or list of np.ndarray
        """
        app.Canvas.__init__(self, keys='interactive', size=(800, 600))
        self.title = "3D scatter plot [click & drag mouse to rotate]"

        if pos is None:
            p = 50000
            n = p
        elif isinstance(pos, list):
            n = pos[0].shape[0]
        else:
            n = pos.shape[0]

        self.cycle_data = isinstance(pos, list)
        self.data = np.zeros(n, [('a_position', np.float32, 3),
                                 ('a_dist', np.float32)])

        if isinstance(pos, np.ndarray):
            colors = [colors]
            pos = [pos]

        self.colors = itertools.cycle(colors)
        self.pos = itertools.cycle(pos)

        self.point_size = 9
        self.program = gloo.Program(VERT_SHADER, FRAG_SHADER)
        self.model = np.eye(4, dtype=np.float32)
        self.projection = np.eye(4, dtype=np.float32)
        self.theta, self.phi, self.xi = 0, 0, 0
        self.update_data()

        self.translate = 5
        self.view = translate((0, 0, -self.translate))

        self.program.bind(gloo.VertexBuffer(self.data))
        # self.program['u_colormap'] = gloo.Texture2D(cmap)
        self.program['u_size'] = self.point_size / self.translate
        self.program['u_model'] = self.model
        self.program['u_view'] = self.view

        self.apply_zoom()

        gloo.set_state(depth_test=False, blend=True,
                       blend_func=('src_alpha', 'one'), clear_color='black')

        # Start the timer upon initialization.
        self.timer = app.Timer('auto', connect=self.on_timer)
        self.timer.start()

        self.show()

    def update_data(self):
        self.data['a_dist'] = next(self.colors)
        self.data['a_position'] = next(self.pos)
        # data['a_dist'] = colors
        # data['a_position'] = pos
        self.program.bind(gloo.VertexBuffer(self.data))

    def on_key_press(self, event):
        if event.text == ' ':
            if self.timer.running:
                self.timer.stop()
            else:
                self.timer.start()

    def on_timer(self, event):
        # self.theta += .11
        # self.phi += .13
        self.xi += .02
        # self.model = np.dot(rotate(self.theta, (0, 0, 1)),
        #                     rotate(self.phi, (0, 1, 0)))
        if self.xi > 1:
            # precise timing is not important
            # reset naively, no need for modulo
            self.xi -= 1
            if self.cycle_data:
                self.update_data()

        self.program['u_model'] = self.model
        self.update()

    def on_resize(self, event):
        self.apply_zoom()

    def on_mouse_wheel(self, event):
        self.translate -= event.delta[1]
        self.translate = max(2, self.translate)
        self.view = translate((0, 0, -self.translate))
        self.program['u_view'] = self.view
        self.program['u_size'] = self.point_size / self.translate
        self.update()

    def on_mouse_move(self, event):
        # TODO use realist rotations
        if event.button == 1 and event.last_event is not None:
            w, h = self.size
            dx, dy = event.pos - event.last_event.pos
            rotation_speed = 100
            self.phi += rotation_speed * dx / w
            self.theta += rotation_speed * dy / h
            self.model = np.dot(rotate(self.theta, (0, 0, 1)),
                                rotate(self.phi, (0, 1, 0)))
            self.update()

    def on_draw(self, event):
        gloo.clear()
        # gloo.clear('white')
        self.program.draw('points')

    def apply_zoom(self):
        gloo.set_viewport(0, 0, self.physical_size[0], self.physical_size[1])
        self.projection = perspective(45.0, self.size[0] /
                                      float(self.size[1]), 1.0, 1000.0)
        self.program['u_projection'] = self.projection


if __name__ == '__main__':
    dir = '../tmp'
    fn = 'out.zip'
    data = util.parse_file(dir, fn, 'out.txt')
    viewtype = 2
    if viewtype == 0:
        n = 100000
        pos = data['v'][:n]
        colors = data['y'][:n, 0]
        print(colors.min(), colors.max(), colors.mean())
        colors -= colors.min()
        colors /= colors.max()
        print(colors.min(), colors.max(), colors.mean())
        # print(colors.min(), colors.max())
        # colors = np.random.normal(0, 1, size=colors.size)
        # colors = np.linspace(0, 1, colors.size)
        for i in range(2):
            pos[:, i] /= pos[:, i].max()

    elif viewtype == 1:
        n = 1000000
        colors = [z[:n, 0] for z in data['z']]
        max_color = max((c.max() for c in colors))
        pos = [w[:n] for w in data['w']]
        for i in range(len(colors)):
            colors[i] /= max_color
        # for c in colors:
        #     c /= max_color

        for p in pos:
            for i in range(2):
                p[:, i] /= p[:, i].max()

    elif viewtype == 2:
        n = 100000
        m = 10
        # normalize z_offset
        max_z_offset = max((np.abs(w[:, -1]).max() for w in data['w'][:m]))
        if max_z_offset > 0:
            for w in data['w'][:m]:
                # normalize
                w[:, -1] /= max_z_offset
                # center
                w[:, -1] -= 0.5

        # pos = np.vstack([data['w'][i][:n] for i in range(len(data['w']))])
        print(len(data['z']))
        pos = np.vstack([w[:n] for w in data['w'][:m]])
        a = np.vstack([z[:n, :1] for z in data['z'][:m]]).reshape(-1)
        # mu = a.mean()
        # std = a.std()
        # pos = np.vstack((w[:n] for w in data['w']))
        print(a.shape, pos.shape)
        a /= a.max()
        for i in range(2):
            pos[:, i] /= pos[:, i].max()

        colors = a

    canvas = MyCanvas(pos, colors)
    app.run()

"""Render an orbit or inside-lookaround video of a textured GLB scene.

Runs headless (EGL) in the `ns` env, which has open3d and ffmpeg:
  python scene_generation/iphone/render_glb_video.py scene.glb out.mp4 --mode lookaround
"""

import argparse
import os
import subprocess
import tempfile

import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering


def main():
  p = argparse.ArgumentParser()
  p.add_argument('glb')
  p.add_argument('out_mp4')
  p.add_argument('--mode', choices=['orbit', 'lookaround'], default='orbit')
  p.add_argument('--frames', type=int, default=150)
  p.add_argument('--fps', type=int, default=30)
  p.add_argument('--width', type=int, default=1280)
  p.add_argument('--height', type=int, default=720)
  p.add_argument('--up', choices=['y', 'z'], default='y', help='up axis of the file (GLB is usually y)')
  p.add_argument('--eye-height', type=float, default=0.35, help='lookaround: camera height as a fraction of scene height')
  args = p.parse_args()

  model = o3d.io.read_triangle_model(args.glb)
  mesh = o3d.io.read_triangle_mesh(args.glb, enable_post_processing=True)
  bbox = mesh.get_axis_aligned_bounding_box()
  center = bbox.get_center()  # (3,)
  extent = bbox.get_extent()  # (3,)
  radius = float(np.linalg.norm(extent)) * 0.55
  up_idx = 1 if args.up == 'y' else 2
  up = np.zeros(3)
  up[up_idx] = 1.0
  side = [i for i in range(3) if i != up_idx]

  r = rendering.OffscreenRenderer(args.width, args.height)
  r.scene.set_background([0.1, 0.1, 0.12, 1.0])
  r.scene.scene.set_sun_light([0.3, -1.0, 0.4], [1.0, 1.0, 1.0], 60000)
  r.scene.scene.enable_sun_light(True)
  r.scene.scene.set_indirect_light_intensity(30000)
  for i, mi in enumerate(model.meshes):
    mat = model.materials[mi.material_idx]
    mat.shader = 'defaultUnlit'
    r.scene.add_geometry(f'm{i}', mi.mesh, mat)
  r.scene.camera.set_projection(
    60.0, args.width / args.height, 0.05, 100.0, rendering.Camera.FovType.Vertical
  )

  with tempfile.TemporaryDirectory() as tmp:
    for k in range(args.frames):
      a = 2 * np.pi * k / args.frames
      if args.mode == 'orbit':
        eye = center.copy()
        eye[side[0]] += radius * np.cos(a)
        eye[side[1]] += radius * np.sin(a)
        eye[up_idx] += extent[up_idx] * 0.9
        r.scene.camera.look_at(center, eye, up)
      else:
        eye = center.copy()
        eye[up_idx] = bbox.min_bound[up_idx] + args.eye_height * extent[up_idx]
        target = eye.copy()
        target[side[0]] += np.cos(a)
        target[side[1]] += np.sin(a)
        r.scene.camera.look_at(target, eye, up)
      o3d.io.write_image(os.path.join(tmp, f'frame_{k:04d}.png'), r.render_to_image())
    subprocess.run(
      ['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(args.fps),
       '-i', os.path.join(tmp, 'frame_%04d.png'), '-c:v', 'libx264',
       '-pix_fmt', 'yuv420p', '-crf', '20', args.out_mp4],
      check=True,
    )
  print('wrote', args.out_mp4)


if __name__ == '__main__':
  main()

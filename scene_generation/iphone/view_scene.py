"""Browser viewer (viser) for a scanned scene: textured mesh, aligned splat, MuJoCo collision parts.

  source ~/.ns_deps/miniconda3/bin/activate ns
  python scene_generation/iphone/view_scene.py --mesh room_mujoco/scene_zup.glb \\
      --splat scene_splat/aligned/splat_aligned.ply --collision-dir room_mujoco/collision --port 8080
Then open http://localhost:8080 and use the checkboxes to toggle layers.
"""

import argparse
import glob
import os
import time

import numpy as np
import trimesh
import viser
from plyfile import PlyData
from scipy.spatial.transform import Rotation as Rot

SH_C0 = 0.28209479177387814


def load_splat(path, min_opacity, max_points):
  v = PlyData.read(path)['vertex']
  centers = np.stack([v['x'], v['y'], v['z']], 1).astype(np.float32)  # (N, 3)
  opac = 1 / (1 + np.exp(-np.asarray(v['opacity'], dtype=np.float32)))  # (N,)
  rgb = np.clip(0.5 + SH_C0 * np.stack([v['f_dc_0'], v['f_dc_1'], v['f_dc_2']], 1), 0, 1).astype(np.float32)
  scales = np.exp(np.stack([v['scale_0'], v['scale_1'], v['scale_2']], 1)).astype(np.float32)  # (N, 3)
  q = np.stack([v['rot_1'], v['rot_2'], v['rot_3'], v['rot_0']], 1)  # xyzw
  keep = opac > min_opacity
  if keep.sum() > max_points:
    idx = np.random.default_rng(0).choice(np.flatnonzero(keep), max_points, replace=False)
    keep = np.zeros_like(keep)
    keep[idx] = True
  R = Rot.from_quat(q[keep]).as_matrix()  # (M, 3, 3)
  S = scales[keep]
  cov = np.einsum('nij,nj,nkj->nik', R, S**2, R)  # (M, 3, 3)
  return centers[keep], cov.astype(np.float32), rgb[keep], opac[keep][:, None]


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--mesh', default=None, help='z-up textured glb')
  p.add_argument('--splat', default=None, help='aligned splat ply')
  p.add_argument('--collision-dir', default=None, help='folder of convex part .obj files')
  p.add_argument('--port', type=int, default=8080)
  p.add_argument('--min-opacity', type=float, default=0.1)
  p.add_argument('--max-points', type=int, default=400000)
  args = p.parse_args()

  server = viser.ViserServer(port=args.port)
  server.scene.set_up_direction('+z')
  server.scene.add_grid('/grid', width=20, height=20, plane='xy')
  handles = {}

  if args.mesh:
    with open(args.mesh, 'rb') as f:
      handles['mesh'] = server.scene.add_glb('/mesh', f.read())
    print('mesh loaded')
  if args.splat:
    c, cov, rgb, op = load_splat(args.splat, args.min_opacity, args.max_points)
    handles['splat'] = server.scene.add_gaussian_splats('/splat', c, cov, rgb, op, visible=False)
    print(f'splat loaded: {len(c)} gaussians')
  if args.collision_dir:
    parts = [trimesh.load(f, force='mesh') for f in sorted(glob.glob(os.path.join(args.collision_dir, '*.obj')))]
    if parts:
      merged = trimesh.util.concatenate(parts)
      handles['collision'] = server.scene.add_mesh_simple(
        '/collision', merged.vertices, merged.faces, color=(60, 220, 90), opacity=0.35, visible=False
      )
      print(f'collision parts loaded: {len(parts)}')

  with server.gui.add_folder('Layers'):
    boxes = {k: server.gui.add_checkbox(k, initial_value=h.visible) for k, h in handles.items()}
  for k, box in boxes.items():
    @box.on_update
    def _(_, k=k):
      handles[k].visible = boxes[k].value

  @server.on_client_connect
  def _(client):
    # Start inside the room at eye height, looking across it.
    client.camera.position = (0.0, -2.0, 1.5)
    client.camera.look_at = (0.0, 2.0, 1.2)
    client.camera.up_direction = (0.0, 0.0, 1.0)

  print(f'open http://localhost:{args.port}')
  while True:
    time.sleep(1.0)


if __name__ == '__main__':
  main()

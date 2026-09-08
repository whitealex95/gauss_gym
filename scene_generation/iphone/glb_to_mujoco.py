"""Turn a Polycam GLB (photo mode, no LiDAR) into a MuJoCo scene.

Outputs, in --out-dir:
  visual_<i>.obj/.mtl/.png  textured visual meshes (one per GLB submesh), z-up, floor at z=0
  scene_zup.glb             the same for viser
  collision/part_<i>.obj    convex parts from CoACD for walls and furniture (a z band, no floor, no ceiling)
  floor_hfield.png          height field of the walkable surface (floor + clutter up to --hfield-max-z)
  scene.xml                 MJCF that loads all of the above

Run in the `ns` env (needs trimesh, open3d, coacd, mujoco):
  python scene_generation/iphone/glb_to_mujoco.py scene.glb --out-dir out/ --scale 1.0
"""

import argparse
import os

import numpy as np
import open3d as o3d
import trimesh
from PIL import Image
from trimesh.exchange.obj import export_obj


def clean_submesh(g: trimesh.Trimesh, min_area: float) -> trimesh.Trimesh:
  # Connectivity must be computed on a copy with merged vertices: the textured
  # mesh duplicates vertices along UV seams, so it looks like thousands of parts.
  geo = trimesh.Trimesh(g.vertices.copy(), g.faces.copy(), process=False)
  geo.merge_vertices()
  comps = trimesh.graph.connected_components(geo.face_adjacency, nodes=np.arange(len(geo.faces)))
  face_area = geo.area_faces  # (F,)
  keep = np.zeros(len(geo.faces), dtype=bool)
  for c in comps:
    if face_area[c].sum() >= min_area:
      keep[c] = True
  g = g.copy()
  g.update_faces(keep)
  g.remove_unreferenced_vertices()
  return g


def frame_transform(meshes, scale: float, floor_percentile: float) -> np.ndarray:
  # GLB is y-up. Rotate to z-up, scale, then put the floor at z=0.
  T = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
  T = trimesh.transformations.scale_matrix(scale) @ T
  verts = np.concatenate([trimesh.transform_points(m.vertices, T) for m in meshes])  # (V, 3)
  floor_z = np.percentile(verts[:, 2], floor_percentile)
  T = trimesh.transformations.translation_matrix([0, 0, -floor_z]) @ T
  return T


def run_coacd(mesh: trimesh.Trimesh, out_dir: str, threshold: float, max_hulls: int, target_faces: int,
              zmin: float, zmax: float):
  import coacd

  # Keep only walls and furniture: the floor is handled by the height field and
  # the ceiling does not matter for a ground robot. Hulls that span floor to
  # ceiling turn into wedges that fill the room.
  mesh = mesh.slice_plane([0, 0, zmin], [0, 0, 1.0])
  mesh = mesh.slice_plane([0, 0, zmax], [0, 0, -1.0])
  o3 = o3d.geometry.TriangleMesh(
    o3d.utility.Vector3dVector(mesh.vertices), o3d.utility.Vector3iVector(mesh.faces)
  )
  if len(mesh.faces) > target_faces:
    o3 = o3.simplify_quadric_decimation(target_faces)
  parts = coacd.run_coacd(
    coacd.Mesh(np.asarray(o3.vertices), np.asarray(o3.triangles)),
    threshold=threshold,
    max_convex_hull=max_hulls,
  )
  os.makedirs(out_dir, exist_ok=True)
  names = []
  for i, (v, f) in enumerate(parts):
    name = f'part_{i:03d}.obj'
    trimesh.Trimesh(v, f).export(os.path.join(out_dir, name))
    names.append(name)
  return names


def make_hfield(mesh: trimesh.Trimesh, out_png: str, res: float, max_z: float):
  # Cast rays straight down on a grid and record the first hit below max_z.
  scene = o3d.t.geometry.RaycastingScene()
  scene.add_triangles(
    o3d.core.Tensor(mesh.vertices, dtype=o3d.core.float32),
    o3d.core.Tensor(mesh.faces, dtype=o3d.core.uint32),
  )
  (x0, y0, _), (x1, y1, _) = mesh.bounds
  xs = np.arange(x0, x1, res)
  ys = np.arange(y0, y1, res)
  gx, gy = np.meshgrid(xs, ys)  # (H, W)
  origins = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, max_z)], 1)  # (H*W, 3)
  dirs = np.tile([0.0, 0.0, -1.0], (origins.shape[0], 1))
  rays = o3d.core.Tensor(np.concatenate([origins, dirs], 1), dtype=o3d.core.float32)
  t_hit = scene.cast_rays(rays)['t_hit'].numpy()  # (H*W,)
  height = np.where(np.isfinite(t_hit), max_z - t_hit, 0.0).reshape(gx.shape)
  height = np.clip(height, 0.0, max_z)
  # MuJoCo reads PNG rows top to bottom as +y to -y, so flip so row 0 is max y.
  img = np.flipud(height / max_z * 65535.0).astype(np.uint16)
  Image.fromarray(img).save(out_png)
  size_x, size_y = xs[-1] - xs[0], ys[-1] - ys[0]
  center = ((xs[0] + xs[-1]) / 2, (ys[0] + ys[-1]) / 2)
  return size_x, size_y, center


def write_mjcf(path, visual_names, collision_names, hfield, add_plane):
  lines = [
    '<mujoco model="scanned_scene">',
    '  <compiler meshdir="." texturedir="." angle="radian"/>',
    '  <option timestep="0.002"/>',
    '  <visual><global offwidth="1280" offheight="720"/></visual>',
    '  <default>',
    '    <geom solref="0.01 1" solimp="0.9 0.95 0.001"/>',
    '  </default>',
    '  <asset>',
  ]
  for i, (obj, png) in enumerate(visual_names):
    lines += [
      f'    <texture name="tex_{i}" type="2d" file="{png}"/>',
      f'    <material name="mat_{i}" texture="tex_{i}" specular="0" shininess="0"/>',
      f'    <mesh name="vis_{i}" file="{obj}"/>',
    ]
  for i, name in enumerate(collision_names):
    lines.append(f'    <mesh name="col_{i:03d}" file="collision/{name}"/>')
  if hfield is not None:
    sx, sy, _, max_z = hfield
    lines.append(f'    <hfield name="floor" file="floor_hfield.png" size="{sx / 2:.3f} {sy / 2:.3f} {max_z:.3f} 0.05"/>')
  lines += [
    '  </asset>',
    '  <worldbody>',
    '    <light pos="0 0 4" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>',
    '    <light pos="2 2 3" dir="-0.5 -0.5 -1" diffuse="0.5 0.5 0.5"/>',
  ]
  for i in range(len(visual_names)):
    lines.append(f'    <geom type="mesh" mesh="vis_{i}" material="mat_{i}" contype="0" conaffinity="0" group="1"/>')
  for i in range(len(collision_names)):
    lines.append(f'    <geom type="mesh" mesh="col_{i:03d}" rgba="0 1 0 0.3" group="3"/>')
  if hfield is not None:
    _, _, (cx, cy), _ = hfield
    lines.append(f'    <geom type="hfield" hfield="floor" pos="{cx:.3f} {cy:.3f} 0" rgba="0.3 0.5 1 0.4" group="3"/>')
  if add_plane:
    lines.append('    <geom type="plane" size="0 0 0.05" pos="0 0 0" rgba="0.5 0.5 0.5 0.3" group="3"/>')
  lines += [
    '    <body name="test_ball" pos="0 0 1.0">',
    '      <freejoint/>',
    '      <geom type="sphere" size="0.1" mass="1" rgba="1 0.3 0.3 1"/>',
    '    </body>',
    '  </worldbody>',
    '</mujoco>',
  ]
  with open(path, 'w') as f:
    f.write('\n'.join(lines) + '\n')


def main():
  p = argparse.ArgumentParser()
  p.add_argument('glb')
  p.add_argument('--out-dir', required=True)
  p.add_argument('--scale', type=float, default=1.0, help='measured_m / in_mesh_m')
  p.add_argument('--min-area', type=float, default=0.05, help='drop pieces smaller than this (m^2)')
  p.add_argument('--floor-percentile', type=float, default=1.0)
  p.add_argument('--coacd-threshold', type=float, default=0.05)
  p.add_argument('--max-hulls', type=int, default=128)
  p.add_argument('--coacd-faces', type=int, default=100000, help='decimate to this many faces before CoACD')
  p.add_argument('--coacd-zmin', type=float, default=0.25, help='CoACD only sees geometry above this height')
  p.add_argument('--coacd-zmax', type=float, default=2.2, help='and below this height (drops the ceiling)')
  p.add_argument('--no-coacd', action='store_true')
  p.add_argument('--no-hfield', action='store_true')
  p.add_argument('--hfield-res', type=float, default=0.05)
  p.add_argument('--hfield-max-z', type=float, default=0.5)
  p.add_argument('--no-plane', action='store_true')
  args = p.parse_args()

  os.makedirs(args.out_dir, exist_ok=True)
  scene = trimesh.load(args.glb)
  subs = list(scene.geometry.values()) if isinstance(scene, trimesh.Scene) else [scene]
  subs = [clean_submesh(g, args.min_area) for g in subs]
  subs = [g for g in subs if len(g.faces) > 0]
  T = frame_transform(subs, args.scale, args.floor_percentile)
  for g in subs:
    g.apply_transform(T)

  visual_names = []
  out_scene = trimesh.Scene()
  for i, g in enumerate(subs):
    obj, png = f'visual_{i}.obj', f'visual_{i}.png'
    obj_text, files = export_obj(g, include_texture=True, return_texture=True, mtl_name=f'visual_{i}.mtl')
    for name, data in files.items():
      if name.endswith('.mtl'):
        data = data.decode()
        for line in data.splitlines():
          if line.startswith('map_Kd'):
            data = data.replace(line.split(None, 1)[1].strip(), png)
        data = data.encode()
        name = f'visual_{i}.mtl'
      elif name.endswith('.png'):
        name = png
      with open(os.path.join(args.out_dir, name), 'wb') as f:
        f.write(data)
    with open(os.path.join(args.out_dir, obj), 'w') as f:
      f.write(obj_text)
    visual_names.append((obj, png))
    out_scene.add_geometry(g, node_name=f'visual_{i}')
  out_scene.export(os.path.join(args.out_dir, 'scene_zup.glb'))
  print('visual meshes:', [(n, len(g.faces)) for n, g in zip(visual_names, subs)])

  geo = trimesh.util.concatenate([trimesh.Trimesh(g.vertices, g.faces) for g in subs])
  print('bounds (z-up, m):', geo.bounds.round(2).tolist())

  collision_names = []
  if not args.no_coacd:
    collision_names = run_coacd(
      geo, os.path.join(args.out_dir, 'collision'), args.coacd_threshold, args.max_hulls, args.coacd_faces,
      args.coacd_zmin, args.coacd_zmax,
    )
    print('collision parts:', len(collision_names))

  hfield = None
  if not args.no_hfield:
    sx, sy, center = make_hfield(
      geo, os.path.join(args.out_dir, 'floor_hfield.png'), args.hfield_res, args.hfield_max_z
    )
    hfield = (sx, sy, center, args.hfield_max_z)
    print('hfield size (m):', round(sx, 2), round(sy, 2), 'center', np.round(center, 2).tolist())

  write_mjcf(os.path.join(args.out_dir, 'scene.xml'), visual_names, collision_names, hfield, not args.no_plane)
  print('wrote', os.path.join(args.out_dir, 'scene.xml'))


if __name__ == '__main__':
  main()

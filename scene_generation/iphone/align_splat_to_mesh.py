"""Align a Gaussian splat trained from photos to the z-up MuJoCo mesh frame.

Frames involved:
  * COLMAP / transforms.json frame: arbitrary orientation and scale. sparse_pc.ply lives here.
  * nerfstudio internal frame: what ns-export writes. Auto-oriented to z-up and
    scaled by the factor saved in splatfacto/dataparser_transforms.json.
  * mesh frame: z-up, metric, floor at z=0 (from glb_to_mujoco.py).

Steps:
  1. Sparse cloud -> mesh: up from the Polycam per-frame gravity vectors rotated
     by the camera poses (falls back to the mean camera up), coarse scale from
     point spreads, then scaled ICP from several starting yaws.
  2. Compose with the exact COLMAP -> splat frame transform (dataparser
     transform + scale, then +90 deg about x), then refine with a tight rigid ICP.

Writes <out_dir>/splat_to_mesh.json (4x4 similarity) and <out_dir>/splat_aligned.ply.

  python scene_generation/iphone/align_splat_to_mesh.py \\
      --ply scene/splatfacto/splat.ply --transforms scene/transforms.json \\
      --gravity-dir raw/keyframes/gravity --images-dir raw/keyframes/images \\
      --mesh room_mujoco/scene_zup.glb --out-dir scene/aligned
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation as Rot


def load_points(ply_path, min_opacity):
  ply = PlyData.read(ply_path)
  v = ply['vertex']
  pts = np.stack([v['x'], v['y'], v['z']], 1).astype(np.float64)  # (N, 3)
  if 'opacity' in v.data.dtype.names:
    op = np.asarray(v['opacity'], dtype=np.float64)
    if op.min() < 0 or op.max() > 1:
      op = 1 / (1 + np.exp(-op))
    pts = pts[op > min_opacity]
  return pts, ply


def up_from_gravity(transforms, gravity_dir, images_dir):
  # ns-process-data copies images in sorted name order as frame_00001, frame_00002, ...
  exts = {'.jpg', '.jpeg', '.png', '.heic'}
  originals = sorted(p for p in Path(images_dir).iterdir() if p.suffix.lower() in exts)
  ups = []
  for f in transforms['frames']:
    idx = int(Path(f['file_path']).stem.split('_')[-1]) - 1
    if idx >= len(originals):
      continue
    g_path = Path(gravity_dir) / (originals[idx].stem + '.json')
    if not g_path.exists():
      continue
    g = json.load(open(g_path))
    g_cam = np.array([g['x'], g['y'], g['z']])  # gravity in device (= OpenGL camera) frame
    c2w = np.array(f['transform_matrix'])[:3, :3]
    ups.append(-(c2w @ g_cam))
  if not ups:
    return None
  up = np.mean(ups, 0)
  print(f'gravity used from {len(ups)} frames, spread of up vectors: {np.std(ups, 0).round(3).tolist()}')
  return up / np.linalg.norm(up)


def up_from_cameras(transforms):
  ups = [np.array(f['transform_matrix'])[:3, 1] for f in transforms['frames']]  # camera +y is up in OpenGL
  up = np.mean(ups, 0)
  return up / np.linalg.norm(up)


def rotation_to_z(up):
  z = np.array([0.0, 0.0, 1.0])
  v = np.cross(up, z)
  s, c = np.linalg.norm(v), np.dot(up, z)
  if s < 1e-8:
    return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
  vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
  return np.eye(3) + vx + vx @ vx * ((1 - c) / s**2)


def yaw(a):
  c, s = np.cos(a), np.sin(a)
  return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def to_h(R, t, s=1.0):
  T = np.eye(4)
  T[:3, :3] = R * s
  T[:3, 3] = t
  return T


def trim_outliers(pts, pct=95):
  c = np.median(pts, 0)
  r = np.linalg.norm(pts - c, axis=1)
  return pts[r < np.percentile(r, pct)]


def fit_to_mesh(src, tgt_pc, R_up, scale, with_scaling, yaw_starts, dist, voxel):
  """Best similarity (or rigid if not with_scaling) from several starting yaws."""
  est = o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=with_scaling)
  crit = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
  src_z = trim_outliers(src @ R_up.T)
  c_src = np.median(src_z, 0)
  c_tgt = np.median(np.asarray(tgt_pc.points), 0)
  src_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(src)).voxel_down_sample(voxel / scale)
  best = None
  for k in range(yaw_starts):
    R0 = yaw(2 * np.pi * k / yaw_starts) @ R_up
    T0 = to_h(R0, c_tgt - scale * (R0 @ c_src), scale)
    res = o3d.pipelines.registration.registration_icp(src_pc, tgt_pc, dist, T0, est, crit)
    res = o3d.pipelines.registration.registration_icp(src_pc, tgt_pc, dist / 5, res.transformation, est, crit)
    score = (res.fitness, -res.inlier_rmse)
    if best is None or score > best[0]:
      best = (score, res.transformation, k)
  (fitness, neg_rmse), T, k = best
  return T, fitness, -neg_rmse, k


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--ply', required=True, help='splat.ply from ns-export, or sparse_pc.ply')
  p.add_argument('--transforms', required=True)
  p.add_argument('--gravity-dir', default=None, help='raw/keyframes/gravity from the Polycam export')
  p.add_argument('--images-dir', default=None, help='raw/keyframes/images, to map frame ids to gravity files')
  p.add_argument('--mesh', required=True, help='z-up mesh, e.g. room_mujoco/scene_zup.glb')
  p.add_argument('--out-dir', required=True)
  p.add_argument('--dataparser-transforms', default=None, help='defaults to dataparser_transforms.json next to the ply')
  p.add_argument('--sparse-ply', default=None, help='defaults to ply_file_path in transforms.json')
  p.add_argument('--min-opacity', type=float, default=0.3)
  p.add_argument('--mesh-samples', type=int, default=200000)
  p.add_argument('--yaw-starts', type=int, default=24)
  p.add_argument('--icp-dist', type=float, default=0.5, help='max correspondence distance (m) for coarse ICP')
  args = p.parse_args()
  os.makedirs(args.out_dir, exist_ok=True)

  transforms = json.load(open(args.transforms))
  mesh = trimesh.load(args.mesh, force='mesh')
  tgt_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(mesh.sample(args.mesh_samples)))
  tgt_pc = tgt_pc.voxel_down_sample(0.05)

  up = up_from_gravity(transforms, args.gravity_dir, args.images_dir) if args.gravity_dir and args.images_dir else None
  source_of_up = 'gravity'
  if up is None:
    up = up_from_cameras(transforms)
    source_of_up = 'camera up vectors'
  print(f'COLMAP-frame up from {source_of_up}: {np.round(up, 3).tolist()}')
  R_up = rotation_to_z(up)

  dp_path = args.dataparser_transforms or os.path.join(os.path.dirname(args.ply), 'dataparser_transforms.json')
  is_export = os.path.exists(dp_path)

  # Step 1: sparse COLMAP cloud -> mesh (scaled ICP). This is also the answer
  # when the input itself is the sparse cloud.
  sparse_path = args.sparse_ply or os.path.join(os.path.dirname(args.transforms), transforms.get('ply_file_path', 'sparse_pc.ply'))
  sparse_src, _ = load_points(args.ply if not is_export else sparse_path, 0.0)
  sp_z = trim_outliers(sparse_src @ R_up.T)
  spread = lambda x: np.percentile(np.linalg.norm((x - np.median(x, 0))[:, :2], axis=1), 80)
  s0 = spread(np.asarray(tgt_pc.points)) / spread(sp_z)
  T_sparse, fit, rmse, k = fit_to_mesh(sparse_src, tgt_pc, R_up, s0, True, args.yaw_starts, args.icp_dist, 0.05)
  s_sparse = np.cbrt(np.linalg.det(T_sparse[:3, :3]))
  print(f'sparse -> mesh: fitness {fit:.3f}, rmse {rmse:.3f} m, scale {s_sparse:.3f} (coarse guess {s0:.3f}, yaw start {k})')

  if not is_export:
    T, fitness, rmse_final, s = T_sparse, fit, rmse, s_sparse
    src, ply = load_points(args.ply, 0.0)
  else:
    # Step 2: compose. The exported splat frame is the dataparser transform
    # (rotation, translation, scale) followed by a +90 degree rotation about x,
    # which takes nerfstudio's y-up pose frame to the z-up internal frame.
    # Checked empirically: rigid ICP between the transformed sparse cloud and the
    # splat gives ~1.5 cm RMSE with that rotation and fails with any other.
    dp = json.load(open(dp_path))
    M = np.array(dp['transform'])
    R_x90 = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])
    T_colmap_to_splat = to_h(R_x90 @ M[:3, :3], R_x90 @ M[:3, 3] * dp['scale'], dp['scale'])
    T0 = T_sparse @ np.linalg.inv(T_colmap_to_splat)
    s = np.cbrt(np.linalg.det(T0[:3, :3]))
    print(f'splat scale in mesh frame = {s_sparse:.3f} / {dp["scale"]:.4f} = {s:.3f}')

    # Refine with a tight rigid ICP of the splat against the mesh (scale stays).
    src, ply = load_points(args.ply, args.min_opacity)
    src_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(src)).voxel_down_sample(0.05 / s)
    est = o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=False)
    crit = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
    ev = o3d.pipelines.registration.evaluate_registration(src_pc, tgt_pc, 0.1, T0)
    print(f'composed:  fitness {ev.fitness:.3f}, rmse {ev.inlier_rmse:.3f} m (within 0.1 m)')
    res = o3d.pipelines.registration.registration_icp(src_pc, tgt_pc, 0.1, T0, est, crit)
    T, fitness, rmse_final = res.transformation, res.fitness, res.inlier_rmse
    print(f'refined:   fitness {fitness:.3f}, rmse {rmse_final:.3f} m (within 0.1 m)')

  json.dump(
    {'splat_to_mesh': T.tolist(), 'scale': float(s), 'fitness': float(fitness), 'rmse': float(rmse_final),
     'up_source': source_of_up, 'input_is_export': bool(is_export)},
    open(os.path.join(args.out_dir, 'splat_to_mesh.json'), 'w'), indent=2,
  )

  # Aligned ply: move positions, rotate rotations, scale the (log) scales.
  v = ply['vertex']
  names = v.data.dtype.names
  pts = np.stack([v['x'], v['y'], v['z']], 1).astype(np.float64)
  pts = pts @ T[:3, :3].T + T[:3, 3]
  out = v.data.copy()
  out['x'], out['y'], out['z'] = pts[:, 0], pts[:, 1], pts[:, 2]
  if all(n in names for n in ['scale_0', 'scale_1', 'scale_2']):
    for n in ['scale_0', 'scale_1', 'scale_2']:
      out[n] = v[n] + np.log(s)
  if all(n in names for n in ['rot_0', 'rot_1', 'rot_2', 'rot_3']):
    q = np.stack([v['rot_1'], v['rot_2'], v['rot_3'], v['rot_0']], 1)  # wxyz -> xyzw
    q2 = (Rot.from_matrix(T[:3, :3] / s) * Rot.from_quat(q)).as_quat()
    out['rot_0'], out['rot_1'], out['rot_2'], out['rot_3'] = q2[:, 3], q2[:, 0], q2[:, 1], q2[:, 2]
  PlyData([PlyElement.describe(out, 'vertex')], text=False).write(os.path.join(args.out_dir, 'splat_aligned.ply'))
  print('wrote', os.path.join(args.out_dir, 'splat_aligned.ply'))


if __name__ == '__main__':
  main()

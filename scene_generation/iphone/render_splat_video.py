"""Render an orbit or inside-lookaround video of a Gaussian splat .ply with gsplat.

Same camera paths as render_glb_video.py, so a splat aligned to the MuJoCo mesh
(z-up, floor at z=0) can be compared with the mesh video frame by frame.

  python scene_generation/iphone/render_splat_video.py splat_aligned.ply out.mp4 --mode lookaround
"""

import argparse
import os
import subprocess
import tempfile

import numpy as np
import torch
from gsplat import rasterization
from PIL import Image
from plyfile import PlyData


def load_splat(path, device):
  v = PlyData.read(path)['vertex']
  names = v.data.dtype.names
  means = np.stack([v['x'], v['y'], v['z']], 1)  # (N, 3)
  scales = np.exp(np.stack([v['scale_0'], v['scale_1'], v['scale_2']], 1))  # (N, 3)
  quats = np.stack([v['rot_0'], v['rot_1'], v['rot_2'], v['rot_3']], 1)  # (N, 4) wxyz
  quats /= np.linalg.norm(quats, axis=1, keepdims=True)
  opac = 1 / (1 + np.exp(-np.asarray(v['opacity'])))  # (N,)
  dc = np.stack([v['f_dc_0'], v['f_dc_1'], v['f_dc_2']], 1)[:, None, :]  # (N, 1, 3)
  rest_names = sorted([n for n in names if n.startswith('f_rest_')], key=lambda n: int(n.split('_')[-1]))
  if rest_names:
    rest = np.stack([v[n] for n in rest_names], 1)  # (N, 3*K) channel-major
    K = len(rest_names) // 3
    rest = rest.reshape(-1, 3, K).transpose(0, 2, 1)  # (N, K, 3)
    sh = np.concatenate([dc, rest], 1)  # (N, K+1, 3)
  else:
    sh = dc
  sh_degree = int(np.sqrt(sh.shape[1]) - 1)
  t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).float().to(device)
  return t(means), t(quats), t(scales), t(opac), t(sh), sh_degree


def look_at_viewmat(eye, target, up):
  # OpenCV camera: x right, y down, z forward.
  z = target - eye
  z /= np.linalg.norm(z)
  x = np.cross(z, up)
  x /= np.linalg.norm(x)
  y = np.cross(z, x)
  c2w = np.eye(4)
  c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = x, y, z, eye
  return np.linalg.inv(c2w)


def main():
  p = argparse.ArgumentParser()
  p.add_argument('ply')
  p.add_argument('out_mp4')
  p.add_argument('--mode', choices=['orbit', 'lookaround'], default='lookaround')
  p.add_argument('--frames', type=int, default=150)
  p.add_argument('--fps', type=int, default=30)
  p.add_argument('--width', type=int, default=1280)
  p.add_argument('--height', type=int, default=720)
  p.add_argument('--fov-deg', type=float, default=60.0, help='vertical field of view')
  p.add_argument('--eye-height', type=float, default=1.2, help='lookaround: camera height above the floor (m)')
  p.add_argument('--center', type=float, nargs=3, default=None, help='lookaround: camera xyz; default is the splat median')
  args = p.parse_args()

  device = 'cuda'
  means, quats, scales, opac, sh, sh_degree = load_splat(args.ply, device)
  pts = means.cpu().numpy()
  lo, hi = np.percentile(pts, 2, 0), np.percentile(pts, 98, 0)
  center = (lo + hi) / 2
  extent = hi - lo
  radius = float(np.linalg.norm(extent)) * 0.55
  up = np.array([0.0, 0.0, 1.0])

  fy = args.height / (2 * np.tan(np.radians(args.fov_deg) / 2))
  K = torch.tensor([[fy, 0, args.width / 2], [0, fy, args.height / 2], [0, 0, 1]], dtype=torch.float32, device=device)

  with tempfile.TemporaryDirectory() as tmp:
    for k in range(args.frames):
      a = 2 * np.pi * k / args.frames
      if args.mode == 'orbit':
        eye = center + np.array([radius * np.cos(a), radius * np.sin(a), extent[2] * 0.9])
        target = center
      else:
        eye = np.array(args.center) if args.center else np.array([center[0], center[1], lo[2] + args.eye_height])
        target = eye + np.array([np.cos(a), np.sin(a), 0.0])
      viewmat = torch.from_numpy(look_at_viewmat(eye, target, up)).float().to(device)
      with torch.no_grad():
        img, _, _ = rasterization(
          means, quats, scales, opac, sh, viewmat[None], K[None], args.width, args.height,
          sh_degree=sh_degree, backgrounds=torch.tensor([[0.1, 0.1, 0.12]], device=device),
        )
      frame = (img[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
      Image.fromarray(frame).save(os.path.join(tmp, f'frame_{k:04d}.png'))
    subprocess.run(
      ['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(args.fps),
       '-i', os.path.join(tmp, 'frame_%04d.png'), '-c:v', 'libx264',
       '-pix_fmt', 'yuv420p', '-crf', '20', args.out_mp4],
      check=True,
    )
  print('wrote', args.out_mp4)


if __name__ == '__main__':
  main()

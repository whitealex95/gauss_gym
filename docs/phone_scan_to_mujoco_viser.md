# From a phone scan to a mesh you can use in MuJoCo and viser

This guide is for a phone **without LiDAR** (for example the Google Pixel 10a).
It covers the full path from a Polycam capture to:

1. a textured mesh you can look at in **viser**, and
2. a collision model that **MuJoCo** can simulate.

The `gauss_gym` Polycam pipeline (`scene_generation/iphone/polycam_scenes.sh`)
needs LiDAR depth maps and camera poses, which only iOS LiDAR devices export.
So this guide takes a different route and uses the textured mesh Polycam
produces from photos.

---

## 0. What the phone can and cannot give you

| Export | Pixel 10a (no LiDAR) | iPhone/iPad Pro (LiDAR) |
|---|---|---|
| Processed textured mesh (GLB, OBJ, ...) | yes | yes |
| Raw data zip (images, camera poses, depth maps) | no | yes, with Developer mode on |
| Metric scale | approximate | good |

Without LiDAR, Polycam runs **photogrammetry** on its servers and returns a
textured mesh. That mesh is what we work with. There is no depth, so the scale
can be off by a few percent and you should check it (Section 2.3).

---

## 1. Capture well (this matters more than anything after it)

Use Polycam **Photo mode**. Photogrammetry fails on the same things every time,
so avoid them:

- Move slowly. Take many photos with **60 to 80 % overlap**. A room needs 150 to 300 photos.
- Walk in loops and come back to where you started so the reconstruction closes.
- Keep the camera **level** and at a constant height. Tilting down toward the floor a bit is fine.
- Avoid plain white walls, glass, mirrors, and shiny floors. If you must, put objects or tape on them.
- Keep lighting constant. Do not switch lights on or off during a capture.
- Put a **known-size object** in the scene (a printed marker, a box you measured) so you can fix scale later.
- Do not move furniture or people during the capture.

Export as **GLTF (.glb)**. The GLB contains the mesh plus its texture image in one file.

---

## 2. Inspect and clean the GLB

Work in the `ns` conda environment. It already has open3d, trimesh, viser, and ffmpeg.

```bash
source ~/.ns_deps/miniconda3/bin/activate ns
```

### 2.1 Look at it

Render a lookaround video (camera stands inside the scene and turns 360°) and an
orbit video (camera circles the scene from above):

```bash
python scene_generation/iphone/render_glb_video.py scene.glb lookaround.mp4 --mode lookaround
python scene_generation/iphone/render_glb_video.py scene.glb orbit.mp4 --mode orbit
```

Or open it live in viser (Section 3). Look for: holes in the floor, floating
fragments, doubled walls, and a ceiling that is missing or full of holes.

### 2.2 Clean it

The Polycam mesh usually has small floating pieces and a rough boundary.
Two things to know before cleaning it:

- The GLB holds two or three submeshes, each with its own 4096 x 4096 texture. Keep them separate. Merging them into one mesh makes trimesh build a huge texture atlas.
- Each submesh is a patchwork of about 1,500 texture charts with duplicated vertices along the seams. `mesh.split()` sees every chart as its own piece and copies the texture for each one, which runs out of memory even on a 128 GB machine. Compute connectivity on a texture-free copy with merged vertices instead, then apply the resulting face mask to the textured mesh.

```python
import trimesh, numpy as np

def clean_submesh(g, min_area=0.05):
  geo = trimesh.Trimesh(g.vertices.copy(), g.faces.copy(), process=False)
  geo.merge_vertices()
  comps = trimesh.graph.connected_components(geo.face_adjacency, nodes=np.arange(len(geo.faces)))
  keep = np.zeros(len(geo.faces), dtype=bool)
  for c in comps:
    if geo.area_faces[c].sum() >= min_area:   # drop pieces smaller than 0.05 m²
      keep[c] = True
  g = g.copy()
  g.update_faces(keep)                         # keeps UVs and the texture
  g.remove_unreferenced_vertices()
  return g

scene = trimesh.load('scene.glb')
subs = [clean_submesh(g) for g in scene.geometry.values()]
```

`scene_generation/iphone/glb_to_mujoco.py` does all of this for you (Section 4).
For heavier cleanup (hole filling, smoothing, decimation) open the GLB in
Blender (`~/blender-5.0.1-linux-x64/blender`) and use
**Mesh > Clean Up** and the **Decimate** modifier, then export GLB again.

### 2.3 Fix the frame: up axis, floor height, scale

GLB files are **Y-up**. MuJoCo and viser are **Z-up**. Also put the floor at
`z = 0` and fix scale using something you measured. Apply the same transform
to every submesh.

```python
import trimesh, numpy as np

T = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])   # Y-up -> Z-up

# Scale. Example: a door you measured is 0.90 m wide but is 0.84 m in the mesh.
measured_m, in_mesh_m = 0.90, 0.84
T = trimesh.transformations.scale_matrix(measured_m / in_mesh_m) @ T

# Floor to z = 0. Use a low percentile so noise below the floor does not matter.
verts = np.concatenate([trimesh.transform_points(g.vertices, T) for g in subs])
floor_z = np.percentile(verts[:, 2], 1.0)
T = trimesh.transformations.translation_matrix([0, 0, -floor_z]) @ T

out = trimesh.Scene()
for i, g in enumerate(subs):
  g.apply_transform(T)
  out.add_geometry(g, node_name=f'visual_{i}')
out.export('scene_zup.glb')
```

To measure a distance in the mesh, open it in viser or Blender and read off two
vertex positions, or pick two points in Blender with the Measure tool.

---

## 3. Show it in viser

viser can load the GLB directly, texture included:

```python
import time, viser

server = viser.ViserServer()          # prints a URL, open it in a browser
with open('scene_zup.glb', 'rb') as f:
  server.scene.add_glb('/scene', f.read())
server.scene.add_grid('/grid', width=20, height=20)

while True:
  time.sleep(1.0)
```

If you already have a trimesh object, `server.scene.add_mesh_trimesh('/scene', mesh)` works too.
viser is the same viewer nerfstudio uses, so a Gaussian splat trained later (Section 5) shows up in the same tool.

---

## 4. Build the MuJoCo model

MuJoCo needs two different things from the scan:

- a **visual mesh** that is only drawn, and
- **collision geometry** that the physics uses.

They must be separate geoms. Never use the raw scan as a collision mesh: a
MuJoCo `mesh` geom collides as its **convex hull**, so a room mesh would become
one solid block and the robot would be stuck inside it.

### 4.1 The one-command way

Install the tools once (already done in the `ns` env on this machine):

```bash
pip install mujoco coacd
```

Then:

```bash
python scene_generation/iphone/glb_to_mujoco.py scene.glb --out-dir room_mujoco --scale 1.0
python -m mujoco.viewer --mjcf room_mujoco/scene.xml
```

The script does Sections 2.2 and 2.3, then writes:

| File | What it is |
|---|---|
| `visual_<i>.obj` + `.mtl` + `.png` | textured visual meshes, one per GLB submesh, z-up, floor at z = 0 |
| `scene_zup.glb` | the same for viser |
| `floor_hfield.png` | height field of the floor and anything on it up to `--hfield-max-z` (0.5 m) |
| `collision/part_<i>.obj` | CoACD convex parts of the walls and furniture |
| `scene.xml` | MJCF that loads all of it, plus a test ball |

In the viewer, press **1** to toggle the visual group and **3** to toggle the
collision group. The red test ball should land on the floor and stay put.

### 4.2 How the collision geometry is built, and why

The script uses two kinds of collision geometry together:

**Height field for the floor.** It casts a ray straight down at every 5 cm grid
cell and records the first hit. MuJoCo `hfield` geoms are fast and handle
stairs, ramps, and clutter on the floor well. They cannot represent overhangs.

**Convex parts for walls and furniture.** CoACD splits a concave mesh into
convex pieces. The script only feeds it the band from `--coacd-zmin` (0.25 m)
to `--coacd-zmax` (2.2 m). Below the band the height field already handles the
floor. Above it the ceiling is dropped, because hulls that reach from floor to
ceiling turn into wedges that fill the room interior. That is what happened on
the first try with the whole room: the test ball rested 16 cm above the floor
and slid away. With the band, walls become flat slabs and the ball rests on the
floor.

A flat `plane` at z = 0 is added underneath as a safety net for holes in the
scan. Pass `--no-plane` to drop it.

Useful knobs:

- `--coacd-threshold 0.05`: how closely hulls follow the surface. 0.02 to 0.1. Lower is tighter and slower.
- `--max-hulls 128`: cap on the number of parts. Fewer parts simulate faster.
- `--hfield-res 0.05`: grid cell size in meters.
- `--min-area 0.05`: drop floating fragments smaller than this.

For a flat room where you only care about walls, `--no-hfield` plus the plane
is enough. If you want to hand-place boxes instead, use the visual mesh as a
reference and skip CoACD with `--no-coacd`.

### 4.3 What the generated MJCF looks like

```xml
<mujoco model="scanned_scene">
  <compiler meshdir="." texturedir="." angle="radian"/>
  <visual><global offwidth="1280" offheight="720"/></visual>   <!-- lets mujoco.Renderer render 1280x720 -->
  <asset>
    <texture name="tex_0" type="2d" file="visual_0.png"/>
    <material name="mat_0" texture="tex_0" specular="0" shininess="0"/>
    <mesh name="vis_0" file="visual_0.obj"/>
    <mesh name="col_000" file="collision/part_000.obj"/>
    <!-- size = (size_x/2, size_y/2, max_z, base_thickness) -->
    <hfield name="floor" file="floor_hfield.png" size="4.475 5.700 0.500 0.05"/>
  </asset>
  <worldbody>
    <light pos="0 0 4" dir="0 0 -1"/>
    <geom type="mesh" mesh="vis_0" material="mat_0" contype="0" conaffinity="0" group="1"/>  <!-- visual only -->
    <geom type="mesh" mesh="col_000" rgba="0 1 0 0.3" group="3"/>                             <!-- collision only -->
    <geom type="hfield" hfield="floor" pos="-0.43 0.02 0" group="3"/>
    <geom type="plane" size="0 0 0.05" group="3"/>
    <body name="test_ball" pos="0 0 1"><freejoint/><geom type="sphere" size="0.1"/></body>
  </worldbody>
</mujoco>
```

Notes on the format:

- MuJoCo reads OBJ with `vt` texture coordinates. One texture per geom, which is why each GLB submesh becomes its own geom.
- `contype="0" conaffinity="0"` turns collision off for the visual geoms.
- Height field PNG rows go from +y (top) to -y (bottom). The script flips the array before saving.

### 4.4 Common problems

- **Robot falls through the floor.** The scan has holes there. Keep the plane, or lower `--hfield-max-z` so wall bottoms do not poke into the field.
- **Robot floats or slides.** A convex part spans floor and furniture. Raise `--coacd-zmin`, or lower `--coacd-threshold`.
- **Everything is tilted.** The floor in the scan is not flat because the capture drifted. Fit a plane to floor vertices and rotate the mesh so the plane normal is `+z`.
- **Scale is wrong.** Photogrammetry has no absolute scale. Always measure something and pass `--scale`.
- **Texture not showing.** The MTL must point at the PNG and `texturedir` must be right. The script writes these next to each other.

---

## 5. Optional: photoreal rendering with a Gaussian splat

If you want the camera images to look real (the `gauss_gym` approach: mesh for
physics, splat for pixels), you can train a splat from the same photos even
without LiDAR. You need the individual photos, not only the GLB. Polycam lets
you export the source images from a photo capture, or just record a video of
the same walk.

```bash
source ~/.ns_deps/miniconda3/bin/activate ns
conda install -c conda-forge colmap       # once

ns-process-data images --data photos/ --output-dir scene_ns/      # or: ns-process-data video --data walk.mp4 ...
ns-train splatfacto --data scene_ns/ --viewer.quit-on-train-completion True
ns-export gaussian-splat --load-config outputs/scene_ns/splatfacto/<timestamp>/config.yml --output-dir scene_ns/splat
```

COLMAP gives poses in an arbitrary frame and scale. Align the splat to the mesh
by picking three or more matching points in both and solving a similarity
transform (`trimesh.registration.procrustes` does this), then apply it to the
splat means. After that, the splat and the MuJoCo model share one frame and you
can render the simulated camera through gsplat or the nerfstudio viewer.

---

## 6. Checklist

1. Capture in Photo mode with heavy overlap and a known-size object.
2. Export GLB. Render a lookaround video and look for holes and fragments.
3. Clean the mesh, rotate to Z-up, fix scale, put the floor at `z = 0`.
4. View it in viser to confirm the frame is right.
5. Export OBJ + texture for visuals.
6. Run `glb_to_mujoco.py`. It writes visual meshes, a floor height field, convex wall parts, and the MJCF.
7. Open `scene.xml` in the MuJoCo viewer and check the test ball lands on the floor.
8. Optional: train a splat from the photos for photoreal camera images.

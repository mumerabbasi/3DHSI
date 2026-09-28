# Paper visualization

Render GenZI, PROX, PhySIC and ours from one manually chosen MeshLab camera.
One shared square crop is applied to every method. This module writes only to
its own output directory; it does not modify evaluation renders or the paper.

## Run

Export **Windows → Save camera settings to file** from MeshLab into
`Render_Cameras/interaction_XX.xml`. Load world-coordinate scene and human meshes
in MeshLab, and use view navigation without transforming the mesh layers.

From the repository root:

```bash
python 3DHSI/00_Visualization_Paper/render_interaction.py --interaction_name interaction_18
```

The provided `interaction_18.xml` is the top-bed camera supplied by the author.
Run the same command with another interaction name after adding its XML.
Every invocation renders all four methods.

Dependencies: Python 3 with Pillow (`python -m pip install Pillow`) and Blender
4.2, available as `blender` or supplied with `--blender-bin /path/to/blender`.
The Blender worker uses Blender's bundled NumPy. No PyTorch is required.
`--gpu-index auto` selects the GPU with most free memory unless
`CUDA_VISIBLE_DEVICES` is already set. Use `--gpu-index 1` to select explicitly.

The module requires the evaluated assets already produced by modules 06–09:

- Module 06's `output/interaction_XX/semantics/assets/render_scene.blend`.
- Each method's `semantics/assets/render_config.json` and its world-space human
  mesh. For baselines these configs live under `evaluation/output/`.
- GenZI uses the **same selected candidate** recorded in its evaluation config.

Old metadata paths under `4DHSI` are resolved against the current `3DHSI` tree,
including the renamed PhySIC and PROX modules.

## Rendering and framing

The worker opens Module 06's saved Blender scene. It preserves the scene mesh,
scene colors, blue human material, lighting and world, Cycles settings,
denoising, Filmic transform, contrast, exposure and gamma. The light remains
fixed across all four methods. The original human is hidden and replaced by
each evaluated mesh. The camera and output resolution are changed; full renders
are RGB JPEGs at quality 100. `--samples` optionally overrides the inherited sample
count (normally 64).

All methods are first rendered at **4290 × 2560**, at 100% resolution. Projected
mesh vertices, including occluded vertices, determine the combined human bounds.
The automatic square crop adds **12% of the longest combined dimension on each
side** and is shared across methods. Bounds do not automatically identify the
semantic interaction object: inspect the crop and add context when necessary.

If the native square is at least 1024 pixels wide, it is downsampled to
**1024 × 1024** (or kept if already that size). Otherwise all methods are rendered
again at the smallest integer multiple of 4290 × 2560 giving at least 1024
native crop pixels. The camera FOV stays fixed, and crop coordinates scale by
that same integer. Images are never upscaled to meet the target resolution.

Use `--margin 0.15` for more context, or `--margin 0.08` for tighter framing.
These options apply to the whole interaction, never to individual methods:

```bash
# Validate inputs/camera and compute framing without rendering.
python 3DHSI/00_Visualization_Paper/render_interaction.py \
  --interaction_name interaction_18 --prepare-only

# Include an additional scene region (coordinates in the 4290x2560 full render).
python 3DHSI/00_Visualization_Paper/render_interaction.py \
  --interaction_name interaction_18 --context-box 1500 400 2900 1800

# Explicit square: left, top, side, in the same base-resolution coordinates.
python 3DHSI/00_Visualization_Paper/render_interaction.py \
  --interaction_name interaction_18 --crop-box 1450 350 1500
```

If a human is outside the camera frame, or a square cannot enclose all humans,
the script stops with an explanation. Increase the MeshLab view's coverage and
export it again. An explicit crop is also checked for clipping. Large placement
differences are retained, not hidden by independently recentering humans. A
warning is recorded if a human fills less than half the final crop's side.

## Outputs

```text
Output/interaction_18/
  genzi.jpg                 # final 1024x1024 JPEG, quality 98, no chroma subsampling
  prox.jpg
  physic.jpg
  ours.jpg
  camera.xml               # copy of the supplied MeshLab view
  render_config.json        # resolved input paths and converted camera
  inspection.json           # projection checks, body bounds and inherited style
  manifest.json             # shared crop, resolution, occupancy and completion status
  full/4290x2560/*.jpg       # full-resolution renders for all four methods
  full/8580x5120/*.jpg       # example additional pass, only when necessary
  logs/*.log
```

Repeating a command rerenders and replaces that interaction's named outputs.
The manifest's `status` becomes `complete` only after all final crops are saved.
Use it to distinguish a completed run from an interrupted attempt.

## Camera convention and checks

The converter supports text-form perspective `VCGCamera` XML with square sensor
pixels and zero lens distortion. MeshLab stores **negative camera position** in
`TranslationVector`; `RotationMatrix` is world-to-camera with OpenGL axes.
Blender receives the transpose rotation and negative translation. The exported
shot already includes trackball zoom; `TrackScale` must not be applied again.
`CenterPx` is converted from bottom-left to top-left image coordinates.

The XML's vertical FOV and principal-point offsets are preserved. For an XML
with a viewport aspect different from 4290:2560, horizontal coverage changes;
there is no image stretching. Module 06's camera clipping range (0.01–100 m) is
used; MeshLab's interactive near/far clipping is recorded but not applied.
The saved Module 06 scene crop is retained, so a new view can expose boundaries
of that crop that were not visible in the original evaluation views.

The inspection pass checks the converted projection against Blender's own
projection for vertices of all four meshes (tolerance 0.05 base-render pixels).
Source conventions: [VCGLib XML reader/writer](https://github.com/cnr-isti-vclab/vcglib/blob/main/wrap/qt/shot_qt.h),
[VCGLib camera geometry](https://github.com/cnr-isti-vclab/vcglib/blob/main/vcg/math/shot.ipp),
[MeshLab exported view](https://github.com/cnr-isti-vclab/meshlab/blob/main/src/meshlab/glarea.cpp).

```bash
python -m unittest discover -s 3DHSI/00_Visualization_Paper -p 'test_*.py'
```

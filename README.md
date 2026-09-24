# geosam2

GeoSAM2 ([Deng et al., CVPR 2026](https://arxiv.org/abs/2508.14036)) as a
library: a mesh's twelve canonical views and a 2D mask on one of them go in,
per-face labels and one mesh per part come out. The model lifts SAM2 from
images to meshes -- it propagates the mask across the views with a video
predictor and back-projects the result onto the faces.

This is [VAST's release](https://github.com/VAST-AI-Research/GeoSAM2) reshaped
the way `segvigen` is for SegviGen in pixmesh-segmentation: one class that
keeps the model loaded, the utilities around it, no Hydra, no Blender, no
sibling checkout. Around the model it adds what the paper leaves to the
user: the views rendered in-process, the seed map painted by a VLM, the
labels split into parts with SegviGen's code.

## Layout

```
geosam2/                  the package (the only top-level name)
├── segmenter.py          GeoSAM2Segmenter: ensure_checkpoint, load, run, parts, clear_vram
├── _model.py             the model built in Python, checkpoint loaded strict
├── _propagation.py       seed mask -> masks on the 12 views -> labels on the faces
├── _lift.py              the lift and the post-processes (VAST's inference_utils)
├── sam2/                 SAM2 as GeoSAM2 modified it
├── ext/mode_ext.cpp      the label vote, compiled at install time
└── util/
    ├── views.py          the 12 canonical views: rendered, validated, read back
    ├── guidance.py       the seed: view pick, VLM description, palette, painted map
    ├── labels.py         labels onto the mesh: palette, fill, parts
    ├── split.py          SegviGen's split, a copy this package owns
    └── logs.py
server.py + static/       the step-by-step app, port 7862
tests/                    unit tests, and views_check.py to compare two renders
example/                  three view directories from VAST, with reference masks
ckpt/                     geosam2.pt, downloaded from Hugging Face on first use (gitignored)
```

## Installation

Linux, Python 3.10+, a CUDA GPU, PyTorch 2.3+ already in the environment.

```bash
pip install -r requirements.txt
pip install --no-deps 'pyrender>=0.1.45'   # its PyOpenGL pin is stale; see requirements.txt
pip install -e . --no-build-isolation      # compiles geosam2/ext/mode_ext.cpp
cp .env.dist .env                          # GEMINI_API_KEY for the VLM stage
```

Without the compiled extension the vote is built on first import instead,
which needs a C++ compiler and `ninja` on the PATH. The checkpoint
(`ckpt/geosam2.pt`, 615 MB) is downloaded from
[VAST-AI/GeoSAM2](https://huggingface.co/VAST-AI/GeoSAM2) the first time it
is needed.

## Usage

```python
from geosam2.segmenter import GeoSAM2Segmenter
from geosam2.util import views, guidance, labels, split

work = "runs/sideboard"
views.render_views("sideboard.glb", f"{work}/views")            # 12 views + meta.json + mesh.glb
view = guidance.pick_seed_view(f"{work}/views")                  # VLM picks the seed view
seed = guidance.generate_seed(f"{work}/views", view)             # VLM describes, paints; mask_XXXX.png

seg = GeoSAM2Segmenter()                                         # loads on the first run(), stays loaded
parts_glb = seg.run(f"{work}/views", seed.map_path, view, f"{work}/out")
# -> out/parts.glb (one geometry per part, the input's frame), parts.npy, labels_raw.npy, labels_post.npy

split.split_glb_by_face_labels(f"{work}/views/mesh.glb", np.load(f"{work}/out/parts.npy"), f"{work}/split.glb")
seg.clear_vram()
```

`run()` takes any 2D mask on one of the views: a label map (`.npy`, `.exr`)
or a flat-colour image such as the painted seed. `postprocess_pa` (0.02) is
the one knob VAST documents; `GeoSAM2Segmenter.POSTPROCESS_PA_CANDIDATES`
lists the values worth trying.

What the process-wide settings of VAST's script did -- bf16 autocast, TF32,
the seeds -- is scoped to each `run()`, so a host process is left as it was,
and two consecutive runs give the same labels.

## The app

```bash
python app.py        # http://127.0.0.1:7862
```

The stages one at a time -- render, pick the view, guidance, GeoSAM2, fill,
split -- each a job on file paths, so a stage can be re-run on its own and
its result compared with the previous one in the viewer. The bundled
`example/sample_*` skip the render.

## Environment

`.env` at the repository root, read by the app (the library reads
`os.environ` only): `GEMINI_API_KEY`, and optionally `GEOSAM2_DESCRIBE_MODEL`
/ `GEOSAM2_PAINT_MODEL` (pydantic-ai model names), `GEOSAM2_LOG_LEVEL`,
`GEOSAM2_APP_HOST` / `GEOSAM2_APP_PORT`.

## Acknowledgements

GeoSAM2 builds on the following open-source projects, which we gratefully
acknowledge:

- [facebookresearch/sam2](https://github.com/facebookresearch/sam2)
- [bpy-renderer](https://github.com/huanngzh/bpy-renderer)
- [Samesh](https://github.com/gtangg12/samesh)
- [PartField](https://github.com/nv-tlabs/PartField)

## License

This project is released under the Apache License 2.0; see [LICENSE](LICENSE).
`geosam2/sam2/` is derived from
[Meta's SAM2](https://github.com/facebookresearch/sam2) under the same license;
`geosam2/util/guidance.py` and `geosam2/util/split.py` are copies of pixmesh's
SegviGen code. See [NOTICE](NOTICE) for details.

## Citation

If you find GeoSAM2 useful in your research, please cite:

```bibtex
@article{deng2025geosam2,
  title   = {GeoSAM2: Unleashing the Power of SAM2 for 3D Part Segmentation},
  author  = {Deng, Ken and Yang, Yunhan and Sun, Jingxiang and
             Liu, Xihui and Liu, Yebin and Liang, Ding and Cao, Yan-Pei},
  journal = {arXiv preprint arXiv:2508.14036},
  year    = {2025}
}
```

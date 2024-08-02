# FLAME PyTorch

PyTorch implementation of **FLAME** (Faces Learned with an Articulated Model and Expressions) — a lightweight 3D head model that maps shape, expression, and pose parameters to a mesh and 3D facial landmarks.

Designed as a differentiable `nn.Module`, so it can be used as a decoder inside deep learning pipelines.

## Features

- Differentiable FLAME layer (`flame_pytorch.FLAME`)
- Shape, expression, and pose parameter controls
- 3D facial landmark prediction (static + optional dynamic contour)
- Demo script for mesh and landmark visualization

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
python setup.py install
mkdir model
```

Place the FLAME model weights and landmark embedding files in the `model/` directory (see `flame_pytorch/config.py` for expected paths).

## Demo

```bash
python main.py
```

This loads the FLAME layer, samples a batch of head poses, and opens a viewer for the mesh and landmarks.

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

Note: use of the FLAME model assets themselves is subject to the [FLAME model license](https://flame.is.tue.mpg.de/modellicense.html).

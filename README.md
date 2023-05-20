# FLAME: Articulated Expressive 3D Head Model (PyTorch)

This is an implementation of the [FLAME](http://flame.is.tue.mpg.de/) 3D head model in PyTorch.

We also provide [Tensorflow FLAME](https://github.com/TimoBolkart/TF_FLAME), a [Chumpy](https://github.com/mattloper/chumpy)-based [FLAME-fitting repository](https://github.com/Rubikplayer/flame-fitting), and code to [convert from Basel Face Model to FLAME](https://github.com/TimoBolkart/BFM_to_FLAME).

<p align="center"> 
<img src="gifs/model_variations.gif">
</p>

FLAME is a lightweight and expressive generic head model learned from over 33,000 of accurately aligned 3D scans. FLAME combines a linear identity shape space (trained from head scans of 3800 subjects) with an articulated neck, jaw, and eyeballs, pose-dependent corrective blendshapes, and additional global expression blendshapes. For details please see the following [scientific publication](https://ps.is.tuebingen.mpg.de/uploads_file/attachment/attachment/400/paper.pdf)

```bibtex
Learning a model of facial shape and expression from 4D scans
Tianye Li*, Timo Bolkart*, Michael J. Black, Hao Li, and Javier Romero
ACM Transactions on Graphics (Proc. SIGGRAPH Asia) 2017
```


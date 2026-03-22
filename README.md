<p align="center">
  <img src="https://github.com/dimdano/aie4ml/blob/main/docs/aie4ml_logo_big.png" alt="aie4ml" width="600"/>
</p>

[![License](https://img.shields.io/badge/License-Apache_2.0-red.svg)](https://opensource.org/licenses/Apache-2.0)
[![PyPI](https://img.shields.io/pypi/v/aie4ml.svg)](https://pypi.org/project/aie4ml/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/aie4ml.svg)](https://pypi.org/project/aie4ml/)
[![arXiv](https://img.shields.io/badge/arXiv-2512.15946-b31b1b.svg)](https://arxiv.org/abs/2512.15946)

`aie4ml` is an end-to-end compiler that generates **optimized** AIE firmware automatically, which can be then built and simulated directly using **AMD Vitis**. Currently, it is developed as a plugin that extends the backends for [`hls4ml`](https://github.com/fastmachinelearning/hls4ml) in order to target the **AMD AI Engine (AIE)**.

- Current support: dense (linear) layers with optional bias & ReLU. Support for AIE-ML/AIE-MLv2 devices.

## Prerequisites

- AMD Vitis 2025.2 and a valid AIE tools license.
- Python 3.10+ and the latest version of [`hls4ml`](https://github.com/fastmachinelearning/hls4ml) package.

# Frontend Compatibility

Operates on the intermediate model representation produced by hls4ml, therefore independent of the frontend (i.e., PyTorch, QKeras, etc.).

# Installation

```bash
pip install hls4ml
pip install aie4ml
```

# Documentation & Tutorials

Documentation and usage: [https://github.com/dimdano/aie4ml](https://github.com/dimdano/aie4ml)

Tutorial 1 : [`tutorials/tutorial_1.ipynb`](tutorials/tutorial_1.ipynb)
Tutorial 2 : [`tutorials/tutorial_2.ipynb`](tutorials/tutorial_2.ipynb)

General `hls4ml` concepts: [https://fastmachinelearning.org/hls4ml](https://fastmachinelearning.org/hls4ml)


## Maintainer

`aie4ml` is developed and maintained by [Dimitrios Danopoulos](https://github.com/dimdano).

## Citation

If `aie4ml` contributes to your research, please cite the corresponding arXiv preprint:

```bibtex
@misc{danopoulos2025aie4mlendtoendframeworkcompiling,
      title={AIE4ML: An End-to-End Framework for Compiling Neural Networks for the Next Generation of AMD AI Engines},
      author={Dimitrios Danopoulos and Enrico Lupi and Chang Sun and Sebastian Dittmeier and Michael Kagan and Vladimir Loncar and Maurizio Pierini},
      year={2025},
      eprint={2512.15946},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2512.15946},
}
```

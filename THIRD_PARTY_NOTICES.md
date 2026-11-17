# Third-party notices

## lanl/OpenFWI

WaveDL contains code adapted from [lanl/OpenFWI](https://github.com/lanl/OpenFWI)
at pinned commit `48754806b7b4c5877259c6b958a87f4513fcdc0b`.
The upstream software is licensed under the BSD 3-Clause License reproduced below.
Credit goes to OmniWave Lab (Youzuo Lin's group, UNC–Chapel Hill) and Los Alamos
National Laboratory (LANL).

Files containing adapted code:

- `src/wavedl/models/openfwi.py`: InversionNet, VelocityGAN discriminator,
  convolution/deconvolution blocks, and legacy checkpoint key conversion adapted
  from upstream `network.py`. WaveDL adds registration, fixed-geometry contracts,
  documentation, type hints, and safe checkpoint loading.
- `src/wavedl/utils/openfwi.py`: reimplementations of OpenFWI's signed-log/min-max
  transforms and SSIM (whose original pytorch-ssim attribution is reproduced below).
- `src/wavedl/utils/gan.py`: WGAN-GP interpolation, losses and update order adapted
  from upstream `utils.py` and `gan_train.py`.
- `src/wavedl/dataset_config.json`: upstream benchmark preprocessing bounds.

### Upstream Triad/LANL source header

The following header is retained verbatim in the adapted model source:

```text
# © 2022. Triad National Security, LLC. All rights reserved.

# This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos

# National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S.

# Department of Energy/National Nuclear Security Administration. All rights in the program are

# reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear

# Security Administration. The Government is granted for itself and others acting on its behalf a

# nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare

# derivative works, distribute copies to the public, perform publicly and display publicly, and to permit

# others to do so.
```

### BSD 3-Clause License

Source: <https://raw.githubusercontent.com/lanl/OpenFWI/48754806b7b4c5877259c6b958a87f4513fcdc0b/LICENSE>.

```text
BSD 3-Clause License

Copyright (c) 2022, Los Alamos National Laboratory
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## pytorch-ssim

`src/wavedl/utils/openfwi.py` contains a reimplementation of SSIM matching
OpenFWI's `pytorch_ssim.py`, which explicitly credits
<https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py>.
Original author: **Po-Hsun (Evan) Su** (as identified in upstream `setup.py`).

Upstream license source:
<https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/LICENSE.txt>.
Verified on 2026-11-12: that file's complete notice text is simply:

```text
MIT
```

The standard MIT permission and disclaimer terms are included here for clarity
(the upstream file supplies no copyright year or expanded notice):

```text
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

## Research references

- Wu & Lin (2020), *InversionNet*, IEEE Transactions on Computational Imaging
  6:419-433. DOI: <https://doi.org/10.1109/TCI.2019.2956866>.
- Zhang & Lin (2020), *VelocityGAN*, IEEE Transactions on Geoscience and Remote
  Sensing 58(10):6900-6913. DOI: <https://doi.org/10.1109/TGRS.2020.2977635>.
- Deng et al. (2022), *OpenFWI*, NeurIPS Datasets & Benchmarks.

### Dataset license (separate from software)

OpenFWI datasets are licensed under **Creative Commons
Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)**,
<https://creativecommons.org/licenses/by-nc-sa/4.0/>. They are **not redistributed**
with WaveDL. Users obtain the datasets separately and must comply with their
license; the BSD-3-Clause software license does not replace the dataset license.

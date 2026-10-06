# Third-party source provenance

This repository retains modified third-party source as ordinary tracked files.
Project Git revisions describe the integrated project, not the upstream release.

| Directory | Donor | Source revision record | Notices |
| --- | --- | --- | --- |
| `third_party/DMM` | https://github.com/cong-yi/DMM | `third_party/DMM/UPSTREAM.json` | Original README and attribution retained. No standalone top-level LICENSE was present in this local donor checkout. No new license is assigned to that code by this repository. |
| `third_party/DMM/third_party/torchmeta` | Bundled with the local DMM tree | Preserved bundled source | Retain existing file headers and notices. |
| `third_party/nvdiffrast` | https://github.com/NVlabs/nvdiffrast | `third_party/nvdiffrast/UPSTREAM.json` | See its `LICENSE.txt` (Nvidia Source Code License) and source headers. |

No dataset, pretrained model, paper PDF, compiled extension, or local Python/CUDA
environment is distributed in this Git repository. Dataset and model permissions
are separate from source provenance. This local repository does not assign a
blanket open-source license to third-party materials.

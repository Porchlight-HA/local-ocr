# Third-party notices

Our code (the `local_ocr_server` package, the App files and the integration) is MIT licensed; see `LICENSE`. The App image also contains third-party components, each under its own licence. The App ships no restricted-weight models.

## Direct dependencies

| Component | Licence | Use |
|---|---|---|
| RapidOCR (`rapidocr`) | Apache-2.0 | OCR pipeline |
| PP-OCR ONNX models (PaddlePaddle / PaddleOCR, shipped in `rapidocr`) | Apache-2.0 | Detection, classification, recognition models |
| onnxruntime | MIT | Inference runtime |
| OpenCV (`opencv-python-headless`) | Apache-2.0 | Image resizing |
| NumPy | BSD-3-Clause (bundled parts: 0BSD, MIT, Zlib, CC0-1.0; the wheel also bundles OpenBLAS, BSD-3-Clause, and libgfortran, GPL-3.0 with the GCC Runtime Library Exception) | Arrays |
| Pillow | MIT-CMU (HPND) | Image decoding |
| aiohttp | Apache-2.0 (parts MIT) | HTTP server and client |
| zeroconf (`python-zeroconf`) | LGPL-2.1-or-later | mDNS advertising in standalone mode |

The `opencv-python-headless` wheel bundles FFmpeg libraries (`libavcodec`, `libavformat`, `libavutil` and others) under the LGPL-2.1-or-later, plus other libraries listed in its `LICENSE-3RD-PARTY.txt`. They are unmodified, dynamically loaded shared libraries.

Other Python packages pulled in by the above (versions as built for 0.1): aiohappyeyeballs (PSF-2.0), aiosignal, frozenlist, multidict, propcache, yarl, flatbuffers, requests (Apache-2.0); attrs, charset-normalizer, colorlog, ifaddr, pyclipper, PyYAML, six, urllib3 (MIT); idna, protobuf, Shapely, omegaconf (BSD-3-Clause); packaging (Apache-2.0 or BSD-2-Clause); certifi (MPL-2.0); tqdm (MPL-2.0 and MIT).

## Base image

The App is built on `ghcr.io/home-assistant/{arch}-base-debian:trixie` and installs Debian's `python3`, `python3-venv` and `libgomp1`. The Debian system packages in the image (for example bash, coreutils and the GCC runtime libraries) are distributed under their own licences, some of them GPL; their copyright files are in `/usr/share/doc/<package>/copyright` inside the image. The s6-overlay init system from the base image is ISC licensed.

## Test fixture fonts (not shipped in the App)

`fixtures/fonts/` holds fonts used only by `fixtures/gen.py` to render synthetic test images.
Each came from a Debian package; the package copyright file sits beside it as `LICENSE.<package>`.

| Font | Debian package | Licence |
|---|---|---|
| DejaVu Sans, DejaVu Sans Bold | fonts-dejavu-core | Bitstream Vera / DejaVu licence |
| Liberation Serif, Mono, Sans Bold | fonts-liberation | SIL OFL 1.1 |
| Comic Neue Bold | fonts-comic-neue | SIL OFL 1.1 |
| Dancing Script Bold | fonts-dancingscript | SIL OFL 1.1 |
| Kristi | fonts-kristi | SIL OFL 1.1 |

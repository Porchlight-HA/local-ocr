# Third-party notices

Local OCR bundles or depends on the following. None are GPL or restricted-weight.

| Component | Licence | Use |
|---|---|---|
| RapidOCR (`rapidocr`) | Apache-2.0 | OCR pipeline |
| PP-OCR ONNX models (PaddlePaddle / PaddleOCR, shipped in `rapidocr`) | Apache-2.0 | Detection, classification, recognition models |
| onnxruntime | MIT | Inference runtime |
| OpenCV (`opencv-python-headless`) | Apache-2.0 | Image resizing |
| NumPy | BSD-3-Clause | Arrays |

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

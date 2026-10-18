# Changelog

## 0.3.1

- Installs a prebuilt image (`ghcr.io/porchlight-ha/{arch}-local-ocr`) instead of building on your device, so installing takes seconds of download instead of minutes of building. No other changes.

## 0.3.0

- The App now announces itself to Home Assistant as `text_reader`: the integration was renamed from Local OCR to Text Reader. Update the integration to 0.3.0 along with the App. If you used the 0.2 "Local OCR" integration, remove it and add Text Reader (the domain changed; there is no migration).
- The App itself is unchanged and keeps its name, slug and API (v1). Text Reader 0.3 can also pair it with an AI assistant, or fall back to one for hard images.

## 0.2.0

- Reading every frame of a large phone photo (MPO) works; it used to fail with "could not decode the image".
- At most 16 multipart parts and one `image` part per request.
- Zeroconf no longer advertises self-assigned (169.254.x.x) addresses.
- `label`: digits are joined only in groups of four, so a ZIP code next to a phone number is no longer read as a FedEx number. Spaced UPS numbers are not matched (documented).
- The image carries OCI labels. Docs: the watchdog needs the App's Watchdog toggle; published images will be per architecture.

## 0.1.0

- First private release: offline OCR server (RapidOCR on onnxruntime, PP-OCR models baked in) with the `text`, `envelope`, `meter` and `label` presets.
- Announces itself to Home Assistant through Supervisor discovery.
- Builds locally from the App folder (no published image yet).

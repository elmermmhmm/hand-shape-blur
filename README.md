# Finger Lens

A camera-based web experience that tracks your hands and carves a live
pixelated/blurred "lens" out of the video feed, shaped by your fingertips.

## How it works

- The webcam feed is drawn (mirrored) to a full-screen canvas.
- [MediaPipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)
  tracks up to two hands and 21 landmarks per hand, fully in-browser.
- Extended fingertips are detected and counted; the count picks the shape,
  and the fingertip positions define it:

  | Fingers | Shape |
  | ------- | ----- |
  | 1       | small circle on the fingertip |
  | 2       | circle spanning the two tips |
  | 3       | triangle through the tips |
  | 4–9     | n-sided polygon through the tips |
  | 10      | circle enclosing all tips (both hands) |

- Inside the shape, the frame is downsampled and re-upscaled with smoothing
  disabled (pixelation), with a CSS-filter blur layered on top. Pixel size
  and blur amount are adjustable with the on-screen sliders.

Everything runs locally in the browser — no video ever leaves your device.

## Run it

The page needs to be served over HTTP(S) (camera access + ES modules don't
work from `file://`):

```sh
npx serve .
# or
python3 -m http.server 8000
```

Then open the printed URL, click **Enable camera**, and hold up some fingers.

## Stack

- Vanilla HTML/CSS/JS (ES modules), no build step
- `@mediapipe/tasks-vision` via CDN

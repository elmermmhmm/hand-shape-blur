import {
  HandLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

const startScreen = document.getElementById("start-screen");
const startBtn = document.getElementById("start-btn");
const statusEl = document.getElementById("status");
const hud = document.getElementById("hud");
const legend = document.getElementById("legend");
const controls = document.getElementById("controls");
const shapeNameEl = document.getElementById("shape-name");
const fingerCountEl = document.getElementById("finger-count");
const pixelSlider = document.getElementById("pixel-size");
const blurSlider = document.getElementById("blur-amount");

// Offscreen canvas used to downsample the frame for the pixelation pass.
const pixCanvas = document.createElement("canvas");
const pixCtx = pixCanvas.getContext("2d");

let landmarker = null;
let lastVideoTime = -1;
let latestResults = null;

// Smoothed fingertip positions keyed by "hand:tipIndex" to tame jitter.
const smoothedTips = new Map();
const SMOOTHING = 0.45;

const FINGERS = [
  { name: "thumb", tip: 4, pip: 2 },
  { name: "index", tip: 8, pip: 6 },
  { name: "middle", tip: 12, pip: 10 },
  { name: "ring", tip: 16, pip: 14 },
  { name: "pinky", tip: 20, pip: 18 },
];

const SHAPE_NAMES = {
  0: "—",
  1: "Dot",
  2: "Circle",
  3: "Triangle",
  4: "Rectangle",
  5: "Circle",
};

startBtn.addEventListener("click", start);

async function start() {
  startBtn.disabled = true;
  statusEl.textContent = "Loading hand-tracking model…";
  try {
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
    );
    landmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        delegate: "GPU",
      },
      runningMode: "VIDEO",
      numHands: 2,
    });

    statusEl.textContent = "Requesting camera…";
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    startScreen.hidden = true;
    hud.hidden = false;
    legend.hidden = false;
    controls.hidden = false;

    requestAnimationFrame(loop);
  } catch (err) {
    startBtn.disabled = false;
    statusEl.textContent =
      err.name === "NotAllowedError"
        ? "Camera permission denied — allow access and try again."
        : `Could not start: ${err.message}`;
  }
}

function loop() {
  if (video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    latestResults = landmarker.detectForVideo(video, performance.now());
  }
  render();
  requestAnimationFrame(loop);
}

function dist(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

// A finger counts as extended when its tip is meaningfully farther from the
// wrist than its lower joint. The thumb instead measures distance from the
// pinky base, since it folds sideways across the palm.
function extendedTips(landmarks) {
  const wrist = landmarks[0];
  const pinkyBase = landmarks[17];
  const tips = [];
  for (const f of FINGERS) {
    const tip = landmarks[f.tip];
    const pip = landmarks[f.pip];
    const isExtended =
      f.name === "thumb"
        ? dist(tip, pinkyBase) > dist(pip, pinkyBase) * 1.15
        : dist(tip, wrist) > dist(pip, wrist) * 1.1;
    if (isExtended) tips.push({ finger: f.name, x: tip.x, y: tip.y });
  }
  return tips;
}

function smoothTip(key, tip) {
  const prev = smoothedTips.get(key);
  const next = prev
    ? {
        x: prev.x + (tip.x - prev.x) * SMOOTHING,
        y: prev.y + (tip.y - prev.y) * SMOOTHING,
      }
    : { x: tip.x, y: tip.y };
  smoothedTips.set(key, next);
  return next;
}

// Collect extended fingertips across all detected hands, mirrored to match
// the selfie-flipped canvas, in pixel coordinates.
function collectTips() {
  const tips = [];
  const seen = new Set();
  const hands = latestResults?.landmarks ?? [];
  hands.forEach((landmarks, handIdx) => {
    for (const tip of extendedTips(landmarks)) {
      const key = `${handIdx}:${tip.finger}`;
      seen.add(key);
      const s = smoothTip(key, tip);
      tips.push({ x: (1 - s.x) * canvas.width, y: s.y * canvas.height });
    }
  });
  for (const key of smoothedTips.keys()) {
    if (!seen.has(key)) smoothedTips.delete(key);
  }
  return tips;
}

function sortByAngle(points) {
  const cx = points.reduce((s, p) => s + p.x, 0) / points.length;
  const cy = points.reduce((s, p) => s + p.y, 0) / points.length;
  return [...points].sort(
    (a, b) => Math.atan2(a.y - cy, a.x - cx) - Math.atan2(b.y - cy, b.x - cx)
  );
}

// Build the clip path the fingers describe:
//   1 tip  → small circle on the fingertip
//   2 tips → circle whose diameter spans the two tips
//   3–4    → polygon through the tips (triangle / rectangle)
//   5+     → circle enclosing every tip
function shapeFromTips(tips) {
  const path = new Path2D();
  const n = tips.length;

  if (n === 1) {
    path.arc(tips[0].x, tips[0].y, canvas.height * 0.07, 0, Math.PI * 2);
  } else if (n === 2) {
    const cx = (tips[0].x + tips[1].x) / 2;
    const cy = (tips[0].y + tips[1].y) / 2;
    const r = Math.max(dist(tips[0], tips[1]) / 2, 20);
    path.arc(cx, cy, r, 0, Math.PI * 2);
  } else if (n === 3 || n === 4) {
    const pts = sortByAngle(tips);
    path.moveTo(pts[0].x, pts[0].y);
    for (let i = 1; i < pts.length; i++) path.lineTo(pts[i].x, pts[i].y);
    path.closePath();
  } else {
    const cx = tips.reduce((s, p) => s + p.x, 0) / n;
    const cy = tips.reduce((s, p) => s + p.y, 0) / n;
    const r = Math.max(...tips.map((p) => dist(p, { x: cx, y: cy }))) + 16;
    path.arc(cx, cy, r, 0, Math.PI * 2);
  }
  return path;
}

function drawMirroredVideo(target, w, h) {
  target.save();
  target.scale(-1, 1);
  target.drawImage(video, -w, 0, w, h);
  target.restore();
}

function render() {
  const w = canvas.width;
  const h = canvas.height;
  if (!w || !h) return;

  drawMirroredVideo(ctx, w, h);

  const tips = collectTips();
  const n = tips.length;
  shapeNameEl.textContent = SHAPE_NAMES[n] ?? "Polygon";
  fingerCountEl.textContent = n
    ? `${n} finger${n > 1 ? "s" : ""}`
    : "show your hand";

  if (n === 0) return;

  const path = shapeFromTips(tips);

  // Pixelation pass: downsample the frame, then scale it back up inside the
  // clip with smoothing off, adding a blur on top.
  const pixelSize = Number(pixelSlider.value);
  const blurPx = Number(blurSlider.value);
  pixCanvas.width = Math.max(2, Math.round(w / pixelSize));
  pixCanvas.height = Math.max(2, Math.round(h / pixelSize));
  drawMirroredVideo(pixCtx, pixCanvas.width, pixCanvas.height);

  ctx.save();
  ctx.clip(path);
  ctx.imageSmoothingEnabled = false;
  if (blurPx > 0) ctx.filter = `blur(${blurPx}px)`;
  ctx.drawImage(pixCanvas, 0, 0, w, h);
  ctx.restore();

  ctx.save();
  ctx.strokeStyle = "rgba(125, 249, 194, 0.9)";
  ctx.lineWidth = 3;
  ctx.shadowColor = "rgba(125, 249, 194, 0.8)";
  ctx.shadowBlur = 14;
  ctx.stroke(path);
  ctx.restore();

  for (const tip of tips) {
    ctx.beginPath();
    ctx.arc(tip.x, tip.y, 6, 0, Math.PI * 2);
    ctx.fillStyle = "#7df9c2";
    ctx.fill();
  }
}

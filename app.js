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

const pixCanvas = document.createElement("canvas");
const pixCtx = pixCanvas.getContext("2d");

let landmarker = null;
let lastVideoTime = -1;
let latestResults = null;

const smoothedTips = new Map();
const blobMeshes = new Map();
const handFlipState = new Map();

const TIP_SMOOTHING = 0.28;
const VERTEX_SMOOTHING = 0.32;
const FLIP_SMOOTHING = 0.18;
const CORNER_ROUND_RATIO = 0.06;
const MIN_CORNER_RADIUS = 5;
const LINE_WIDTH_RATIO = 0.022;

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
  2: "Line",
  3: "Triangle",
  4: "Quadrilateral",
  5: "Pentagon",
  6: "Hexagon",
  7: "Heptagon",
  8: "Octagon",
  9: "Nonagon",
  10: "Decagon",
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

function centroid(points) {
  const n = points.length;
  if (!n) return { x: 0, y: 0 };
  return {
    x: points.reduce((s, p) => s + p.x, 0) / n,
    y: points.reduce((s, p) => s + p.y, 0) / n,
  };
}

function normalize(v) {
  const len = Math.hypot(v.x, v.y) || 1;
  return { x: v.x / len, y: v.y / len };
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function lerpPoint(a, b, t) {
  return { x: lerp(a.x, b.x, t), y: lerp(a.y, b.y, t) };
}

function smoothToward(current, target, amount) {
  return current + (target - current) * amount;
}

function shapeLabel(count) {
  return SHAPE_NAMES[count] ?? `${count}-gon`;
}

function lineWidth() {
  return Math.max(8, canvas.height * LINE_WIDTH_RATIO);
}

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
    if (isExtended) {
      tips.push({ finger: f.name, x: tip.x, y: tip.y, z: tip.z ?? 0 });
    }
  }
  return tips;
}

function detectHandFlipTarget(landmarks, worldLandmarks, handedness) {
  const pts = worldLandmarks ?? landmarks;
  const wrist = pts[0];
  const indexMcp = pts[5];
  const pinkyMcp = pts[17];
  const middleTip = pts[12];

  const v1 = {
    x: indexMcp.x - wrist.x,
    y: indexMcp.y - wrist.y,
    z: indexMcp.z - wrist.z,
  };
  const v2 = {
    x: pinkyMcp.x - wrist.x,
    y: pinkyMcp.y - wrist.y,
    z: pinkyMcp.z - wrist.z,
  };
  const nz = v1.x * v2.y - v1.y * v2.x;

  const label = handedness?.categoryName ?? handedness?.[0]?.categoryName ?? "Right";
  const isRight = label === "Right";
  const palmFacing = isRight ? nz > 0 : nz < 0;

  const depthHint = (middleTip.z ?? landmarks[12].z ?? 0) > (wrist.z ?? landmarks[0].z ?? 0);
  const dorsal = palmFacing ? 0 : 1;
  const blended = depthHint ? Math.max(dorsal, 0.65) : dorsal;
  return blended > 0.5 ? 1 : 0;
}

function smoothHandFlip(handIdx, target) {
  const prev = handFlipState.get(handIdx) ?? target;
  const next = smoothToward(prev, target, FLIP_SMOOTHING);
  handFlipState.set(handIdx, next);
  return next;
}

function smoothTip(key, tip) {
  const prev = smoothedTips.get(key);
  const next = prev
    ? {
        x: smoothToward(prev.x, tip.x, TIP_SMOOTHING),
        y: smoothToward(prev.y, tip.y, TIP_SMOOTHING),
        z: smoothToward(prev.z, tip.z ?? 0, TIP_SMOOTHING),
      }
    : { x: tip.x, y: tip.y, z: tip.z ?? 0 };
  smoothedTips.set(key, next);
  return next;
}

function toScreen(point) {
  return {
    x: (1 - point.x) * canvas.width,
    y: point.y * canvas.height,
    z: point.z ?? 0,
  };
}

function collectTipsByHand() {
  const groups = [];
  const seen = new Set();
  const hands = latestResults?.landmarks ?? [];
  const worldHands = latestResults?.worldLandmarks ?? [];
  const handednessList = latestResults?.handedness ?? [];

  hands.forEach((landmarks, handIdx) => {
    const tips = [];
    for (const tip of extendedTips(landmarks)) {
      const key = `${handIdx}:${tip.finger}`;
      seen.add(key);
      const s = smoothTip(key, tip);
      tips.push(toScreen(s));
    }
    if (!tips.length) return;

    const flipTarget = detectHandFlipTarget(
      landmarks,
      worldHands[handIdx],
      handednessList[handIdx]
    );
    const flip = smoothHandFlip(handIdx, flipTarget);
    groups.push({ handIdx, tips, flip });
  });

  for (const key of smoothedTips.keys()) {
    if (!seen.has(key)) smoothedTips.delete(key);
  }
  const activeHands = new Set(groups.map((g) => g.handIdx));
  for (const key of handFlipState.keys()) {
    if (!activeHands.has(key)) handFlipState.delete(key);
  }

  return groups;
}

function sortByAngle(points) {
  const c = centroid(points);
  return [...points].sort(
    (a, b) => Math.atan2(a.y - c.y, a.x - c.x) - Math.atan2(b.y - c.y, b.x - c.x)
  );
}

function orientPoints(points, flip) {
  const sorted = sortByAngle(points);
  if (flip < 0.5) return sorted;
  return [...sorted].reverse();
}

function orderLineTips(tips, flip) {
  const sorted = [...tips].sort((a, b) => a.x - b.x || a.y - b.y);
  if (flip < 0.5) return sorted;
  return [sorted[1], sorted[0]];
}

function averageFlip(groups) {
  if (!groups.length) return 0;
  return groups.reduce((s, g) => s + (g.flip ?? 0), 0) / groups.length;
}

function collectAllTips(handGroups) {
  const tips = handGroups.flatMap((g) => g.tips);
  return { tips, flip: averageFlip(handGroups) };
}

function getTargetPoints(tips, flip) {
  if (tips.length === 2) return orderLineTips(tips, flip);
  return orientPoints(tips, flip);
}

function spawnVertex(existing, targetCount) {
  if (existing.length === 2 && targetCount === 3) {
    return {
      x: (existing[0].x + existing[1].x) / 2,
      y: (existing[0].y + existing[1].y) / 2,
    };
  }
  const c = centroid(existing);
  return { x: c.x, y: c.y };
}

function remapVertices(vertices, targetCount) {
  if (vertices.length === targetCount) return vertices;
  if (vertices.length < targetCount) {
    const next = [...vertices];
    while (next.length < targetCount) next.push(spawnVertex(next, targetCount));
    return next;
  }
  if (targetCount === 2) {
    let best = [0, 1];
    let bestDist = 0;
    for (let i = 0; i < vertices.length; i++) {
      for (let j = i + 1; j < vertices.length; j++) {
        const d = dist(vertices[i], vertices[j]);
        if (d > bestDist) {
          bestDist = d;
          best = [i, j];
        }
      }
    }
    return [vertices[best[0]], vertices[best[1]]];
  }
  const c = centroid(vertices);
  const scored = vertices.map((v, i) => ({ v, i, d: dist(v, c) }));
  scored.sort((a, b) => a.d - b.d);
  const keep = new Set(scored.slice(0, targetCount).map((s) => s.i));
  return vertices.filter((_, i) => keep.has(i));
}

function seedVerticesFromMesh(mesh, targetPoints) {
  if (mesh?.kind === "circle") {
    const c = { x: mesh.cx, y: mesh.cy };
    return targetPoints.map((p) => ({
      x: lerp(c.x, p.x, 0.5),
      y: lerp(c.y, p.y, 0.5),
    }));
  }
  return targetPoints.map((p) => ({ x: p.x, y: p.y }));
}

function syncMorphMesh(targetPoints) {
  const id = "shape";
  let mesh = blobMeshes.get(id);
  const n = targetPoints.length;

  if (!mesh || mesh.kind !== "morph") {
    mesh = {
      kind: "morph",
      vertices: seedVerticesFromMesh(mesh, targetPoints),
      lineWidth: n === 2 ? lineWidth() : 0,
    };
  } else {
    mesh.vertices = remapVertices(mesh.vertices, n);
    for (let i = 0; i < n; i++) {
      mesh.vertices[i] = lerpPoint(mesh.vertices[i], targetPoints[i], VERTEX_SMOOTHING);
    }
    mesh.lineWidth = smoothToward(mesh.lineWidth ?? 0, n === 2 ? lineWidth() : 0, VERTEX_SMOOTHING);
  }

  blobMeshes.set(id, mesh);
  return mesh;
}

function syncCircleMesh(spec) {
  const id = "shape";
  let mesh = blobMeshes.get(id);
  if (!mesh || mesh.kind !== "circle") {
    mesh = { kind: "circle", cx: spec.cx, cy: spec.cy, r: spec.r };
  } else {
    mesh.cx = smoothToward(mesh.cx, spec.cx, VERTEX_SMOOTHING);
    mesh.cy = smoothToward(mesh.cy, spec.cy, VERTEX_SMOOTHING);
    mesh.r = smoothToward(mesh.r, spec.r, VERTEX_SMOOTHING);
  }
  blobMeshes.set(id, mesh);
  return mesh;
}

function cornerRadiusFor(points) {
  if (points.length < 3) return 0;
  const edges = points.map((p, i) => dist(p, points[(i + 1) % points.length]));
  const avg = edges.reduce((s, e) => s + e, 0) / edges.length;
  return Math.min(MIN_CORNER_RADIUS, avg * CORNER_ROUND_RATIO);
}

function addCapsulePath(path, a, b, halfWidth) {
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const len = Math.hypot(dx, dy);
  if (len < 0.001) {
    path.arc(a.x, a.y, halfWidth, 0, Math.PI * 2);
    return;
  }

  const ux = dx / len;
  const uy = dy / len;
  const px = -uy * halfWidth;
  const py = ux * halfWidth;

  const a1 = { x: a.x + px, y: a.y + py };
  const a2 = { x: a.x - px, y: a.y - py };
  const b1 = { x: b.x + px, y: b.y + py };
  const b2 = { x: b.x - px, y: b.y - py };

  path.moveTo(a1.x, a1.y);
  path.lineTo(b1.x, b1.y);
  path.arc(b.x, b.y, halfWidth, Math.atan2(py, px), Math.atan2(-py, -px));
  path.lineTo(a2.x, a2.y);
  path.arc(a.x, a.y, halfWidth, Math.atan2(-py, -px), Math.atan2(py, px));
  path.closePath();
}

function addPolygonPath(path, points, radius) {
  const n = points.length;
  if (n < 3) return;
  if (radius < 1) {
    path.moveTo(points[0].x, points[0].y);
    for (let i = 1; i < n; i++) path.lineTo(points[i].x, points[i].y);
    path.closePath();
    return;
  }

  for (let i = 0; i < n; i++) {
    const prev = points[(i - 1 + n) % n];
    const curr = points[i];
    const next = points[(i + 1) % n];

    const inVec = normalize({ x: prev.x - curr.x, y: prev.y - curr.y });
    const outVec = normalize({ x: next.x - curr.x, y: next.y - curr.y });
    const inLen = dist(prev, curr);
    const outLen = dist(next, curr);
    const r = Math.min(radius, inLen * 0.3, outLen * 0.3);

    const start = { x: curr.x + inVec.x * r, y: curr.y + inVec.y * r };
    const end = { x: curr.x + outVec.x * r, y: curr.y + outVec.y * r };

    if (i === 0) path.moveTo(start.x, start.y);
    else path.lineTo(start.x, start.y);
    path.quadraticCurveTo(curr.x, curr.y, end.x, end.y);
  }
  path.closePath();
}

function buildBlobPath(mesh) {
  const path = new Path2D();

  if (mesh.kind === "circle") {
    path.arc(mesh.cx, mesh.cy, mesh.r, 0, Math.PI * 2);
    return path;
  }

  const verts = mesh.vertices;
  const n = verts.length;

  if (n === 2) {
    const width = Math.max(6, mesh.lineWidth ?? lineWidth());
    addCapsulePath(path, verts[0], verts[1], width / 2);
    return path;
  }

  if (n === 3 && (mesh.lineWidth ?? 0) > 2) {
    const [a, b, c] = verts;
    const mid = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
    const blend = Math.min(1, mesh.lineWidth / lineWidth());
    const cBlend = lerpPoint(c, mid, blend * 0.2);
    addPolygonPath(path, [a, b, cBlend], cornerRadiusFor([a, b, cBlend]));
    return path;
  }

  addPolygonPath(path, verts, cornerRadiusFor(verts));
  return path;
}

function buildShapeMesh(tips, flip) {
  const n = tips.length;
  if (n === 1) {
    return syncCircleMesh({
      cx: tips[0].x,
      cy: tips[0].y,
      r: canvas.height * 0.07,
    });
  }
  return syncMorphMesh(getTargetPoints(tips, flip));
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

  const handGroups = collectTipsByHand();
  const totalFingers = handGroups.reduce((s, g) => s + g.tips.length, 0);

  if (totalFingers === 0) {
    shapeNameEl.textContent = "—";
    fingerCountEl.textContent = "show your hand";
    blobMeshes.delete("shape");
    return;
  }

  const { tips, flip } = collectAllTips(handGroups);
  const mesh = buildShapeMesh(tips, flip);
  const path = buildBlobPath(mesh);

  shapeNameEl.textContent = shapeLabel(totalFingers);
  fingerCountEl.textContent = `${totalFingers} finger${totalFingers > 1 ? "s" : ""}`;

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
  ctx.strokeStyle = "rgba(255, 255, 255, 0.9)";
  ctx.lineWidth = 3;
  ctx.lineJoin = "miter";
  ctx.lineCap = "butt";
  ctx.shadowColor = "rgba(255, 255, 255, 0.8)";
  ctx.shadowBlur = 14;
  ctx.stroke(path);
  ctx.restore();

  for (const tip of tips) {
    ctx.beginPath();
    ctx.arc(tip.x, tip.y, 6, 0, Math.PI * 2);
    ctx.fillStyle = "#ffffff";
    ctx.fill();
  }
}

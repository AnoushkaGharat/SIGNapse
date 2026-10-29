import argparse
import json
import os
import time
from collections import Counter, deque

import numpy as np

FEAT = 64
IN_EXP = 7


def to_points(landmarks, w, h, handedness):
    p = np.array([(l.x * w, l.y * h, l.z * w) for l in landmarks], dtype=np.float32)
    p -= p[0]
    if handedness == "Left":
        p[:, 0] = -p[:, 0]
    return p


def features(P):
    P = np.asarray(P, dtype=np.float32).reshape(-1, 21, 3)
    s = np.maximum(np.abs(P[:, :, :2]).max(axis=(1, 2), keepdims=True), 1e-6)
    X = np.clip(P / s, -1, 1).reshape(len(P), 63)
    return np.pad(X, ((0, 0), (0, FEAT - 63)))


def quant_input(X):
    return np.clip(np.round(X * 2.0 ** IN_EXP), -128, 127).astype(np.int8)


HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
              (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]


class Tracker:
    def __init__(self):
        import subprocess
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
        if not os.path.exists(path) or os.path.getsize(path) < 100000:
            print("downloading hand tracking model (one time)...")
            subprocess.run(["curl", "-L", "-s", "-o", path, HAND_MODEL_URL], check=True)
        self.mp = mp
        self.det = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=path),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5))
        self.ts = 0

    def __call__(self, frame):
        import cv2
        self.ts = max(self.ts + 1, int(time.monotonic() * 1000))
        img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        r = self.det.detect_for_video(img, self.ts)
        if not r.hand_landmarks:
            return None
        lm = r.hand_landmarks[0]
        h, w = frame.shape[:2]
        pts = [(int(l.x * w), int(l.y * h)) for l in lm]
        for s, e in HAND_EDGES:
            cv2.line(frame, pts[s], pts[e], (0, 255, 0), 2)
        for p in pts:
            cv2.circle(frame, p, 4, (0, 0, 255), -1)
        return to_points(lm, w, h, r.handedness[0][0].category_name)


def load_data(path):
    if not os.path.exists(path):
        return np.zeros((0, 21, 3), np.float32), np.zeros((0,), "<U32")
    d = np.load(path)
    return d["P"], d["y"]


def cmd_collect(a):
    import cv2
    cap = cv2.VideoCapture(a.camera)
    if not cap.isOpened():
        raise SystemExit("Can't open the camera. On Mac: System Settings > Privacy & Security > Camera > turn on Terminal, then try again.")
    tr = Tracker()
    got, last, paused = [], 0.0, False
    t0 = time.time()
    print(f"Recording '{a.label}' starts in 3 seconds. Hold up the sign and move your hand around a little.")
    while len(got) < a.samples:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        p = tr(frame)
        left = 3 - (time.time() - t0)
        recording = left <= 0 and not paused
        if recording and p is not None and time.time() - last >= a.interval:
            got.append(p)
            last = time.time()
        if left > 0:
            msg, color = f"{a.label}: get ready... {int(left) + 1}", (0, 200, 255)
        elif paused:
            msg, color = f"{a.label}: {len(got)}/{a.samples}  PAUSED (space to resume)", (0, 200, 0)
        elif p is None:
            msg, color = f"{a.label}: {len(got)}/{a.samples}  no hand seen", (0, 200, 255)
        else:
            msg, color = f"{a.label}: {len(got)}/{a.samples}  RECORDING", (0, 0, 255)
        cv2.putText(frame, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.imshow("collect", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord(" "):
            paused = not paused
        elif k in (27, ord("q")):
            break
    cap.release()
    cv2.destroyAllWindows()
    if not got:
        print("Nothing saved: no hand was seen. Make sure your hand is in view and well lit.")
    if got:
        P, y = load_data(a.data)
        P = np.concatenate([P, np.stack(got)])
        y = np.concatenate([y, np.array([a.label] * len(got), "<U32")])
        np.savez(a.data, P=P, y=y)
        print(f"saved {len(got)} samples of '{a.label}', dataset now {len(y)} samples, classes {sorted(set(y))}")


def fq(x, e):
    s = 2.0 ** e
    return np.clip(np.round(x * s), -128, 127) / s


def pow2_exp(m):
    return int(np.floor(np.log2(127.0 / max(float(m), 1e-8))))


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def float_forward(params, X, q=None):
    h = X if q is None else fq(X, q["e"][0])
    hs, zs, Wqs = [], [], []
    for i, (W, b) in enumerate(params):
        Wq = W if q is None else fq(W, q["ew"][i])
        hs.append(h)
        Wqs.append(Wq)
        z = h @ Wq + b
        zs.append(z)
        if i < len(params) - 1:
            h = np.maximum(z, 0)
            if q is not None:
                h = fq(h, q["e"][i + 1])
    return z, hs, zs, Wqs


class Adam:
    def __init__(self, params, lr):
        self.lr, self.t = lr, 0
        self.m = [[np.zeros_like(p) for p in layer] for layer in params]
        self.v = [[np.zeros_like(p) for p in layer] for layer in params]

    def step(self, params, grads):
        self.t += 1
        b1, b2 = 0.9, 0.999
        for i, (layer, g) in enumerate(zip(params, grads)):
            for j in range(2):
                self.m[i][j] = b1 * self.m[i][j] + (1 - b1) * g[j]
                self.v[i][j] = b2 * self.v[i][j] + (1 - b2) * g[j] ** 2
                mh = self.m[i][j] / (1 - b1 ** self.t)
                vh = self.v[i][j] / (1 - b2 ** self.t)
                layer[j] -= self.lr * mh / (np.sqrt(vh) + 1e-8)


def train_step(params, X, Y, opt, wd, q=None):
    logits, hs, zs, Wqs = float_forward(params, X, q)
    p = softmax(logits)
    n = len(Y)
    loss = -np.log(p[np.arange(n), Y] + 1e-12).mean()
    g = p
    g[np.arange(n), Y] -= 1
    g /= n
    grads = [None] * len(params)
    for i in reversed(range(len(params))):
        grads[i] = (hs[i].T @ g + wd * params[i][0], g.sum(0))
        if i > 0:
            g = (g @ Wqs[i].T) * (zs[i - 1] > 0)
    opt.step(params, grads)
    return loss


def augment(P, rng):
    n = len(P)
    th = rng.uniform(-0.26, 0.26, n)[:, None]
    c, s = np.cos(th), np.sin(th)
    Q = P.copy()
    Q[:, :, 0] = c * P[:, :, 0] - s * P[:, :, 1]
    Q[:, :, 1] = s * P[:, :, 0] + c * P[:, :, 1]
    X = features(Q)
    X[:, :63] = np.clip(X[:, :63] + rng.normal(0, 0.02, (n, 63)), -1, 1)
    return X


def calibrate(params, X):
    e, ew = [IN_EXP], []
    h = fq(X, IN_EXP)
    for i, (W, b) in enumerate(params):
        ew.append(pow2_exp(np.abs(W).max()))
        z = h @ fq(W, ew[-1]) + b
        last = i == len(params) - 1
        h = z if last else np.maximum(z, 0)
        eo = pow2_exp(np.abs(h).max() if last else np.percentile(np.abs(h), 99.9))
        eo = min(eo, e[i] + ew[i])
        eo = max(eo, e[i] + ew[i] - 31)
        e.append(eo)
        h = fq(h, eo)
    return {"e": e, "ew": ew}


def quantize(params, q):
    layers = []
    for i, (W, b) in enumerate(params):
        acc_e = q["e"][i] + q["ew"][i]
        Wq = np.clip(np.round(W * 2.0 ** q["ew"][i]), -128, 127).astype(np.int8)
        bq = np.clip(np.round(b * 2.0 ** acc_e), -2 ** 31, 2 ** 31 - 1).astype(np.int32)
        layers.append(dict(W=Wq, b=bq, shift=int(acc_e - q["e"][i + 1]), relu=i < len(params) - 1))
    return layers


def wrap32(x):
    return ((x + 2 ** 31) % 2 ** 32) - 2 ** 31


def requant(acc, shift, relu):
    if shift > 0:
        acc = (acc + (1 << (shift - 1))) >> shift
    if relu:
        acc = np.maximum(acc, 0)
    return np.clip(acc, -128, 127)


def int_forward(layers, xq):
    h = np.atleast_2d(xq).astype(np.int64)
    for L in layers:
        acc = wrap32(h @ L["W"].astype(np.int64))
        h = requant(acc + L["b"].astype(np.int64), L["shift"], L["relu"])
    return h.astype(np.int8)


def export(out, layers, classes, q):
    os.makedirs(out, exist_ok=True)
    meta = {"classes": classes, "input_exp": IN_EXP, "output_exp": q["e"][-1], "feat": FEAT, "layers": []}
    arrays = {}
    for i, L in enumerate(layers):
        k, n = L["W"].shape
        wf, bf = f"layer{i}_w.hex", f"layer{i}_b.hex"
        with open(os.path.join(out, wf), "w") as f:
            f.write("\n".join(f"{int(v) & 0xFF:02x}" for v in L["W"].reshape(-1)) + "\n")
        with open(os.path.join(out, bf), "w") as f:
            f.write("\n".join(f"{int(v) & 0xFFFFFFFF:08x}" for v in L["b"]) + "\n")
        meta["layers"].append({"m": 1, "k": k, "n": n, "shift": L["shift"], "relu": L["relu"], "w_layout": "k_major", "w_file": wf, "b_file": bf})
        arrays[f"W{i}"], arrays[f"b{i}"] = L["W"], L["b"]
    meta["macs_per_inference"] = sum(l["k"] * l["n"] for l in meta["layers"])
    with open(os.path.join(out, "model.json"), "w") as f:
        json.dump(meta, f, indent=2)
    np.savez(os.path.join(out, "model.npz"), **arrays)


def load_model(d):
    if not os.path.exists(os.path.join(d, "model.json")):
        raise SystemExit("No trained model yet. Run collect for at least 2 signs, then run train.")
    with open(os.path.join(d, "model.json")) as f:
        meta = json.load(f)
    z = np.load(os.path.join(d, "model.npz"))
    layers = [dict(W=z[f"W{i}"], b=z[f"b{i}"], shift=l["shift"], relu=l["relu"]) for i, l in enumerate(meta["layers"])]
    return meta, layers


def cmd_train(a):
    rng = np.random.default_rng(a.seed)
    P, ylab = load_data(a.data)
    classes = sorted(set(ylab.tolist()))
    if len(classes) < 2:
        raise SystemExit(f"Need at least 2 different signs to train, you have {len(classes)}: {classes}. Run collect for another sign.")
    Y = np.array([classes.index(c) for c in ylab])
    idx = rng.permutation(len(Y))
    nv = max(1, int(len(Y) * a.val))
    vi, ti = idx[:nv], idx[nv:]
    Xv = features(P[vi])
    sizes = [FEAT] + a.hidden + [len(classes)]
    params = [[rng.normal(0, np.sqrt(2 / k), (k, n)), np.zeros(n)] for k, n in zip(sizes[:-1], sizes[1:])]

    def run(epochs, lr, q):
        opt = Adam(params, lr)
        for ep in range(epochs):
            order = rng.permutation(ti)
            Xa = augment(P[order], rng)
            losses = [train_step(params, Xa[s:s + a.batch], Y[order][s:s + a.batch], opt, a.wd, q) for s in range(0, len(order), a.batch)]
            if (ep + 1) % 25 == 0 or ep == epochs - 1:
                acc = (float_forward(params, Xv, q)[0].argmax(1) == Y[vi]).mean()
                print(f"{'qat' if q else 'fp32'} epoch {ep + 1}/{epochs} loss {np.mean(losses):.4f} val {acc:.3f}")

    run(a.epochs, a.lr, None)
    q = calibrate(params, features(P[ti]))
    run(a.qat_epochs, a.lr * 0.2, q)
    q = calibrate(params, features(P[ti]))
    layers = quantize(params, q)
    fp = (float_forward(params, Xv)[0].argmax(1) == Y[vi]).mean()
    ip = (int_forward(layers, quant_input(Xv)).argmax(1) == Y[vi]).mean()
    export(a.out, layers, classes, q)
    print(f"classes {classes}")
    print(f"val acc fp32 {fp:.3f}  int8 chip-exact {ip:.3f}")
    print(f"layers {[(l['W'].shape, l['shift']) for l in layers]}  exported to {a.out}/")


class GoldenBackend:
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, xq):
        return int_forward(self.layers, xq)[0]


class SerialBackend:
    def __init__(self, port, baud, n_out):
        import serial
        self.s = serial.Serial(port, baud, timeout=1)
        self.n = n_out
        time.sleep(2)
        self.s.reset_input_buffer()

    def __call__(self, xq):
        self.s.write(b"\xa5" + np.asarray(xq, np.int8).tobytes())
        r = self.s.read(self.n)
        if len(r) != self.n:
            raise RuntimeError(f"chip returned {len(r)}/{self.n} bytes")
        return np.frombuffer(r, dtype=np.int8)


class VerilogBackend:
    def __init__(self, layers, rtl_dir, build_dir):
        import glob
        import subprocess
        os.makedirs(build_dir, exist_ok=True)
        wf, bf, df = (os.path.join(build_dir, f) for f in ("weights.hex", "bias.hex", "desc.hex"))
        wbytes, bwords, desc, wb, bb = [], [], [], 0, 0
        for L in layers:
            k, n = L["W"].shape
            wbytes += [int(v) & 0xFF for v in L["W"].reshape(-1)]
            bwords += [int(v) & 0xFFFFFFFF for v in L["b"]]
            desc += [k, n, (int(L["relu"]) << 5) | L["shift"], wb, bb]
            wb += k * n
            bb += n
        if wb > 32768 or bb > 1024 or max(max(L["W"].shape) for L in layers) > 256:
            raise SystemExit("model too big for mlp_engine memories, raise WDEPTH/BDEPTH/ADEPTH")
        with open(wf, "w") as f:
            f.write("\n".join(f"{v:02x}" for v in wbytes) + "\n")
        with open(bf, "w") as f:
            f.write("\n".join(f"{v:08x}" for v in bwords) + "\n")
        with open(df, "w") as f:
            f.write("\n".join(f"{v:08x}" for v in desc) + "\n")
        vvp = os.path.join(build_dir, "mlp.vvp")
        srcs = sorted(glob.glob(os.path.join(rtl_dir, "*.v")) + glob.glob(os.path.join(rtl_dir, "*.sv")))
        subprocess.run(["iverilog", "-g2012", "-s", "tb_mlp", "-o", vvp] + srcs, check=True)
        self.n = layers[-1]["W"].shape[1]
        self.last_cycles = 0
        self.p = subprocess.Popen(
            ["vvp", "-n", vvp, f"+W={wf}", f"+B={bf}", f"+D={df}", f"+NL={len(layers)}", f"+NIN={layers[0]['W'].shape[0]}", f"+NOUT={self.n}"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self._wait("READY")

    def _wait(self, tag):
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("verilog simulator exited")
            if line.startswith(tag):
                return line

    def __call__(self, xq):
        self.p.stdin.write(" ".join(str(int(v)) for v in xq) + "\n")
        self.p.stdin.flush()
        vals = [int(t) for t in self._wait("OUT").split()[1:]]
        self.last_cycles = vals[-1]
        return np.array(vals[:-1], dtype=np.int8)

    def close(self):
        self.p.stdin.close()
        self.p.wait()


def make_backend(a, meta, layers):
    if a.backend == "serial":
        return SerialBackend(a.port, a.baud, len(meta["classes"]))
    if a.backend == "verilog":
        return VerilogBackend(layers, a.rtl, os.path.join(a.model, "sim_build"))
    return GoldenBackend(layers)


def cmd_eval(a):
    meta, layers = load_model(a.model)
    P, ylab = load_data(a.data)
    classes = meta["classes"]
    keep = np.isin(ylab, classes)
    Xq = quant_input(features(P[keep]))
    Y = np.array([classes.index(c) for c in ylab[keep]])
    be = make_backend(a, meta, layers)
    gold = int_forward(layers, Xq)
    t0 = time.perf_counter()
    out = np.stack([be(x) for x in Xq])
    dt = (time.perf_counter() - t0) / len(Xq) * 1e3
    print(f"backend {a.backend}: acc {(out.argmax(1) == Y).mean():.3f} over {len(Y)} samples, {dt:.3f} ms/inference on this computer")
    if hasattr(be, "last_cycles"):
        print(f"chip needs {be.last_cycles} clock cycles per inference = {be.last_cycles / a.mhz:.1f} us at {a.mhz:g} MHz")
    if a.backend != "golden":
        bad = np.any(out != gold, axis=1)
        print(f"bit-exact vs golden: {len(Y) - bad.sum()}/{len(Y)} match")
        for i in np.where(bad)[0][:5]:
            print(f"  sample {i}: chip {out[i].tolist()} golden {gold[i].tolist()}")


def cmd_live(a):
    import cv2
    meta, layers = load_model(a.model)
    classes = meta["classes"]
    oscale = 2.0 ** -meta["output_exp"]
    be = make_backend(a, meta, layers)
    tr = Tracker()
    cap = cv2.VideoCapture(a.camera)
    hist = deque(maxlen=a.smooth)
    import shutil
    import subprocess
    voice = shutil.which("say") or shutil.which("espeak")
    spoken = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        p = tr(frame)
        text = "no hand"
        if p is not None:
            xq = quant_input(features(p))[0]
            t0 = time.perf_counter()
            logits = be(xq)
            ms = (time.perf_counter() - t0) * 1e3
            pr = softmax(logits[None].astype(np.float64) * oscale)[0]
            hist.append(int(pr.argmax()) if pr.max() >= a.threshold else -1)
            top, cnt = Counter(hist).most_common(1)[0]
            label = classes[top] if top >= 0 and cnt > len(hist) // 2 else "?"
            if voice and not a.mute and label != "?" and label != spoken and len(hist) == hist.maxlen:
                subprocess.Popen([voice, label])
                spoken = label
            text = f"{label}  {pr.max():.2f}  {a.backend} {ms:.2f} ms"
            if hasattr(be, "last_cycles"):
                text += f"  chip {be.last_cycles / a.mhz:.0f} us"
        else:
            hist.clear()
            spoken = None
        cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.imshow("signapse", frame)
        if cv2.waitKey(1) & 0xFF in (27, ord("q")):
            break
    cap.release()
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)

    c = sp.add_parser("collect")
    c.add_argument("label")
    c.add_argument("--samples", type=int, default=200)
    c.add_argument("--interval", type=float, default=0.05)
    c.add_argument("--data", default="hands.npz")
    c.add_argument("--camera", type=int, default=0)

    t = sp.add_parser("train")
    t.add_argument("--data", default="hands.npz")
    t.add_argument("--out", default="model")
    t.add_argument("--hidden", type=int, nargs="+", default=[64, 32])
    t.add_argument("--epochs", type=int, default=150)
    t.add_argument("--qat-epochs", type=int, default=50)
    t.add_argument("--lr", type=float, default=3e-3)
    t.add_argument("--wd", type=float, default=1e-4)
    t.add_argument("--batch", type=int, default=64)
    t.add_argument("--val", type=float, default=0.2)
    t.add_argument("--seed", type=int, default=0)

    for name in ("eval", "live"):
        p = sp.add_parser(name)
        p.add_argument("--model", default="model")
        p.add_argument("--backend", choices=["golden", "verilog", "serial"], default="golden")
        p.add_argument("--rtl", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "sim"))
        p.add_argument("--mhz", type=float, default=50.0)
        p.add_argument("--port", default="/dev/ttyUSB0")
        p.add_argument("--baud", type=int, default=115200)
        if name == "eval":
            p.add_argument("--data", default="hands.npz")
        else:
            p.add_argument("--camera", type=int, default=0)
            p.add_argument("--smooth", type=int, default=7)
            p.add_argument("--threshold", type=float, default=0.6)
            p.add_argument("--mute", action="store_true")

    a = ap.parse_args()
    {"collect": cmd_collect, "train": cmd_train, "eval": cmd_eval, "live": cmd_live}[a.cmd](a)


if __name__ == "__main__":
    main()
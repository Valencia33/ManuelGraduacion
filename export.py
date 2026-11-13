"""PSD -> web layers.   python export.py [art.psd] [--hidden] [--watch] [--push]
(needs: pip install psd-tools; with no path it uses the .psd next to this file)

Writes layers/<n>.webp + layers/layers.js, bottom layer first.
Each visible top-level folder becomes one moving piece, named after the folder.
A loose layer becomes its own piece, named after the layer, unless it fills the
whole canvas: those are flattened together (the backdrop). Folders named in SPLIT are opened up
instead, and layers inside them are merged by the text before "_"
(M_white + M_color + M_lineart -> "M"). Folders named in SKIP are left out.
A piece under a clipped Hue/Saturation layer is colour-matched to the PSD's own
flattened picture, because psd-tools cannot apply that adjustment in CMYK files.
The eye icons are respected. --hidden also exports folders whose eye is off
(handy when you saved with folders hidden while painting); hidden loose layers stay out.
--watch keeps running and re-exports every time the PSD is saved.
--push then commits layers/ and pushes, so the published site follows the PSD.
"""
import json, os, subprocess, sys, time
from itertools import groupby
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms
from psd_tools import PSDImage
from psd_tools.composite import composite

OUT_W = 1920        # width of the full canvas on the web
SPLIT = {"todo", "letras"}  # folders to open up instead of flattening
SKIP = {"a"}                # folders to leave out (here: flattened copies of the whole picture)
# The PSD is CMYK with no embedded profile; Photoshop shows it through its working
# CMYK space (SWOP by default). Windows ships a SWOP profile, so use it to match.
# Colours look off? Point this at the .icc you use, or at a missing path for raw maths.
CMYK_PROFILE = Path(r"C:\Windows\System32\spool\drivers\color\RSWOP.icm")


SHOW_HIDDEN_FOLDERS = False
W = H = 0  # canvas size, set in main


def units(layers, split=False):
    if SHOW_HIDDEN_FOLDERS:
        for l in layers:
            if l.is_group():
                l.visible = True  # in memory only, the PSD is never written
    live = (l for l in layers if l.visible and l.width and l.height and l.name.strip() not in SKIP)
    for is_group, run in groupby(live, key=lambda l: l.is_group()):
        run = list(run)
        if is_group:
            for g in run:
                if g.name.strip() in SPLIT:
                    yield from units(g, True)
                else:
                    yield g.name.strip(), [g]
        elif split:
            for prefix, same in groupby(run, key=lambda l: l.name.split("_")[0]):
                yield prefix, list(same)
        else:
            for full, same in groupby(run, key=lambda l: l.left <= 0 and l.top <= 0 and l.right >= W and l.bottom >= H):
                same = list(same)
                if full:
                    yield same[0].name.strip(), same
                else:
                    yield from ((l.name.strip(), [l]) for l in same)


def to_rgba(psd, color, alpha):
    a = (alpha[..., 0] * 255).round().astype(np.uint8)
    if psd.color_mode.name != "CMYK":
        rgb = Image.fromarray((color[..., :3] * 255).round().astype(np.uint8), "RGB")
    elif CMYK_PROFILE.exists():
        ink = Image.fromarray(((1 - color) * 255).round().astype(np.uint8), "CMYK")
        rgb = ImageCms.profileToProfile(ink, str(CMYK_PROFILE), ImageCms.createProfile("sRGB"), outputMode="RGB",
                                       renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
                                       flags=ImageCms.Flags.BLACKPOINTCOMPENSATION)  # Photoshop's defaults
    else:
        rgb = Image.fromarray((color[..., :3] * color[..., 3:] * 255).round().astype(np.uint8), "RGB")
    rgb.putalpha(Image.fromarray(a, "L"))
    return rgb


def flat_rgb(psd):
    """The whole picture as Photoshop flattened it on save, in sRGB."""
    im = psd.topil()
    if psd.color_mode.name != "CMYK":
        return im.convert("RGB")
    color = 1 - np.asarray(im, np.float32) / 255
    return to_rgba(psd, color, np.ones(color.shape[:2] + (1,), np.float32)).convert("RGB")


def match_colors(img, ref):
    """Recolour a piece so it looks as it does in the flattened picture `ref` (same size).
    One 3x4 colour matrix, fitted on the piece's opaque pixels. Wherever something else
    covers the piece in `ref` the pixels disagree wildly, so the worst 40% are dropped and
    the fit repeated. ponytail: one matrix per piece; a piece that is mostly covered, or
    tinted differently in different places, would need a finer model."""
    a = np.asarray(img, np.float32) / 255
    solid = a[..., 3] > 0.99
    x = np.c_[a[solid][:, :3], np.ones(solid.sum(), np.float32)]
    y = (np.asarray(ref, np.float32) / 255)[solid]
    keep = np.ones(len(x), bool)
    for _ in range(4):
        m = np.linalg.lstsq(x[keep], y[keep], rcond=None)[0]
        err = np.abs(x @ m - y).sum(1)
        keep = err <= np.quantile(err, 0.6)
    a[..., :3] = np.clip(a[..., :3] @ m[:3] + m[3], 0, 1)
    return Image.fromarray((a * 255).round().astype(np.uint8), "RGBA")


HERE = Path(__file__).parent


def main(psd_path, *flags, out=HERE / "layers"):
    global SHOW_HIDDEN_FOLDERS
    SHOW_HIDDEN_FOLDERS = "--hidden" in flags
    global W, H
    psd = PSDImage.open(psd_path)
    W, H = psd.size
    out = Path(out)
    out.mkdir(exist_ok=True)
    k = OUT_W / psd.width
    meta, flat = [], None
    for i, (name, layers) in enumerate(units(psd)):
        nodes = list(layers)  # the piece's layers, everything inside them, and the folders they sit in
        for l in layers:
            nodes += l.descendants() if l.is_group() else []
            p = l.parent
            while p is not None and p is not psd:
                nodes.append(p)
                p = p.parent
        keep = {id(l) for l in nodes}
        tinted = psd.color_mode.name == "CMYK" and any(
            c.kind == "huesaturation" and c.visible for l in nodes for c in getattr(l, "clip_layers", []))
        # clip to the canvas
        x0 = max(0, min(l.left for l in layers)); y0 = max(0, min(l.top for l in layers))
        x1 = min(psd.width, max(l.right for l in layers)); y1 = min(psd.height, max(l.bottom for l in layers))
        color, _, alpha = composite(psd, color=1.0, alpha=0.0, viewport=(x0, y0, x1, y1),
                                    layer_filter=lambda l: l.visible and id(l) in keep)
        img = to_rgba(psd, color, alpha)
        # premultiplied resize, otherwise transparent pixels bleed a white fringe
        size = (max(1, round(img.width * k)), max(1, round(img.height * k)))
        img = img.convert("RGBa").resize(size, Image.LANCZOS).convert("RGBA")
        if tinted:
            flat = flat or flat_rgb(psd)
            img = match_colors(img, flat.crop((x0, y0, x1, y1)).resize(size, Image.LANCZOS))
        box = img.getbbox()
        if not box:
            continue
        img = img.crop(box)
        src = f"{len(meta):02d}.webp"
        img.save(out / src, quality=90, method=6)
        meta.append({"name": name, "src": src, "x": round(x0 * k) + box[0], "y": round(y0 * k) + box[1],
                     "w": img.width, "h": img.height})
        print(src, name, img.size, "colour-matched" if tinted else "", flush=True)
    for old in set(out.glob("*.webp")) - {out / m["src"] for m in meta}:
        old.unlink()  # pieces that no longer exist
    doc = {"w": OUT_W, "h": round(psd.height * k), "layers": meta}
    (out / "layers.js").write_text("const ART = " + json.dumps(doc, ensure_ascii=False, indent=1) + ";\n", encoding="utf-8")


def publish():
    git = lambda *a, **k: subprocess.run(["git", *a], cwd=HERE, **k)
    git("add", "layers", check=True)
    if git("diff", "--cached", "--quiet", "--", "layers").returncode:  # the art really changed
        git("commit", "-m", "Update art from PSD", "--", "layers", check=True)
        git("push", check=True)


def watch(psd_path, *flags):
    seen = None
    while True:
        m = os.path.getmtime(psd_path)
        if m != seen and time.time() - m > 3:  # 3 quiet seconds: Photoshop has finished writing
            seen = m
            try:
                main(psd_path, *flags)
                if "--push" in flags:
                    publish()
            except Exception as e:  # half-written file, git refused...: report and wait for the next save
                print("failed:", e)
            print("watching", psd_path, "- Ctrl+C to stop", flush=True)
        time.sleep(2)


if __name__ == "__main__":
    flags = [a for a in sys.argv[1:] if a.startswith("--")]
    psds = [a for a in sys.argv[1:] if not a.startswith("--")] or sorted(HERE.glob("*.psd"))
    if not psds:
        sys.exit(__doc__)
    (watch if "--watch" in flags else main)(str(psds[0]), *flags)

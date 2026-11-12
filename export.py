"""PSD -> web layers.   python export.py [art.psd] [--hidden] [--watch] [--push]
(needs: pip install psd-tools; with no path it uses the .psd next to this file)

Writes layers/<n>.webp + layers/layers.js, bottom layer first.
Each visible top-level folder becomes one moving piece, named after the folder.
A loose layer becomes its own piece, named after the layer, unless it fills the
whole canvas: those are flattened together (the backdrop). Folders named in SPLIT are opened up
instead, and layers inside them are merged by the text before "_"
(M_white + M_color + M_lineart -> "M").
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
SPLIT = {"letras"}  # folders to open up instead of flattening
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
    live = (l for l in layers if l.visible and l.width and l.height)
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
    meta = []
    for i, (name, layers) in enumerate(units(psd)):
        keep = {id(l) for l in layers}
        for l in layers:
            keep |= {id(d) for d in l.descendants()} if l.is_group() else set()
            p = l.parent
            while p is not None and p is not psd:
                keep.add(id(p))
                p = p.parent
        # clip to the canvas
        x0 = max(0, min(l.left for l in layers)); y0 = max(0, min(l.top for l in layers))
        x1 = min(psd.width, max(l.right for l in layers)); y1 = min(psd.height, max(l.bottom for l in layers))
        color, _, alpha = composite(psd, color=1.0, alpha=0.0, viewport=(x0, y0, x1, y1),
                                    layer_filter=lambda l: l.visible and id(l) in keep)
        img = to_rgba(psd, color, alpha)
        # premultiplied resize, otherwise transparent pixels bleed a white fringe
        size = (max(1, round(img.width * k)), max(1, round(img.height * k)))
        img = img.convert("RGBa").resize(size, Image.LANCZOS).convert("RGBA")
        box = img.getbbox()
        if not box:
            continue
        img = img.crop(box)
        if not meta:
            bg = "#%02x%02x%02x" % img.getpixel((0, 0))[:3]  # page colour behind the art
        src = f"{len(meta):02d}.webp"
        img.save(out / src, quality=90, method=6)
        meta.append({"name": name, "src": src, "x": round(x0 * k) + box[0], "y": round(y0 * k) + box[1],
                     "w": img.width, "h": img.height})
        print(src, name, img.size, flush=True)
    for old in set(out.glob("*.webp")) - {out / m["src"] for m in meta}:
        old.unlink()  # pieces that no longer exist
    doc = {"bg": bg, "w": OUT_W, "h": round(psd.height * k), "layers": meta}
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

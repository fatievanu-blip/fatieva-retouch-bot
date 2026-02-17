from PIL import Image, ImageDraw, ImageFont
import io


def make_before_after_collage(before_jpeg: bytes, after_jpeg: bytes) -> bytes:
    before = Image.open(io.BytesIO(before_jpeg)).convert("RGB")
    after = Image.open(io.BytesIO(after_jpeg)).convert("RGB")

    h = min(before.height, after.height)
    before = before.resize((int(before.width * h / before.height), h))
    after = after.resize((int(after.width * h / after.height), h))

    gap = max(20, h // 60)
    w = before.width + gap + after.width
    canvas = Image.new("RGB", (w, h))

    canvas.paste(before, (0, 0))
    canvas.paste(after, (before.width + gap, 0))

    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=max(26, h // 28))
    except Exception:
        font = ImageFont.load_default()

    pad = max(18, h // 50)
    draw.rectangle([pad, pad, pad + max(120, h // 4), pad + max(40, h // 14)], fill=(0, 0, 0))
    draw.text((pad + 12, pad + 6), "До", fill=(255, 255, 255), font=font)

    x2 = before.width + gap + pad
    draw.rectangle([x2, pad, x2 + max(160, h // 3), pad + max(40, h // 14)], fill=(0, 0, 0))
    draw.text((x2 + 12, pad + 6), "После", fill=(255, 255, 255), font=font)

    out = io.BytesIO()
    canvas.save(out, format="JPEG", quality=94, subsampling=0, optimize=True)
    return out.getvalue()

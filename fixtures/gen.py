"""Generate synthetic envelope images with FAKE data only + ground truth.

Run: python gen.py (needs Pillow). Deterministic: random.seed(42).
"""
import json, random, os
from PIL import Image, ImageDraw, ImageFont, ImageFilter
random.seed(42)
HERE = os.path.dirname(os.path.abspath(__file__))
F = os.path.join(HERE, "fonts")
OUT = os.path.join(HERE, "envelopes")
def font(name, size, idx=0):
    return ImageFont.truetype(os.path.join(F, name), size, index=idx)
# Fonts come from Debian packages (DejaVu, Liberation, Comic Neue, Dancing Script, Kristi); see fonts/LICENSE.*
FONTS = {
 "helv": ("DejaVuSans.ttf", 0), "helvb": ("DejaVuSans-Bold.ttf", 0), "times": ("LiberationSerif-Regular.ttf", 0),
 "courier": ("LiberationMono-Regular.ttf", 0), "verdana": ("DejaVuSans.ttf", 0),
 "arialn": ("LiberationSans-Bold.ttf", 0), "bradley": ("ComicNeue-Bold.otf", 0),
 "note": ("DancingScript-Bold.otf", 0), "georgia": ("LiberationSerif-Regular.ttf", 0), "marker": ("Kristi.ttf", 0),
}
def f(k, s): n, i = FONTS[k]; return font(n, s, i)
W, H = 1900, 830  # #10 envelope at ~200 dpi
NAMES = ["AVERY Q TESTPERSON", "JORDAN SAMPLE", "RILEY PLACEHOLDER", "MORGAN EXAMPLE", "CASEY FICTIONAL", "TAYLOR DEMOUSER"]
STREETS = ["123 EXAMPLE AVE", "4567 SAMPLE RD APT 2B", "89 PLACEHOLDER LN", "1010 FICTION BLVD", "22 TESTING CT"]
CITIES = ["SAMPLETON OH 45999-1234", "EXAMPLEVILLE TX 79999", "FAKETOWN WA 98999-0001", "DEMOCITY MN 55999"]
SENDERS = [("ACME UTILITY CO", "PO BOX 98765", "SAMPLETON OH 45999-8765"),
           ("FIRST FICTIONAL BANK", "PO BOX 4321", "DEMOCITY MN 55999"),
           ("COUNTY OF EXAMPLE TREASURER", "100 MAIN ST RM 5", "EXAMPLEVILLE TX 79999"),
           ("NORTHWIND INSURANCE", "PO BOX 1111", "FAKETOWN WA 98999-1111"),
           ("CONTOSO HEALTH PLAN", "PO BOX 7070", "SAMPLETON OH 45999")]
def barcode(d, x, y, n=65):
    for i in range(n):
        h = random.choice([14, 22, 30, 38]); top = y + (38 - h) // 2 if h in (14, 22) else y
        d.rectangle([x + i * 9, top, x + i * 9 + 4, top + h], fill=0)
def stamp(d, x, y, kind):
    d.rectangle([x, y, x + 230, y + 150], outline=0, width=3)
    if kind == "meter":
        d.text((x + 15, y + 15), "US POSTAGE", font=f("helvb", 24), fill=0)
        d.text((x + 15, y + 50), "$000.69", font=f("helvb", 34), fill=0)
        d.text((x + 15, y + 100), "ZIP 45999", font=f("helv", 22), fill=0)
    elif kind == "stamp":
        d.rectangle([x + 20, y + 15, x + 210, y + 135], fill=150)
def block(d, x, y, lines, fnt, gap=1.25):
    sz = fnt.size
    for i, l in enumerate(lines):
        d.text((x, y + int(i * sz * gap)), l, font=fnt, fill=0)
def make(i, kind):
    im = Image.new("L", (W, H), 245); d = ImageDraw.Draw(im)
    gt = {"id": f"env{i:02d}", "kind": kind, "addressee": [], "return": [], "markers": []}
    name, st, city = random.choice(NAMES), random.choice(STREETS), random.choice(CITIES)
    if kind in ("business", "window", "typewriter"):
        s = random.choice(SENDERS); rf = f("courier", 34) if kind == "typewriter" else f(random.choice(["helvb", "arialn", "verdana"]), 30)
        block(d, 60, 50, list(s), rf); gt["return"] = list(s)
        stamp(d, W - 300, 40, "meter")
        af = f("courier", 44) if kind == "typewriter" else f(random.choice(["helv", "verdana", "arialn", "times"]), 42)
        ax, ay = 640, 380
        if kind == "window":
            d.rounded_rectangle([ax - 40, ay - 60, ax + 900, ay + 230], 18, fill=228, outline=200, width=2)
            d.text((60, 230), "IMPORTANT TAX DOCUMENT ENCLOSED", font=f("helvb", 32), fill=0); gt["markers"].append("IMPORTANT TAX DOCUMENT ENCLOSED")
        barcode(d, ax, ay - 50)
        block(d, ax, ay, [name, st, city], af); gt["addressee"] = [name, st, city]
    elif kind == "bulk":
        d.rectangle([W - 330, 40, W - 50, 220], outline=0, width=3)
        ind = ["PRSRT STD", "US POSTAGE", "PAID", "PERMIT NO 1234", "SAMPLETON OH"]
        block(d, W - 300, 55, ind, f("helvb", 26)); gt["markers"] += ["PRSRT STD", "PERMIT NO 1234"]
        promo = random.choice(["YOU'RE PRE-APPROVED!", "LIMITED TIME OFFER INSIDE", "FINAL NOTICE: ACT NOW"])
        d.text((60, 60), promo, font=f("helvb", 56), fill=0); gt["markers"].append(promo)
        addr = random.choice(["CURRENT RESIDENT", "OR CURRENT RESIDENT", "POSTAL CUSTOMER"])
        line0 = "*****ECRWSS**" if random.random() < .5 else "****AUTO**5-DIGIT 45999"
        gt["markers"].append(line0.strip("*").replace("**", " ").split()[0])
        block(d, 640, 380, [line0, addr, st, city], f("helv", 38)); gt["addressee"] = [addr, st, city]
    elif kind == "handwritten":
        rfn = random.choice(["bradley", "note", "marker"])
        rn = random.choice(NAMES).title(); rs = random.choice(STREETS).title(); rc = random.choice(CITIES).title()
        block(d, 60, 50, [rn, rs, rc], f(rfn, 36)); gt["return"] = [rn, rs, rc]
        stamp(d, W - 300, 40, "stamp")
        n2 = name.title(); block(d, 600, 360, [n2, st.title(), city.title()], f(rfn, 56)); gt["addressee"] = [n2, st.title(), city.title()]
    elif kind == "postcard":
        d.text((60, 60), "WE MISS YOU! 20% OFF", font=f("georgia", 64), fill=40); gt["markers"].append("20% OFF")
        d.text((60, 160), "Visit example.com/save by 12/31", font=f("verdana", 28), fill=60)
        block(d, W - 760, 470, ["ECRWSS", "RESIDENTIAL CUSTOMER", city], f("helv", 30)); gt["addressee"] = ["RESIDENTIAL CUSTOMER", city]; gt["markers"].append("ECRWSS")
        d.rectangle([W - 330, 40, W - 50, 200], outline=0, width=3); block(d, W - 300, 55, ["PRSRT MKTG", "US POSTAGE PAID"], f("helvb", 24)); gt["markers"].append("PRSRT MKTG")
    return im, gt
def informed_delivery(im):
    # USPS Informed Delivery look: grayscale, downscaled, soft, JPEG-compressed
    w = 780; im2 = im.resize((w, int(im.height * w / im.width)), Image.BILINEAR).filter(ImageFilter.GaussianBlur(0.6))
    return im2
kinds = ["business", "window", "bulk", "handwritten", "typewriter", "postcard"] * 2
gts = []
os.makedirs(OUT, exist_ok=True)
for i, k in enumerate(kinds):
    im, gt = make(i, k)
    im.save(f"{OUT}/{gt['id']}_clean.png")
    informed_delivery(im).save(f"{OUT}/{gt['id']}_id.jpg", quality=55)
    informed_delivery(im.rotate(random.choice([-3, 3]), expand=True, fillcolor=245)).save(f"{OUT}/{gt['id']}_idrot.jpg", quality=55)
    gts.append(gt)
json.dump(gts, open(f"{OUT}/ground_truth.json", "w"), indent=1)
print(len(gts), "envelopes,", len(os.listdir(OUT)) - 1, "images")

from PIL import Image, ImageDraw, ImageFont

# =========================
# INPUT / OUTPUT
# =========================
img_path = r"C:\Users\aataiwo\Downloads\Buist_Spruill\sumo_network.png"

out_png = "fig_sumo_network_annotated_final.png"
out_pdf = "fig_sumo_network_annotated_final.pdf"

# =========================
# LOAD IMAGE
# =========================
img = Image.open(img_path).convert("RGB")
draw = ImageDraw.Draw(img, "RGBA")

w, h = img.size
cx, cy = w // 2, h // 2

# If marker is slightly off, adjust here:
# cx, cy = w // 2, h // 2 - 5

# =========================
# FONTS
# =========================
try:
    font_dir = ImageFont.truetype("arial.ttf", 34)
    font_med = ImageFont.truetype("arial.ttf", 22)
    font_small = ImageFont.truetype("arial.ttf", 18)
    font_legend = ImageFont.truetype("arial.ttf", 16)
except:
    font_dir = ImageFont.load_default()
    font_med = ImageFont.load_default()
    font_small = ImageFont.load_default()
    font_legend = ImageFont.load_default()

# =========================
# COLORS
# =========================
yellow = (255, 240, 0, 255)
cyan = (0, 220, 255, 80)
cyan_solid = (0, 230, 255, 255)
red = (255, 40, 40, 255)
white = (255, 255, 255, 255)
black = (0, 0, 0, 220)
dark_box = (20, 20, 20, 175)

# =========================
# HELPERS
# =========================
def outlined_text(x, y, text, font, fill=white, outline=black):
    draw.text((x + 2, y + 2), text, font=font, fill=outline)
    draw.text((x, y), text, font=font, fill=fill)


def arrow_line(start, end, color=yellow, width=4):
    draw.line((start, end), fill=color, width=width)

    x1, y1 = start
    x2, y2 = end

    dx = x2 - x1
    dy = y2 - y1

    if abs(dx) > abs(dy):  # horizontal arrow
        if dx > 0:
            pts = [(x2, y2), (x2 - 16, y2 - 9), (x2 - 16, y2 + 9)]
        else:
            pts = [(x2, y2), (x2 + 16, y2 - 9), (x2 + 16, y2 + 9)]
    else:  # vertical arrow
        if dy > 0:
            pts = [(x2, y2), (x2 - 9, y2 - 16), (x2 + 9, y2 - 16)]
        else:
            pts = [(x2, y2), (x2 - 9, y2 + 16), (x2 + 9, y2 + 16)]

    draw.polygon(pts, fill=color)

# =========================
# FLOOD-PRONE LINK HIGHLIGHTS
# =========================
draw.rectangle((cx - 185, cy - 6, cx - 55, cy + 8), fill=cyan)
draw.rectangle((cx + 55, cy - 6, cx + 185, cy + 8), fill=cyan)

# =========================
# DIRECTION ARROWS
# =========================
arrow_line((cx, 68), (cx, cy - 72), width=4)
arrow_line((cx, h - 68), (cx, cy + 72), width=4)
arrow_line((68, cy), (cx - 82, cy), width=4)
arrow_line((w - 68, cy), (cx + 82, cy), width=4)

# =========================
# DIRECTION LABELS
# =========================
outlined_text(cx - 10, 20, "N", font_dir)
outlined_text(cx - 10, h - 48, "S", font_dir)
outlined_text(22, cy - 18, "W", font_dir)
outlined_text(w - 48, cy - 18, "E", font_dir)

# =========================
# STREET LABELS
# =========================
outlined_text(cx - 220, cy - 72, "Buist Avenue", font_med)
outlined_text(cx + 40, 75, "Spruill Avenue", font_med)
# =========================
# SIGNALIZED INTERSECTION MARKER
# =========================
draw.ellipse((cx - 17, cy - 17, cx + 17, cy + 17), outline=red, width=5)

# Move label away from roadway
label_x, label_y = cx + 30, cy - 145
draw.line((cx + 10, cy - 12, label_x + 18, label_y + 50), fill=red, width=2)
outlined_text(label_x, label_y, "Signalized\nIntersection", font_small, red)

# =========================
# LEGEND
# =========================
legend_x, legend_y = 18, h - 88
legend_w, legend_h = 245, 72

draw.rectangle(
    (legend_x, legend_y, legend_x + legend_w, legend_y + legend_h),
    fill=dark_box,
    outline=white,
    width=2
)

outlined_text(legend_x + 10, legend_y + 6, "Legend", font_legend)

# Flood legend
draw.rectangle(
    (legend_x + 12, legend_y + 33, legend_x + 62, legend_y + 44),
    fill=cyan
)
outlined_text(
    legend_x + 75,
    legend_y + 25,
    "Flood-prone links",
    font_legend,
    cyan_solid
)

# Signal legend
draw.ellipse(
    (legend_x + 23, legend_y + 51, legend_x + 41, legend_y + 69),
    outline=red,
    width=3
)
outlined_text(
    legend_x + 75,
    legend_y + 48,
    "Traffic signal",
    font_legend,
    red
)

# =========================
# SAVE
# =========================
img.save(out_png, dpi=(600, 600))
img.save(out_pdf, "PDF", resolution=600)

print(f"Saved: {out_png}")
print(f"Saved: {out_pdf}")
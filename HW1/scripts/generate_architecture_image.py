#!/usr/bin/env python3

from PIL import Image, ImageDraw, ImageFont


def load_font(size):
    candidates = [
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def draw_rounded_box(draw, xy, radius, fill, outline, width=2):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def draw_multiline_centered(draw, box, text, font, fill):
    x0, y0, x1, y1 = box
    lines = text.split("\n")
    line_heights = []
    line_widths = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_widths.append(bbox[2] - bbox[0])
        line_heights.append(bbox[3] - bbox[1])

    total_h = sum(line_heights) + max(0, len(lines) - 1) * 4
    y = y0 + ((y1 - y0) - total_h) / 2
    for i, line in enumerate(lines):
        w = line_widths[i]
        h = line_heights[i]
        x = x0 + ((x1 - x0) - w) / 2
        draw.text((x, y), line, font=font, fill=fill)
        y += h + 4


def arrow(draw, start, end, color, width=3):
    draw.line([start, end], fill=color, width=width)
    ex, ey = end
    sx, sy = start
    if abs(ex - sx) > abs(ey - sy):
        if ex > sx:
            pts = [(ex, ey), (ex - 12, ey - 6), (ex - 12, ey + 6)]
        else:
            pts = [(ex, ey), (ex + 12, ey - 6), (ex + 12, ey + 6)]
    else:
        if ey > sy:
            pts = [(ex, ey), (ex - 6, ey - 12), (ex + 6, ey - 12)]
        else:
            pts = [(ex, ey), (ex - 6, ey + 12), (ex + 6, ey + 12)]
    draw.polygon(pts, fill=color)


def main():
    w, h = 1800, 1150
    img = Image.new("RGB", (w, h), (10, 15, 24))
    draw = ImageDraw.Draw(img)

    title_font = load_font(46)
    h2_font = load_font(26)
    box_font = load_font(22)
    small_font = load_font(18)

    draw.text((56, 30), "System Architecture Diagram (Current Implementation)", font=title_font, fill=(225, 238, 255))
    draw.text((56, 90), "Only implemented features are shown. Conditional capabilities depend on runtime keys/packages.", font=small_font, fill=(161, 182, 212))

    ui_box = (70, 170, 470, 1040)
    be_box = (540, 170, 1170, 1040)
    data_box = (1240, 240, 1730, 610)
    ext_box = (1240, 680, 1730, 1040)

    draw_rounded_box(draw, ui_box, 20, (19, 33, 52), (63, 102, 155))
    draw_rounded_box(draw, be_box, 20, (20, 36, 58), (72, 120, 181))
    draw_rounded_box(draw, data_box, 20, (20, 41, 40), (73, 140, 132))
    draw_rounded_box(draw, ext_box, 20, (40, 30, 22), (163, 129, 84))

    draw.text((92, 188), "Frontend UI", font=h2_font, fill=(219, 233, 255))
    draw.text((560, 188), "Backend Orchestration", font=h2_font, fill=(219, 233, 255))
    draw.text((1260, 258), "Persistence", font=h2_font, fill=(206, 238, 229))
    draw.text((1260, 698), "External Providers", font=h2_font, fill=(246, 225, 199))

    ui_nodes = [
        (100, 250, 440, 330, "Chat Threads UI"),
        (100, 350, 440, 430, "Settings\n(Routing/Style/Expertise)"),
        (100, 450, 440, 530, "Attachments +\nCapability Badges"),
        (100, 550, 440, 630, "Memory Manager"),
        (100, 650, 440, 730, "Execution / Agent Trace\n(Collapsible)"),
        (100, 750, 440, 830, "Run Agent Task"),
    ]

    be_nodes = [
        (580, 250, 1130, 330, "Session Manager\n(short-term memory)"),
        (580, 350, 1130, 430, "Context Builder\n(recent + summary + memory + profile)"),
        (580, 450, 1130, 530, "Model Router\n(task / latency / cost)"),
        (580, 550, 1130, 630, "Multimodal Preprocess\n(OCR / STT, video disabled)"),
        (580, 650, 1130, 730, "Tool Executor\n(JSON schema + trace)"),
        (580, 750, 1130, 830, "Agent Loop\n(planner / executor + guardrails)"),
        (580, 850, 1130, 930, "Observability\n(request/tool/chat events)"),
    ]

    data_nodes = [
        (1270, 320, 1700, 390, "Long-term Memory DB (JSON)"),
        (1270, 410, 1700, 480, "User Profile State"),
        (1270, 500, 1700, 570, "Archived Memories"),
    ]

    ext_nodes = [
        (1270, 760, 1700, 830, "Groq Models"),
        (1270, 850, 1700, 920, "OpenAI Models"),
        (1270, 940, 1700, 1010, "Web Search Providers\nDuckDuckGo / Tavily / Brave / SerpAPI"),
    ]

    for n in ui_nodes:
        draw_rounded_box(draw, n[:4], 12, (27, 47, 74), (92, 140, 203))
        draw_multiline_centered(draw, n[:4], n[4], box_font, (226, 238, 255))

    for n in be_nodes:
        draw_rounded_box(draw, n[:4], 12, (29, 52, 82), (102, 158, 228))
        draw_multiline_centered(draw, n[:4], n[4], box_font, (229, 239, 255))

    for n in data_nodes:
        draw_rounded_box(draw, n[:4], 12, (28, 60, 58), (88, 165, 155))
        draw_multiline_centered(draw, n[:4], n[4], box_font, (218, 245, 240))

    for n in ext_nodes:
        draw_rounded_box(draw, n[:4], 12, (65, 49, 33), (181, 141, 95))
        draw_multiline_centered(draw, n[:4], n[4], box_font, (250, 233, 210))

    edge = (154, 193, 245)
    arrow(draw, (440, 290), (580, 290), edge)
    arrow(draw, (440, 390), (580, 390), edge)
    arrow(draw, (440, 490), (580, 490), edge)
    arrow(draw, (440, 590), (580, 590), edge)
    arrow(draw, (440, 690), (580, 690), edge)
    arrow(draw, (440, 790), (580, 790), edge)

    data_edge = (124, 212, 196)
    arrow(draw, (1130, 390), (1270, 350), data_edge)
    arrow(draw, (1130, 430), (1270, 445), data_edge)
    arrow(draw, (1130, 650), (1270, 535), data_edge)

    ext_edge = (221, 186, 138)
    arrow(draw, (1130, 490), (1270, 790), ext_edge)
    arrow(draw, (1130, 490), (1270, 880), ext_edge)
    arrow(draw, (1130, 650), (1270, 975), ext_edge)
    arrow(draw, (1130, 590), (1270, 975), ext_edge)

    draw.text((56, 1080), "Generated from current implemented architecture. Video pipeline remains disabled by runtime flag.", font=small_font, fill=(156, 176, 207))

    img.save("/root/code/GenAI/HW1/system-architecture-diagram.png", "PNG")


if __name__ == "__main__":
    main()

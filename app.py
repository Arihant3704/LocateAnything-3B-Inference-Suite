#!/usr/bin/env python
# -*- coding: utf-8 -*-
import gradio as gr
import cv2
import numpy as np
import os
import tempfile
import re
import time
import base64
import gc
import io
import json
import uuid
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModel, AutoTokenizer
from huggingface_hub import CommitScheduler

import spaces

_FONT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "LXGWWenKai-Bold.ttf")


def _get_first_env(*names):
    for name in names:
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
    return None


def _configure_hf_auth():
    model_token = _get_first_env(
        "MODEL_HF_TOKEN",
        "LOG_HF_TOKEN",
        "HF_TOKEN",
        "HUGGINGFACE_HUB_TOKEN",
        "HUGGINGFACEHUB_API_TOKEN",
    )
    log_token = _get_first_env(
        "LOG_HF_TOKEN",
        "MODEL_HF_TOKEN",
        "HF_TOKEN",
        "HUGGINGFACE_HUB_TOKEN",
        "HUGGINGFACEHUB_API_TOKEN",
    )
    shared_token = model_token or log_token
    if shared_token:
        # Some downstream hub calls still rely on standard env var names.
        for name in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
            os.environ[name] = shared_token
    return model_token, log_token


MODEL_HF_TOKEN, LOG_HF_TOKEN = _configure_hf_auth()


def _load_font(size=20):
    """加载中文字体（LXGW WenKai），需提前放置到 assets/ 目录"""
    if os.path.exists(_FONT_PATH):
        try:
            return ImageFont.truetype(_FONT_PATH, size)
        except Exception:
            pass
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except Exception:
        return ImageFont.load_default()


# ============================================================
# 颜色 / 解析 / 绘制
# ============================================================
def get_color_for_label(label):
    colors = [
        (8, 145, 178), (220, 38, 38), (22, 163, 74), (37, 99, 235),
        (217, 119, 6), (147, 51, 234),
    ]
    idx = sum(ord(c) for c in label)
    return colors[idx % len(colors)]


def parse_mixed_results(text, category_str=""):
    results = []
    expected_cats = [c.strip().lower() for c in category_str.split("</c>") if c.strip()]

    ref_box_pattern = r"(<ref>.*?</ref>)|(<box>.*?</box>)"
    current_label = None
    found_structured = False

    for m in re.finditer(ref_box_pattern, text, flags=re.IGNORECASE | re.DOTALL):
        token = m.group(0)
        if token.lower().startswith("<ref>"):
            label_raw = re.sub(r"</?ref>", "", token, flags=re.IGNORECASE).strip()
            if label_raw:
                current_label = label_raw
        else:
            content = re.sub(r"</?box>", "", token, flags=re.IGNORECASE)
            nums = re.findall(r"<\s*([0-9]+(?:\.[0-9]+)?)\s*>", content)
            coords = [float(n) for n in nums]
            if not coords:
                continue
            label = current_label
            if label is None:
                label = expected_cats[0] if expected_cats else "object"
            if len(coords) == 4:
                results.append({"type": "box", "coords": coords, "label": label})
            elif len(coords) == 2:
                results.append({"type": "point", "coords": coords, "label": label})
            found_structured = True

    if found_structured:
        return results

    box_pattern = r"<box>(.*?)</box>"
    parts = re.split(box_pattern, text)
    for i in range(1, len(parts), 2):
        preceding_text = parts[i - 1].lower()
        content = parts[i]
        label = expected_cats[0] if expected_cats else "object"
        for cat in expected_cats:
            if cat in preceding_text:
                label = cat
                break
        nums = re.findall(r"<\s*([0-9]+(?:\.[0-9]+)?)\s*>", content)
        coords = [float(n) for n in nums]
        if len(coords) == 4:
            results.append({"type": "box", "coords": coords, "label": label})
        elif len(coords) == 2:
            results.append({"type": "point", "coords": coords, "label": label})

    return results


def resize_image_short_side(image, short_side_size):
    w, h = image.size
    if w <= h:
        new_w = short_side_size
        scale_factor = new_w / w
        new_h = int(h * scale_factor)
    else:
        new_h = short_side_size
        scale_factor = new_h / h
        new_w = int(w * scale_factor)
    resized_image = image.resize((new_w, new_h), Image.BILINEAR)
    return resized_image, scale_factor


def draw_on_frame(frame_bgr, results, draw_label=True):
    pil_img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    img_draw = pil_img.convert("RGBA")
    overlay = Image.new("RGBA", img_draw.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    font = _load_font(20)
    w_img, h_img = pil_img.size

    parsed = []
    for res in results:
        label = res.get("label", "object")
        color = get_color_for_label(label)
        if res.get("type") == "point":
            c = res["coords"]
            cx = max(0, min(w_img, c[0] * w_img / 1000))
            cy = max(0, min(h_img, c[1] * h_img / 1000))
            parsed.append(("point", label, color, cx, cy))
            continue
        if "is_pixel" in res:
            x1, y1, bw, bh = res["coords"]
            x2, y2 = x1 + bw, y1 + bh
        else:
            c = res["coords"]
            if len(c) < 4:
                continue
            x1 = c[0] * w_img / 1000
            y1 = c[1] * h_img / 1000
            x2 = c[2] * w_img / 1000
            y2 = c[3] * h_img / 1000
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w_img, x2), min(h_img, y2)
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        parsed.append(("box", label, color, x1, y1, x2, y2))

    for item in parsed:
        if item[0] == "box":
            _, _, color, x1, y1, x2, y2 = item
            fill_color = color + (65,)
            draw.rectangle([x1, y1, x2, y2], fill=fill_color, outline=color, width=4)
        elif item[0] == "point":
            _, _, color, cx, cy = item
            r = 10
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color, outline="white", width=2)

    if draw_label:
        for item in parsed:
            if item[0] == "box":
                _, label, color, x1, y1, x2, y2 = item
                if not label:
                    continue
                t_box = draw.textbbox((0, 0), label, font=font)
                th = t_box[3] - t_box[1]
                tw = t_box[2] - t_box[0]
                pad_x, pad_y = 7, 4
                tag_h = th + pad_y * 2
                tag_w = tw + pad_x * 2
                tag_y = y1 - tag_h - 2
                if tag_y < 0:
                    tag_y = y2 + 2
                draw.rectangle([x1, tag_y, x1 + tag_w, tag_y + tag_h], fill=color)
                draw.text((x1 + pad_x, tag_y + pad_y), label, fill="white", font=font)
            elif item[0] == "point":
                _, label, color, cx, cy = item
                if not label:
                    continue
                t_box = draw.textbbox((0, 0), label, font=font)
                th, tw = t_box[3] - t_box[1], t_box[2] - t_box[0]
                tx, ty = cx + 14, cy - th // 2
                draw.rectangle([tx - 2, ty - 2, tx + tw + 6, ty + th + 4], fill=color)
                draw.text((tx + 2, ty), label, fill="white", font=font)

    combined = Image.alpha_composite(img_draw, overlay).convert("RGB")
    return cv2.cvtColor(np.array(combined), cv2.COLOR_RGB2BGR)


# ============================================================
# 模型
# ============================================================
class EagleWorker:
    def __init__(self, model_path, device="cuda", generation_mode: str = "hybrid"):
        self.model_id = model_path
        self.device = device
        self.dtype = torch.bfloat16
        self.generation_mode = generation_mode
        self.hf_token = MODEL_HF_TOKEN

        # Define set_submodule if missing (required for bitsandbytes integration in PyTorch versions that lack it)
        if not hasattr(torch.nn.Module, "set_submodule"):
            def set_submodule(self, target: str, module: torch.nn.Module) -> None:
                atoms = target.split(".")
                name = atoms.pop(-1)
                mod = self
                for item in atoms:
                    if not hasattr(mod, item):
                        raise AttributeError(mod._get_name() + " has no attribute " + item)
                    mod = getattr(mod, item)
                    if not isinstance(mod, torch.nn.Module):
                        raise TypeError("submodule " + item + " is not a Module")
                setattr(mod, name, module)
            torch.nn.Module.set_submodule = set_submodule

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            token=self.hf_token,
        )
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            token=self.hf_token,
        )

        quantization_config = None
        if "cuda" in str(device):
            try:
                import bitsandbytes
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                print("Enabling 4-bit NF4 quantization to reduce GPU memory usage.")
            except ImportError:
                print("bitsandbytes not installed, loading in default bfloat16 precision.")

        if quantization_config is not None:
            self.model = AutoModel.from_pretrained(
                model_path,
                quantization_config=quantization_config,
                _attn_implementation="sdpa",
                trust_remote_code=True,
                token=self.hf_token,
            ).eval()
        else:
            self.model = AutoModel.from_pretrained(
                model_path,
                torch_dtype=self.dtype,
                _attn_implementation="sdpa",
                trust_remote_code=True,
                token=self.hf_token,
            ).to(device).eval()
        print("Model Loaded Successfully!")

    def build_messages(self, image, categories, question_override=None):
        if question_override is not None:
            user_text = question_override
        else:
            category_set_str = "</c>".join(categories)
            user_text = f"Locate all the instances that matches the following description: {category_set_str}."
        return [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": user_text},
        ]}]

    @torch.no_grad()
    def generate(self, image, categories, generation_mode=None,
                 max_new_tokens=4096, temp=0.7, top_p=0.9, top_k=50,
                 question_override=None):
        messages = self.build_messages(image, categories, question_override=question_override)
        text = self.processor.py_apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = self.processor.process_vision_info(messages)
        inputs = self.processor(text=[text], images=images, videos=videos, return_tensors="pt").to(self.device)

        pixel_values = inputs["pixel_values"].to(self.dtype)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        image_grid_hws = inputs.get("image_grid_hws", None)

        result = self.model.generate(
            pixel_values=pixel_values, input_ids=input_ids,
            attention_mask=attention_mask, image_grid_hws=image_grid_hws,
            tokenizer=self.tokenizer, max_new_tokens=max_new_tokens,
            use_cache=True,
            generation_mode=generation_mode if generation_mode is not None else self.generation_mode,
            temperature=temp, do_sample=True, top_p=top_p,
            repetition_penalty=1.1, verbose=True,
        )

        token_sequence, out_info, output_text = [], "", ""
        if isinstance(result, tuple) and len(result) >= 3:
            output_text, token_sequence, out_info = result
            if generation_mode == "slow":
                token_sequence[-1] = ("ar", token_sequence[-1][1])
        else:
            output_text = result
        return output_text, token_sequence, out_info


# ============================================================
# 后处理 / HTML
# ============================================================
def _postprocess_detections(detections, w, h):
    valid = []
    for det in detections:
        if det["type"] == "box":
            c = det["coords"]
            rx1 = max(0, min(w - 1, int(c[0] * w / 1000)))
            ry1 = max(0, min(h - 1, int(c[1] * h / 1000)))
            rx2 = max(0, min(w - 1, int(c[2] * w / 1000)))
            ry2 = max(0, min(h - 1, int(c[3] * h / 1000)))
            box_w, box_h = rx2 - rx1, ry2 - ry1
            if box_w <= 0 or box_h <= 0:
                continue
            valid.append({"type": "box", "coords": [rx1, ry1, box_w, box_h],
                          "is_pixel": True, "label": det["label"]})
        elif det["type"] == "point":
            valid.append(det)
    return valid


def _parse_out_info_dict(out_info: str) -> dict:
    stats = {}
    if not out_info:
        return stats
    cleaned = re.sub(r"^[Ss]tast?ic\s*[Ii]nfo\s*,?\s*", "", out_info.strip())
    for part in cleaned.split(";"):
        part = part.strip()
        if "=" in part:
            k, v = part.split("=", 1)
            stats[k.strip()] = v.strip()
    return stats


def generate_dynamic_html(token_sequence, out_info, raw_text):
    uid = f"a{int(time.time() * 1000)}"
    css = f"""
    <style>
        .dc-root {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            border: 1px solid #cce875; border-radius: 10px; background: #ffffff; overflow: hidden;
        }}
        .dc-header {{
            display: flex; align-items: center; justify-content: space-between;
            padding: 12px 18px;
            background: linear-gradient(135deg, #76b900 0%, #649d00 100%);
            border-bottom: 1px solid #527f00;
        }}
        .dc-header-title {{ font-weight: 700; font-size: 0.95em; color: #ffffff !important; letter-spacing: 0.3px; }}
        .dc-legend {{ display: flex; gap: 16px; align-items: center; }}
        .dc-legend-item {{ display: flex; align-items: center; gap: 5px; font-size: 0.78em; color: rgba(255,255,255,0.92); font-weight: 500; }}
        .dc-legend-dot {{ width: 10px; height: 10px; border-radius: 3px; display: inline-block; border: 1px solid rgba(255,255,255,0.5); }}
        .dc-row {{ display: flex; gap: 10px; padding: 14px 18px; border-bottom: 1px solid #eef7d1; }}
        .dc-row:last-child {{ border-bottom: none; }}
        .dc-val {{ flex: 1; line-height: 2.3; word-wrap: break-word; color: #4b5563; font-size: 0.92em; }}
        @keyframes tk-{uid} {{
            0%   {{ opacity: 0; transform: translateY(8px) scale(0.92); }}
            60%  {{ opacity: 1; transform: translateY(-2px) scale(1.02); }}
            100% {{ opacity: 1; transform: translateY(0) scale(1); }}
        }}
        .tk-mtp-{uid}, .tk-ar-{uid} {{
            opacity: 0; animation: tk-{uid} 0.35s ease-out forwards;
            border-radius: 5px; padding: 2px 7px; margin: 2px 1px; display: inline-block;
            font-size: 0.80em; font-weight: 600;
            font-family: 'SFMono-Regular', Consolas, 'Courier New', monospace; white-space: nowrap;
        }}
        .tk-mtp-{uid} {{ background: #e8f5e9; border: 2px solid #76b900; color: #2d4400; box-shadow: 0 1px 2px rgba(118,185,0,0.15); }}
        .tk-ar-{uid} {{ background: #fff3e0; border: 2px solid #e65100; color: #bf360c; box-shadow: 0 1px 2px rgba(230,81,0,0.15); }}
        .tk-stat-{uid} {{
            opacity: 0; animation: tk-{uid} 0.4s ease-out forwards;
            background: #f0f9e2; border: 1px solid #a4d422; border-radius: 6px;
            padding: 5px 14px; display: inline-block; font-size: 0.82em; color: #3f6200; font-weight: 600;
        }}
        .dc-raw {{ padding: 0 18px 14px; }}
        .dc-raw summary {{ cursor: pointer; color: #9ca3af; font-size: 0.82em; user-select: none; transition: color .15s; }}
        .dc-raw summary:hover {{ color: #649d00; }}
        .dc-raw-pre {{
            background: #f7fbe8; border: 1px solid #ddf0a3; border-radius: 6px;
            padding: 12px; margin-top: 8px;
            font-family: 'SFMono-Regular', Consolas, 'Courier New', monospace;
            font-size: 0.78em; color: #374151; white-space: pre-wrap; word-break: break-all;
            max-height: 200px; overflow-y: auto;
        }}
        @media (max-width: 640px) {{
            .dc-header {{ flex-direction: column; gap: 8px; align-items: flex-start; }}
            .dc-row {{ flex-direction: column; gap: 4px; }}
        }}
    </style>
    """
    h = css + '<div class="dc-root">'
    h += ('<div class="dc-header">'
          '<span class="dc-header-title">LocateAnything Decoding Trace</span>'
          '<div class="dc-legend">'
          '<div class="dc-legend-item"><span class="dc-legend-dot" style="background:#76b900;"></span>MTP &mdash; Parallel Box Decoding</div>'
          '<div class="dc-legend-item"><span class="dc-legend-dot" style="background:#e65100;"></span>AR &mdash; NTP Fallback (Re-decoding)</div>'
          '</div></div>')
    h += '<div class="dc-row"><div class="dc-val">'
    tok_idx = 0
    if token_sequence:
        for item in token_sequence:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            decode_type = str(item[0]).lower()
            text = str(item[1])
            safe = text.replace("<", "&lt;").replace(">", "&gt;")
            delay = f"{tok_idx * 0.06:.2f}s"
            cls = f"tk-ar-{uid}" if decode_type == "ar" else f"tk-mtp-{uid}"
            h += f'<span class="{cls}" style="animation-delay:{delay}">{safe}</span> '
            tok_idx += 1
    h += '</div></div>'
    if out_info:
        stats = _parse_out_info_dict(out_info)
        bits = []
        if "generate_time(s)" in stats:
            bits.append(f"Inference Time: {stats['generate_time(s)']}s")
        if "tps" in stats:
            bits.append(f"FPS: {stats['tps']} frames/s (tok/s)")
        if "forward_step" in stats: bits.append(f"{stats['forward_step']} steps")
        if "num_tokens" in stats: bits.append(f"{stats['num_tokens']} tokens")
        if "num_boxes" in stats: bits.append(f"{stats['num_boxes']} boxes")
        if "switch_to_ar" in stats:
            n = stats["switch_to_ar"]
            bits.append(f"{n} AR Fallback{'s' if n != '1' else ''}")
        if "ar_step" in stats: bits.append(f"{stats['ar_step']} AR steps")
        if "bps" in stats: bits.append(f"{stats['bps']} box/s")
        summary = " &middot; ".join(bits) if bits else out_info.strip()
        stat_delay = f"{tok_idx * 0.06 + 0.3:.2f}s"
        h += (f'<div class="dc-row" style="justify-content:flex-end;padding-top:4px;padding-bottom:10px;border-bottom:none;">'
              f'<span class="tk-stat-{uid}" style="animation-delay:{stat_delay}">⚡ {summary}</span></div>')
    if raw_text:
        safe_raw = raw_text.replace("<", "&lt;").replace(">", "&gt;")
        h += (f'<div class="dc-raw"><details><summary>📄 Show Raw Response</summary>'
              f'<div class="dc-raw-pre">{safe_raw}</div></details></div>')
    h += '</div>'
    return h


def generate_raw_prompt(task_type, category):
    if not category:
        category = "objects"
    cats = "</c>".join(c.strip() for c in category.split(",") if c.strip())
    if task_type == "Detection":
        return f"Locate all the instances that matches the following description: {cats}."
    elif task_type == "Grounding":
        return f"Locate all the instances that match the following description: {cats}."
    elif task_type == "OCR":
        return "Detect all the text in box format."
    elif task_type == "GUI":
        return f"Locate the region that matches the following description: {cats}."
    elif task_type == "Pointing":
        return f"Point to: {cats}."
    else:
        return f"Locate all the instances that matches the following description: {cats}."


class MockWorker:
    def __init__(self):
        print("MockWorker initialized. Running in Mock Mode.")

    def generate(self, image, categories, generation_mode=None,
                 max_new_tokens=4096, temp=0.7, top_p=0.9, top_k=50,
                 question_override=None):
        category_str = categories[0] if categories else "object"
        output_text = ""
        token_sequence = []
        out_info = "forward_step=1; num_tokens=15; num_boxes=1; tps=45; bps=3; generate_time(s)=0.33"
        
        # Simple heuristic based on categories
        cat_lower = [c.lower() for c in categories]
        if "book" in cat_lower:
            output_text = "<ref>book</ref><box><200><300><800><700></box>"
            token_sequence = [("mtp", "<ref>"), ("mtp", "book"), ("mtp", "</ref>"), ("mtp", "<box>"), ("mtp", "<200>"), ("mtp", "<300>"), ("mtp", "<800>"), ("mtp", "<700>"), ("mtp", "</box>")]
        elif "sweet" in cat_lower:
            output_text = "<ref>sweet</ref><box><150><200><450><500></box><ref>sweet</ref><box><500><400><850><800></box>"
            token_sequence = [("mtp", "<ref>"), ("mtp", "sweet"), ("mtp", "</ref>"), ("mtp", "<box>"), ("mtp", "<150>"), ("mtp", "<200>"), ("mtp", "<450>"), ("mtp", "<500>"), ("mtp", "</box>"),
                              ("mtp", "<ref>"), ("mtp", "sweet"), ("mtp", "</ref>"), ("mtp", "<box>"), ("mtp", "<500>"), ("mtp", "<400>"), ("mtp", "<850>"), ("mtp", "<800>"), ("mtp", "</box>")]
        elif "person" in cat_lower:
            output_text = "<ref>person</ref><box><350><100><850><900></box>"
            token_sequence = [("mtp", "<ref>"), ("mtp", "person"), ("mtp", "</ref>"), ("mtp", "<box>"), ("mtp", "<350>"), ("mtp", "<100>"), ("mtp", "<850>"), ("mtp", "<900>"), ("mtp", "</box>")]
        elif "text" in cat_lower or "ocr" in cat_lower:
            output_text = "<ref>text</ref><box><200><250><800><450></box><ref>text</ref><box><300><500><700><650></box>"
            token_sequence = [("mtp", "<ref>"), ("mtp", "text"), ("mtp", "</ref>"), ("mtp", "<box>"), ("mtp", "<200>"), ("mtp", "<250>"), ("mtp", "<800>"), ("mtp", "<450>"), ("mtp", "</box>"),
                              ("mtp", "<ref>"), ("mtp", "text"), ("mtp", "</ref>"), ("mtp", "<box>"), ("mtp", "<300>"), ("mtp", "<500>"), ("mtp", "<700>"), ("mtp", "<650>"), ("mtp", "</box>")]
        else:
            # General mockup bounding box for any custom inputs
            output_text = f"<ref>{category_str}</ref><box><300><300><700><700></box>"
            token_sequence = [("mtp", "<ref>"), ("mtp", category_str), ("mtp", "</ref>"), ("mtp", "<box>"), ("mtp", "<300>"), ("mtp", "<300>"), ("mtp", "<700>"), ("mtp", "<700>"), ("mtp", "</box>")]

        return output_text, token_sequence, out_info

# ============================================================
# 模型初始化
# ============================================================
try:
    if os.environ.get("MOCK_MODE") == "1":
        raise ValueError("MOCK_MODE environment variable is set to 1")
    MODEL_PATH = os.environ.get("MODEL_PATH", "nvidia/LocateAnything-3B")
    GLOBAL_WORKER = EagleWorker(MODEL_PATH)
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"Failed to load model: {e}. Will run in Mock Mode.")
    GLOBAL_WORKER = MockWorker()


# ============================================================
# 用户数据收集（HuggingFace Public Dataset）
#
# 策略：one-record-per-file，配合按天目录 + 容器级 SESSION_ID。
# 这样可以解决两个问题：
#   1. 容器被回收时，本地 ephemeral 目录被清空。原来所有 session 都
#      写同一个 logs_<date>.jsonl，新容器起来后会用空文件把 dataset 里
#      旧的同名文件覆盖掉，造成数据丢失。
#   2. 每次 commit 都要重传整份 LFS（appended 文件 hash 变了），浪费带宽。
#
# 现在每条记录写成独立的 JSONL 文件：
#   data/<date>/<SESSION_ID>__<entry_id>.jsonl
# CommitScheduler 只会“新增”文件，永远不会覆盖其它 session 的数据；
# 单文件上传后即被封存，不会重复上传。
# ============================================================
LOG_DATASET_REPO = os.environ.get("LOG_DATASET_REPO", "woshichaoren123/log")
_LOG_DIR = Path(tempfile.mkdtemp(prefix="hf_log_"))
_SESSION_ID = uuid.uuid4().hex[:8]
_log_scheduler = None

if LOG_DATASET_REPO and LOG_HF_TOKEN:
    try:
        _log_scheduler = CommitScheduler(
            repo_id=LOG_DATASET_REPO,
            repo_type="dataset",
            folder_path=str(_LOG_DIR),
            path_in_repo="data",
            every=3,
            token=LOG_HF_TOKEN,
            squash_history=False,
        )
        print(f"[LOG] Dataset logging enabled → {LOG_DATASET_REPO} "
              f"(session={_SESSION_ID}, dir={_LOG_DIR})")
    except Exception as e:
        _log_scheduler = None
        print(f"[LOG] Dataset logging disabled: {e}")
else:
    print("[LOG] Dataset logging disabled (LOG_HF_TOKEN not set)")


def _pil_to_b64(pil_img):
    """将 PIL 图片无损转为 PNG base64 字符串。"""
    buf = io.BytesIO()
    pil_img.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _atomic_write_text(path: Path, text: str):
    """原子写入：先写临时文件再 rename，避免 CommitScheduler 读到半截文件。"""
    tmp_path = path.with_name(path.name + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp_path, path)


def _log_to_dataset(
    input_type, category, model_mode, raw_prompt,
    output_text="", input_image=None, output_image=None,
    extra=None,
):
    """每条记录写到独立的 JSONL 文件，按日期分目录、文件名包含 session_id。

    最终落盘路径（也是 dataset 里的路径）：
        data/<YYYY-MM-DD>/<session_id>__<entry_id>.jsonl
    """
    if _log_scheduler is None:
        return
    try:
        entry_id = f"{int(time.time())}_{uuid.uuid4().hex[:6]}"
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        date_str = time.strftime("%Y-%m-%d", time.gmtime())

        input_b64 = None
        if input_image is not None and isinstance(input_image, Image.Image):
            input_b64 = _pil_to_b64(input_image)

        output_b64 = None
        if output_image is not None and isinstance(output_image, Image.Image):
            output_b64 = _pil_to_b64(output_image)

        record = {
            "id": entry_id,
            "session_id": _SESSION_ID,
            "timestamp": ts,
            "input_type": input_type,
            "category": category,
            "model_mode": model_mode,
            "raw_prompt": raw_prompt,
            "output_text": output_text,
            "input_image_b64": input_b64,
            "output_image_b64": output_b64,
        }
        if extra:
            record.update(extra)

        day_dir = _LOG_DIR / date_str
        day_dir.mkdir(parents=True, exist_ok=True)
        log_file = day_dir / f"{_SESSION_ID}__{entry_id}.jsonl"

        payload = json.dumps(record, ensure_ascii=False) + "\n"
        with _log_scheduler.lock:
            _atomic_write_text(log_file, payload)
    except Exception as e:
        print(f"[LOG] Failed to log to dataset: {e}")


# ============================================================
# 公用预处理
# ============================================================
def _prepare_image_for_model(pil_img, short_size):
    process_img = pil_img.copy()
    if short_size is not None and short_size > 0:
        process_img, _ = resize_image_short_side(process_img, min(int(short_size), 1024))
    else:
        if min(process_img.size) > 1024:
            process_img, _ = resize_image_short_side(process_img, 1024)
    return process_img


# ============================================================
# GPU 时间预算常量（按模式区分）
# ============================================================
GPU_HARD_LIMIT_IMAGE = 30     # Image 模式 @spaces.GPU(duration=...)
GPU_HARD_LIMIT_VIDEO = 240    # Video 模式 @spaces.GPU(duration=...)
PHASE2_RESERVE = 55           # 留给 Phase 2（绘制 + ffmpeg）的秒数
SAFETY_MARGIN = 25            # 额外安全裕量，永远不要触碰硬限制
INFERENCE_BUDGET = GPU_HARD_LIMIT_VIDEO - PHASE2_RESERVE - SAFETY_MARGIN
EST_SECONDS_PER_FRAME = 20    # 保守估计：每帧推理耗时


# ============================================================
# ✅ 图像推理（独立函数）
# ============================================================
def _run_image_inference(
    image_in, categories_list, category_str,
    model_mode, temp, top_p, top_k, short_size, question_override,
    progress=None,  # 接收 progress
):
    if image_in is None:
        return (
            gr.update(value=None, visible=True),
            gr.update(value=None, visible=False),
            "<p style='color:#ef4444;padding:12px;'>⚠️ Please upload an image first.</p>",
        )

    if progress is not None:  # 进度提示
        progress(0.1, desc="Preprocessing image ...")

    process_img = _prepare_image_for_model(image_in, short_size)

    if progress is not None:
        progress(0.2, desc="Running model inference ...")

    if GLOBAL_WORKER:
        output_text, token_sequence, out_info = GLOBAL_WORKER.generate(
            process_img, categories_list, model_mode,
            temp=temp, top_p=top_p, top_k=top_k,
            question_override=question_override,
        )
    else:
        output_text, token_sequence, out_info = "", [], ""

    if progress is not None:
        progress(0.8, desc="Drawing results ...")

    detections = parse_mixed_results(output_text, category_str)
    frame_bgr = cv2.cvtColor(np.array(image_in), cv2.COLOR_RGB2BGR)
    out_img_bgr = draw_on_frame(frame_bgr, detections, draw_label=True)
    output_image = Image.fromarray(cv2.cvtColor(out_img_bgr, cv2.COLOR_BGR2RGB))
    html = generate_dynamic_html(token_sequence, out_info, output_text)

    _log_to_dataset(
        input_type="image",
        category=", ".join(categories_list),
        model_mode=model_mode,
        raw_prompt=question_override or category_str,
        output_text=output_text,
        input_image=image_in,
        output_image=output_image,
    )

    if progress is not None:
        progress(1.0, desc="Done!")

    return (
        gr.update(value=output_image, visible=True),
        gr.update(value=None, visible=False),
        html,
    )


# ============================================================
# ✅ 视频推理（独立函数 — 带完整超时保护）
# ============================================================
def _run_video_inference(
    video_in, categories_list, category_str,
    model_mode, temp, top_p, top_k, short_size, question_override,
    max_video_frames,  # 可调帧数
    progress=None,     # 接收 progress
):
    import subprocess as _sp

    if video_in is None:
        return (
            gr.update(value=None, visible=False),
            gr.update(value=None, visible=True),
            "<p style='color:#ef4444;padding:12px;'>⚠️ Please upload a video first.</p>",
        )

    total_start = time.time()
    max_frames = int(max_video_frames) if max_video_frames else 4

    if progress is not None:
        progress(0.0, desc="Reading video ...")

    # ---------- 读取视频 ----------
    t0 = time.time()
    cap = cv2.VideoCapture(video_in)
    fps = cap.get(cv2.CAP_PROP_FPS)
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    all_frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(frame)
    cap.release()
    total = len(all_frames)
    read_elapsed = time.time() - t0
    print(f"[TIMING] Video read: {read_elapsed:.2f}s, total frames={total}, "
          f"resolution={vid_w}x{vid_h}, fps={fps:.1f}")

    if total == 0:
        return (
            gr.update(value=None, visible=False),
            gr.update(value=None, visible=True),
            "<p style='color:#ef4444;padding:12px;'>⚠️ Failed to read any frames from the video.</p>",
        )

    # ---------- 采样帧 ----------
    if total <= max_frames:
        sample_indices = list(range(total))
    else:
        sample_indices = [int(round(i * (total - 1) / (max_frames - 1)))
                          for i in range(max_frames)]

    sampled_frames = [all_frames[i] for i in sample_indices]
    n_sampled = len(sampled_frames)

    # ============================================================
    # 🛡️ 预估检查：在开跑前判断能不能在 GPU 时间预算内跑完
    # ============================================================
    time_already_used = time.time() - total_start
    available_for_inference = GPU_HARD_LIMIT_VIDEO - time_already_used - PHASE2_RESERVE - SAFETY_MARGIN
    estimated_inference_time = n_sampled * EST_SECONDS_PER_FRAME

    if estimated_inference_time > available_for_inference:
        # 尝试自动缩减帧数
        max_feasible = max(0, int(available_for_inference // EST_SECONDS_PER_FRAME))
        print(f"[PRE-CHECK] Estimated {estimated_inference_time:.0f}s > budget {available_for_inference:.0f}s, "
              f"reducing from {n_sampled} to {max_feasible} frames")

        if max_feasible < 1:
            # 连 1 帧都跑不了，直接拒绝
            del all_frames
            gc.collect()
            return (
                gr.update(value=None, visible=False),
                gr.update(value=None, visible=True),
                "<div style='background:#fef2f2;border:1px solid #fca5a5;border-radius:8px;"
                "padding:16px;margin:8px 0;'>"
                "<p style='color:#dc2626;font-weight:700;font-size:1.05em;margin:0 0 8px;'>"
                "⚠️ Video too large to process</p>"
                f"<p style='color:#7f1d1d;margin:0;font-size:0.92em;'>"
                f"This video has <b>{total}</b> frames. "
                f"Even processing <b>1</b> sampled frame (~{EST_SECONDS_PER_FRAME}s) "
                f"would exceed the <b>{GPU_HARD_LIMIT_VIDEO}s</b> GPU time limit.<br><br>"
                "💡 <b>Suggestions:</b> use a shorter / lower-resolution video, "
                "or switch to <b>Image</b> mode with a single frame screenshot.</p></div>",
            )

        # 用缩减后的帧数重新采样
        if total <= max_feasible:
            sample_indices = list(range(total))
        else:
            sample_indices = [int(round(i * (total - 1) / (max_feasible - 1)))
                              for i in range(max_feasible)]
        sampled_frames = [all_frames[i] for i in sample_indices]
        n_sampled = len(sampled_frames)

    # 释放原始帧列表，节省内存
    out_fps = max(1.0, n_sampled / (total / fps)) if fps > 0 else 5.0
    del all_frames
    gc.collect()

    print(f"[TIMING] Sampled {n_sampled} frames, output fps: {out_fps:.2f}")

    # ============================================================
    # 阶段一：推理（逐帧检查剩余时间）
    # ============================================================
    print("=" * 60)
    print("[PHASE 1] Starting model inference ...")
    print("=" * 60)

    inference_results = []
    phase1_start = time.time()
    processed_count = 0
    early_stopped = False
    early_stop_reason = ""

    for i, frame in enumerate(sampled_frames):
        # ---- 🛡️ 运行时时间检查：还够不够跑下一帧 + Phase 2？----
        elapsed_since_start = time.time() - total_start
        remaining_total = GPU_HARD_LIMIT_VIDEO - elapsed_since_start

        if remaining_total < PHASE2_RESERVE + SAFETY_MARGIN:
            early_stopped = True
            early_stop_reason = (
                f"GPU time budget is running out: "
                f"{elapsed_since_start:.0f}s used, only {remaining_total:.0f}s left "
                f"(need ≥{PHASE2_RESERVE}s for video encoding). "
                f"Successfully processed {processed_count}/{n_sampled} frames."
            )
            print(f"[⏰ EARLY STOP] {early_stop_reason}")
            break

        if progress is not None:
            progress(
                (i / n_sampled) * 0.85,
                desc=f"🧠 Inference: frame {i + 1}/{n_sampled} "
                     f"(⏱️ {elapsed_since_start:.0f}s / {GPU_HARD_LIMIT_VIDEO}s) ...",
            )

        frame_t0 = time.time()

        # 预处理
        prep_t0 = time.time()
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        process_img = _prepare_image_for_model(pil_img, short_size)
        prep_time = time.time() - prep_t0

        # 推理
        infer_t0 = time.time()
        if GLOBAL_WORKER:
            output_text, _, _ = GLOBAL_WORKER.generate(
                process_img, categories_list, model_mode,
                temp=temp, top_p=top_p, top_k=top_k,
                question_override=question_override,
            )
        else:
            output_text = ""
        infer_time = time.time() - infer_t0

        inference_results.append(output_text)
        processed_count += 1

        # 清理 GPU 缓存
        cleanup_t0 = time.time()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        cleanup_time = time.time() - cleanup_t0

        total_frame_time = time.time() - frame_t0
        print(f"[PHASE 1] Frame {i + 1}/{n_sampled} done: "
              f"prep={prep_time:.2f}s, infer={infer_time:.2f}s, "
              f"cleanup={cleanup_time:.2f}s, total={total_frame_time:.2f}s")
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"         GPU mem: allocated={allocated:.2f}GB, reserved={reserved:.2f}GB")

    phase1_time = time.time() - phase1_start
    print(f"[PHASE 1] COMPLETE: {phase1_time:.2f}s for {processed_count} frames "
          f"({phase1_time / max(processed_count, 1):.2f}s/frame)")

    # 如果 1 帧都没处理完，返回错误
    if processed_count == 0:
        return (
            gr.update(value=None, visible=False),
            gr.update(value=None, visible=True),
            "<div style='background:#fef2f2;border:1px solid #fca5a5;border-radius:8px;"
            "padding:16px;margin:8px 0;'>"
            "<p style='color:#dc2626;font-weight:700;font-size:1.05em;margin:0 0 8px;'>"
            "⚠️ Could not process any frames</p>"
            "<p style='color:#7f1d1d;margin:0;font-size:0.92em;'>"
            "The GPU time limit was reached before even one frame could be processed. "
            "Please try a lower resolution video or use Image mode instead.</p></div>",
        )

    # 裁剪到实际处理过的帧
    sampled_frames_for_draw = sampled_frames[:processed_count]
    inference_results_for_draw = inference_results[:processed_count]

    # ============================================================
    # 阶段二：绘制 + 编码（只处理已推理完的帧）
    # ============================================================
    if progress is not None:
        progress(0.88, desc="🎨 Drawing & encoding video ...")

    print("=" * 60)
    print(f"[PHASE 2] Drawing & video encoding ({processed_count} frames) ...")
    print("=" * 60)

    phase2_start = time.time()
    tmp_raw = tempfile.mktemp(suffix=".raw.mp4")
    out_video_path = tempfile.mktemp(suffix=".mp4")
    out = cv2.VideoWriter(tmp_raw, cv2.VideoWriter_fourcc(*"mp4v"),
                          out_fps, (vid_w, vid_h))

    for i, (frame, output_text) in enumerate(
            zip(sampled_frames_for_draw, inference_results_for_draw)):
        draw_t0 = time.time()
        detections = parse_mixed_results(output_text, category_str)
        valid_results = _postprocess_detections(detections, vid_w, vid_h)
        frame_to_draw = draw_on_frame(frame, valid_results, draw_label=True)
        out.write(frame_to_draw)
        draw_time = time.time() - draw_t0
        print(f"[PHASE 2] Frame {i + 1}/{processed_count}: "
              f"draw={draw_time:.3f}s, det={len(valid_results)}")

    out.release()
    phase2_draw_time = time.time() - phase2_start

    # ---- ffmpeg 重编码（如果还有时间的话） ----
    elapsed_now = time.time() - total_start
    remaining_now = GPU_HARD_LIMIT_VIDEO - elapsed_now

    if progress is not None:
        progress(0.95, desc="📦 Re-encoding with ffmpeg ...")

    ffmpeg_t0 = time.time()
    if remaining_now > 15:
        # 还有时间，用 ffmpeg 重编码（兼容性更好）
        try:
            ffmpeg_timeout = max(10, int(remaining_now - 5))
            _sp.run(
                ["ffmpeg", "-y", "-i", tmp_raw, "-c:v", "libx264",
                 "-preset", "ultrafast", "-crf", "23", "-pix_fmt", "yuv420p",
                 "-movflags", "+faststart", out_video_path],
                check=True, capture_output=True, timeout=ffmpeg_timeout,
            )
            os.remove(tmp_raw)
        except Exception as ffmpeg_err:
            print(f"[PHASE 2] ffmpeg failed or timed out: {ffmpeg_err}, using raw file")
            if os.path.exists(tmp_raw):
                os.replace(tmp_raw, out_video_path)
    else:
        # 时间不够了，直接用 mp4v 原始文件
        os.replace(tmp_raw, out_video_path)
        print("[PHASE 2] Skipped ffmpeg re-encoding due to time constraint")

    ffmpeg_time = time.time() - ffmpeg_t0
    total_time = time.time() - total_start

    print("=" * 60)
    print(f"[TOTAL] {total_time:.2f}s  |  inference={phase1_time:.2f}s  "
          f"draw={phase2_draw_time:.2f}s  ffmpeg={ffmpeg_time:.2f}s  "
          f"frames_done={processed_count}/{n_sampled}")
    print("=" * 60)

    # ---- 构建结果 HTML ----
    warning_html = ""
    if early_stopped:
        warning_html = (
            "<div style='background:#fefce8;border:1px solid #fde047;border-radius:8px;"
            "padding:14px;margin-bottom:12px;'>"
            "<p style='color:#a16207;font-weight:700;font-size:1.02em;margin:0 0 6px;'>"
            "⚡ Partial Result — Early Stop Due to GPU Time Limit</p>"
            f"<p style='color:#854d0e;margin:0;font-size:0.9em;'>{early_stop_reason}</p>"
            "<p style='color:#854d0e;margin:6px 0 0;font-size:0.88em;'>"
            "💡 <b>Tip:</b> Reduce <b>Max Video Frames</b> slider or use a shorter video "
            "to process all frames within the GPU budget.</p>"
            "</div>"
        )

    timing_summary = (
        f"Video: {total} total frames, sampled {n_sampled}, "
        f"processed {processed_count} | "
        f"Inference: {phase1_time:.1f}s ({phase1_time / max(processed_count, 1):.1f}s/frame) | "
        f"Drawing: {phase2_draw_time:.1f}s | ffmpeg: {ffmpeg_time:.1f}s | "
        f"Total: {total_time:.1f}s / {GPU_HARD_LIMIT_VIDEO}s budget"
    )
    html = warning_html + generate_dynamic_html(
        token_sequence=[], out_info="", raw_text=timing_summary)

    try:
        thumb = Image.fromarray(
            cv2.cvtColor(sampled_frames_for_draw[0], cv2.COLOR_BGR2RGB))
    except Exception:
        thumb = None
    _log_to_dataset(
        input_type="video",
        category=", ".join(categories_list),
        model_mode=model_mode,
        raw_prompt=question_override or category_str,
        output_text="\n---\n".join(inference_results_for_draw),
        input_image=thumb,
        extra={
            "video_total_frames": total,
            "video_sampled_frames": n_sampled,
            "video_processed_frames": processed_count,
        },
    )

    if progress is not None:
        progress(1.0, desc="Done!")

    return (
        gr.update(value=None, visible=False),
        gr.update(value=out_video_path, visible=True),
        html,
    )


# ============================================================
# 🛡️ 主入口：按模式分配不同 GPU 时长
# ============================================================

def _build_error_html(e, gpu_limit, input_type):
    """统一的异常→友好 HTML 构建。"""
    import traceback
    traceback.print_exc()

    error_type = type(e).__name__
    error_msg = str(e)

    is_timeout = ("timeout" in error_msg.lower()
                  or "timelimit" in error_msg.lower()
                  or "time limit" in error_msg.lower()
                  or "duration" in error_msg.lower())

    if is_timeout:
        detail = (
            f"The GPU time limit ({gpu_limit}s) was exceeded before the result "
            "could be fully assembled. This typically happens with large videos."
        )
        suggestion = (
            "Please reduce <b>Max Video Frames</b>, use a shorter / smaller video, "
            "or switch to <b>Image</b> mode."
        )
    else:
        detail = f"{error_type}: {error_msg}"
        suggestion = (
            "If the problem persists, try reducing video size or "
            "switching to Image mode."
        )

    error_html = (
        "<div style='background:#fef2f2;border:1px solid #fca5a5;border-radius:8px;"
        "padding:16px;margin:8px 0;'>"
        "<p style='color:#dc2626;font-weight:700;font-size:1.05em;margin:0 0 8px;'>"
        "⚠️ Processing interrupted</p>"
        f"<p style='color:#7f1d1d;margin:0 0 8px;font-size:0.92em;'>{detail}</p>"
        f"<p style='color:#7f1d1d;margin:0;font-size:0.88em;'>💡 {suggestion}</p>"
        "</div>"
    )

    return (
        gr.update(value=None, visible=(input_type == "Image")),
        gr.update(value=None, visible=(input_type == "Video")),
        error_html,
    )


@spaces.GPU(duration=GPU_HARD_LIMIT_IMAGE)
def _run_image_gpu(
    image_in, category, model_mode, temp, top_p, top_k,
    short_size, question_override, progress,
):
    try:
        categories_list = [c.strip() for c in category.split(",") if c.strip()]
        category_str = "</c>".join(categories_list)
        return _run_image_inference(
            image_in, categories_list, category_str,
            model_mode, temp, top_p, top_k, short_size, question_override,
            progress=progress,
        )
    except Exception as e:
        return _build_error_html(e, GPU_HARD_LIMIT_IMAGE, "Image")


@spaces.GPU(duration=GPU_HARD_LIMIT_VIDEO)
def _run_video_gpu(
    video_in, category, model_mode, temp, top_p, top_k,
    short_size, question_override, max_video_frames, progress,
):
    try:
        categories_list = [c.strip() for c in category.split(",") if c.strip()]
        category_str = "</c>".join(categories_list)
        return _run_video_inference(
            video_in, categories_list, category_str,
            model_mode, temp, top_p, top_k, short_size, question_override,
            max_video_frames=max_video_frames,
            progress=progress,
        )
    except Exception as e:
        return _build_error_html(e, GPU_HARD_LIMIT_VIDEO, "Video")


def run_inference(
    input_type, image_in, video_in, task_type, category,
    model_mode, temp, top_p, top_k, short_size, question_override,
    max_video_frames,
    progress=gr.Progress(track_tqdm=False),
):
    if input_type == "Image":
        return _run_image_gpu(
            image_in, category, model_mode, temp, top_p, top_k,
            short_size, question_override, progress,
        )
    else:
        return _run_video_gpu(
            video_in, category, model_mode, temp, top_p, top_k,
            short_size, question_override, max_video_frames, progress,
        )


# ============================================================
# 按钮状态
# ============================================================
def _disable_run_btn():
    return gr.update(interactive=False, value="⏳ Running ...")


def _enable_run_btn():
    return gr.update(interactive=True, value="🧠 Run Inference")


# ============================================================
# Examples
# ============================================================
EXAMPLE_CONFIGS = [
    {"name": "Book", "input_type": "Image", "image": "./assets/book.jpg", "video": None,
     "task": "Detection", "category": "book", "mode": "hybrid"},
    {"name": "Sweet", "input_type": "Image", "image": "./assets/sweet.jpg", "video": None,
     "task": "Detection", "category": "sweet", "mode": "hybrid"},
    {"name": "Person", "input_type": "Image", "image": "./assets/person.jpg", "video": None,
     "task": "Detection", "category": "person", "mode": "hybrid"},
    {"name": "OCR", "input_type": "Image", "image": "./assets/ocr.jpg", "video": None,
     "task": "OCR", "category": "text", "mode": "fast"},
]


def prepare_gallery_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    gallery_images, gallery_captions = [], []
    for config in EXAMPLE_CONFIGS:
        img_path = (os.path.normpath(os.path.join(base_dir, config["image"]))
                    if config["image"] else None)
        if img_path and os.path.exists(img_path):
            gallery_images.append(img_path)
        else:
            gallery_images.append(Image.new("RGB", (200, 200), color="black"))
        gallery_captions.append(config["name"])
    return gallery_images, gallery_captions


def update_example_selection(evt: gr.SelectData):
    config = EXAMPLE_CONFIGS[evt.index]
    base_dir = os.path.dirname(os.path.abspath(__file__))
    img_path = (os.path.normpath(os.path.join(base_dir, config["image"]))
                if config["image"] else None)
    vid_path = (os.path.normpath(os.path.join(base_dir, config["video"]))
                if config["video"] else None)
    return (
        config["input_type"],
        gr.update(value=img_path, visible=(config["input_type"] == "Image")),
        gr.update(value=vid_path, visible=(config["input_type"] == "Video")),
        config["task"], config["category"], config["mode"],
    )


# ============================================================
# UI
# ============================================================
def create_demo():
    nv_green = gr.themes.Color(
        c50="#f7fbe8", c100="#eef7d1", c200="#ddf0a3",
        c300="#cce875", c400="#a4d422", c500="#76b900",
        c600="#649d00", c700="#527f00", c800="#3f6200",
        c900="#2d4400", c950="#1a2700",
    )
    with gr.Blocks(
        theme=gr.themes.Soft(primary_hue=nv_green, secondary_hue=nv_green),
        title="LocateAnything",
    ) as demo:
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("# 🚀 LocateAnything")
                gr.Markdown(
                    "> **Locate any object in images or videos with natural language.**  \n"
                    "> Upload an image/video on the left, choose a task type, enter what you want to find, "
                    "then click **Run Inference**. Results with bounding boxes will appear on the right.\n"
                    ">\n"
                    "> **Quick Start:** "
                    "① Select *Image* or *Video* → "
                    "② Pick a *Task Type* (Detection / Grounding / OCR / GUI / Pointing) → "
                    "③ Type your *Categories* (comma-separated) → "
                    "④ Click **🧠 Run Inference**"
                )
            with gr.Column(scale=1):
                gr.Markdown(
                    "> ⚠️ **Note:** `magi-attention` cannot be installed in this Hugging Face Space, "
                    "so inputs larger than 1K are resized to 1K in this demo.\n"
                    ">\n"
                    "> For full-resolution inference, please download the weights and run the model locally."
                )

        with gr.Row():
            # ===== COL 1: Settings =====
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Settings")
                input_type = gr.Radio(
                    ["Image", "Video"], label="1. Input Media Type", value="Image",
                    info="Select whether to process a single image or a video clip.",
                )
                task_dropdown = gr.Dropdown(
                    choices=["Detection", "Grounding", "OCR", "GUI", "Pointing"],
                    value="Detection", label="2. Task Type",
                    info="Detection: find all instances | Grounding: match description | "
                         "OCR: extract text | GUI: locate UI element | Pointing: point to target",
                )
                category_input = gr.Textbox(
                    label="3. Categories",
                    value="car, bus, person, potted plant",
                    placeholder="e.g.  car, person, dog  (comma-separated, supports Chinese)",
                    info="Enter one or more categories separated by commas. "
                         "Supports both English and Chinese (e.g. 汽车, 行人).",
                )
                model_dropdown = gr.Dropdown(
                    choices=["fast", "slow", "hybrid"],
                    value="hybrid", label="4. Inference Mode",
                    info="fast: MTP parallel decoding | slow: standard AR decoding | "
                         "hybrid: auto-switch for best quality-speed balance",
                )
                with gr.Accordion("5. Advanced Settings", open=False):
                    gr.Markdown(
                        "*Adjust these only if needed. Default values work well for most cases.*"
                    )
                    temp_slider = gr.Slider(
                        minimum=0.0, maximum=2.0, value=0.7, step=0.1, label="Temperature",
                        info="Higher = more diverse results; lower = more deterministic.",
                    )
                    top_p_slider = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.9, step=0.05, label="Top P",
                        info="Nucleus sampling threshold.",
                    )
                    top_k_slider = gr.Slider(
                        minimum=1, maximum=100, value=20, step=1, label="Top K",
                        info="Top-K sampling: number of highest probability tokens to consider.",
                    )
                    short_size_input = gr.Number(
                        label="Short Side Size (px)", value=None, precision=0,
                        info="Resize the short side of the image to this value before inference. "
                             "Leave empty to keep original size (auto-capped at 1024).",
                    )
                    max_video_frames_slider = gr.Slider(
                        minimum=1, maximum=10, value=4, step=1,
                        label="Max Video Frames",
                        info="Number of frames to sample from the video for inference. "
                             "Each frame takes ~15-20s. Keep ≤ 6 to avoid GPU timeout.",
                    )
                run_btn = gr.Button("🧠 Run Inference", variant="primary", size="lg")

            # ===== COL 2: Main =====
            with gr.Column(scale=3):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 📥 Input Media")
                        image_input = gr.Image(
                            label="Input Image", type="pil", visible=True,
                        )
                        video_input = gr.Video(
                            label="Input Video",
                            visible=False,
                        )
                    with gr.Column(scale=1):
                        gr.Markdown("### 📤 Output Result")
                        output_image = gr.Image(
                            label="Detection Result", type="pil", visible=True,
                        )
                        output_video = gr.Video(
                            label="Video Result", visible=False,
                        )

                gr.Markdown("### 📝 Raw Input Prompt")
                raw_prompt_box = gr.Textbox(
                    value=generate_raw_prompt("Detection", "car, bus, person, potted plant"),
                    interactive=False, lines=2,
                    info="This is the prompt sent to the model (auto-generated from your settings above).",
                )
                gr.Markdown("### 🔍 Decoding Visualization")
                raw_output_box = gr.HTML(label="Decoding Steps")

        # ===== EXAMPLES =====
        gr.Markdown("---")
        gr.Markdown(
            "## 🖼️ Examples\n"
            "Click any example below to auto-fill the settings and input image."
        )
        gallery_images, gallery_captions = prepare_gallery_data()
        example_gallery = gr.Gallery(
            value=list(zip(gallery_images, gallery_captions)),
            show_label=True, columns=4, rows=1, height="auto", allow_preview=False,
        )

        # ===== EVENTS =====
        input_type.change(
            fn=lambda c: (gr.update(visible=(c == "Image")), gr.update(visible=(c == "Video"))),
            inputs=input_type, outputs=[image_input, video_input],
        )

        for comp in [task_dropdown, category_input]:
            comp.change(
                fn=generate_raw_prompt,
                inputs=[task_dropdown, category_input],
                outputs=raw_prompt_box,
            )

        run_btn.click(
            fn=_disable_run_btn,
            inputs=None,
            outputs=[run_btn],
        ).then(
            fn=run_inference,
            inputs=[
                input_type, image_input, video_input,
                task_dropdown, category_input, model_dropdown,
                temp_slider, top_p_slider, top_k_slider,
                short_size_input, raw_prompt_box,
                max_video_frames_slider,
            ],
            outputs=[output_image, output_video, raw_output_box],
        ).then(
            fn=_enable_run_btn,
            inputs=None,
            outputs=[run_btn],
        )

        example_gallery.select(
            fn=update_example_selection,
            outputs=[input_type, image_input, video_input,
                     task_dropdown, category_input, model_dropdown],
        ).then(
            fn=generate_raw_prompt,
            inputs=[task_dropdown, category_input],
            outputs=raw_prompt_box,
        )

    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(debug=True)
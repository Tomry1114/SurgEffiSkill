import argparse
import base64
import io
import json
import os
import random
import time
from multiprocessing import Pool

from openai import OpenAI
from PIL import Image

CLASSIFY_PROMPT = (
    "You are performing a structured classification task.\n"
    "These frames come from a laparoscopic gallbladder surgery.\n\n"
    "Return ONLY a JSON object with:\n"
    "action: [retract, dissect, cauterize, clip, cut, other]\n"
    "phase: [exposure, dissection, clipping, removal, other]\n"
    "instrument: [hook, grasper, dissector, clipper, scissors, other]\n\n"
    "STRICT RULE:\n"
    "Output ONLY the JSON object. No extra text."
)

_CLIENT = None


def get_client():
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = OpenAI(api_key=os.environ["OPENAI_API_KEY"],
                         base_url=os.environ.get("OPENAI_BASE_URL"))
    return _CLIENT


def compress_image(path, max_size=640):
    img = Image.open(path)
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGB")
    img.thumbnail((max_size, max_size))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def safe_parse_llm_json(result_str):
    if not result_str:
        return {}
    cleaned = result_str.strip()
    try:
        parsed = json.loads(cleaned)
    except Exception:
        return {}
    if isinstance(parsed, dict):
        return parsed
    if isinstance(parsed, list) and len(parsed) > 0:
        if isinstance(parsed[0], dict):
            return parsed[0]
    return {}


def classify_4_frames(frame_paths, model="gpt-4.1-mini", retries=3):
    content = []
    for p in frame_paths:
        b64 = compress_image(p)
        content.append({"type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64," + b64}})
    content.append({"type": "text", "text": CLASSIFY_PROMPT})
    for attempt in range(retries):
        try:
            response = get_client().chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content}],
                max_tokens=300,
                temperature=0,
            )
            msg = response.choices[0].message.content
            if msg and msg.strip():
                return msg.strip()
        except Exception as e:
            err_msg = str(e).lower()
            if any(k in err_msg for k in ["insufficient_quota", "billing",
                                          "hard limit", "out of credits", "quota"]):
                print("quota exhausted, terminating")
                os._exit(1)
            print("API error attempt %d: %s" % (attempt + 1, e))
            time.sleep(1 + random.random())
    return None


def process_group(args):
    idx, group, model = args
    print("processing group at index %d" % idx)
    result = classify_4_frames(group, model=model)
    return idx, result


def inject_llm_into_metadata(folder, frames, llm_results):
    meta_path = os.path.join(folder, "metadata.json")
    if not os.path.exists(meta_path):
        print("metadata.json missing, skip injection")
        return
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    meta_map = {os.path.basename(m["file"]): m for m in metadata}
    for start_idx, raw in llm_results:
        parsed = safe_parse_llm_json(raw)
        action = parsed.get("action", "other")
        phase = parsed.get("phase", "other")
        instrument = parsed.get("instrument", "other")
        for offset in range(4):
            frame_idx = start_idx + offset
            if frame_idx >= len(frames):
                break
            fname = os.path.basename(frames[frame_idx])
            if fname not in meta_map:
                continue
            item = meta_map[fname]
            item["action"] = action
            item["phase"] = phase
            item["instrument"] = instrument
            item["llm_raw"] = raw
    out_path = os.path.join(folder, "metadata_with_llm.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(list(meta_map.values()), f, indent=2, ensure_ascii=False)
    print("saved %s" % out_path)


def classify_folder_every4(folder, model="gpt-4.1-mini", workers=8):
    done_json = os.path.join(folder, "metadata_with_llm.json")
    if os.path.exists(done_json):
        print("skip folder %s (already done)" % folder)
        return
    frames = sorted([os.path.join(folder, f) for f in os.listdir(folder)
                    if f.lower().endswith((".jpg", ".png"))])
    groups = [(i, frames[i:i + 4], model) for i in range(0, len(frames), 4)
              if len(frames[i:i + 4]) == 4]
    print("found %d frames, %d groups" % (len(frames), len(groups)))
    with Pool(workers) as p:
        results = p.map(process_group, groups)
    results = [(idx, r) for idx, r in sorted(results)]
    inject_llm_into_metadata(folder, frames, results)


def classify_many_folders(root_dir, model="gpt-4.1-mini", workers=8):
    subfolders = [os.path.join(root_dir, name) for name in os.listdir(root_dir)
                  if os.path.isdir(os.path.join(root_dir, name))]
    print("found %d subfolders under %s" % (len(subfolders), root_dir))
    for folder in subfolders:
        print("processing folder %s" % folder)
        try:
            classify_folder_every4(folder, model=model, workers=workers)
        except Exception as e:
            print("error processing %s: %s" % (folder, e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--model", default="gpt-4.1-mini")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    classify_many_folders(a.root, a.model, a.workers)


if __name__ == "__main__":
    main()

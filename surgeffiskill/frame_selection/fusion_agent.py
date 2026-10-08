import argparse
import json
import os
import time

from openai import OpenAI


class QuotaExceededError(Exception):
    pass


FUSION_SYSTEM_PROMPT = """
You are the FUSION AGENT for laparoscopic surgical video analysis.

INPUT:
- A JSON ARRAY of consecutive frames (typically 10).
- Each frame includes:
  - file
  - vision scores (semantic, motion, merged)
  - LLM labels (phase, action, instrument)
- Frames are in temporal order.

CONTEXT:
- Frames are adjacent in time, so changes are often subtle.
- Similar frames are normal and do NOT require forced score separation.

TASK:
1. Judge the importance of this short segment as a whole:
   - true surgical event,
   - minor transition,
   - or stable/redundant period.
2. Assign an importance score in [0,1] to EACH frame that reflects:
   - the segment's importance in the entire video,
   - and the frame's relative position within the segment.

IMPORTANCE SCALE (GLOBAL):
- 0.85-1.00 : Clear surgical event or key maneuver.
- 0.60-0.84 : Informative but not a discrete event.
- 0.30-0.59 : Minor or transitional segment.
- 0.00-0.29 : Stable or redundant segment.

RULES:
- Frames in the same segment may have very similar scores.
- Only apply small, smooth variations if appropriate.
- Do NOT create large differences when changes are minimal.
- Use a short event name only if the segment is a clear event; otherwise use "none".

OUTPUT (STRICT JSON ONLY):
{
  "frames": [
    {
      "file": "<exact input file>",
      "importance": <float 0-1>,
      "event": "<short name or 'none'>",
      "reason": "<ONE short sentence>"
    }
  ],
  "timeline_explanation": "<=5 sentences summarizing this segment>"
}

CONSTRAINTS:
- One output per input frame.
- Preserve order.
- No extra keys or text.
"""

_CLIENT = None


def get_client():
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = OpenAI(api_key=os.environ["OPENAI_API_KEY"],
                         base_url=os.environ.get("OPENAI_BASE_URL"))
    return _CLIENT


def run_batch(frames_batch, model="gpt-4.1-mini", max_attempts=4):
    for _ in range(max_attempts):
        try:
            response = get_client().chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": FUSION_SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(frames_batch)},
                ],
                max_tokens=2000,
                temperature=0,
            )
            raw = response.choices[0].message.content
            if not raw:
                time.sleep(1)
                continue
            data = json.loads(raw)
            frames = data.get("frames", [])
            timeline_explanation = data.get("timeline_explanation", "")
            if len(frames) != len(frames_batch):
                time.sleep(1)
                continue
            for in_f, out_f in zip(frames_batch, frames):
                out_f["file"] = in_f.get("file", "")
                imp = out_f.get("importance", 0.0)
                out_f["importance"] = float(min(max(imp, 0.0), 1.0))
                out_f.setdefault("event", "none")
                out_f.setdefault("reason", "Scored by visual and semantic signals.")
            return frames, timeline_explanation
        except Exception as e:
            msg = str(e).lower()
            if any(k in msg for k in ["quota", "insufficient", "exceeded",
                                      "billing", "payment", "401", "403", "429"]):
                print("API quota exhausted or access denied")
                raise QuotaExceededError(str(e))
            print("batch error: %s" % e)
            time.sleep(2)
    return [], ""


def run_fusion_agent_score_only(metadata_json_path, save_path, explanation_path,
                                model="gpt-4.1-mini", batch_size=10, top_k=200):
    with open(metadata_json_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    total_frames = len(metadata)
    total_batches = (total_frames + batch_size - 1) // batch_size
    all_scored_frames = []
    all_explanations = []
    print("total frames %d, batch_size %d, total batches %d"
          % (total_frames, batch_size, total_batches))
    for batch_idx, start in enumerate(range(0, total_frames, batch_size), start=1):
        end = min(start + batch_size, total_frames)
        batch = metadata[start:end]
        print("batch %d/%d frames %d-%d" % (batch_idx, total_batches, start, end - 1))
        frames, explain = run_batch(batch, model=model)
        if not frames:
            print("empty batch result")
            continue
        avg_imp = sum(f["importance"] for f in frames) / len(frames)
        print("batch %d done, avg importance %.3f" % (batch_idx, avg_imp))
        all_scored_frames.extend(frames)
        if explain:
            all_explanations.append(explain)
        if batch_idx % 10 == 0 or end == total_frames:
            print("progress %d/%d frames" % (end, total_frames))
    if not all_scored_frames:
        print("no scored frames")
        return
    all_scored_frames.sort(key=lambda x: -x["importance"])
    final_frames = all_scored_frames[:top_k]
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(final_frames, f, indent=2, ensure_ascii=False)
    with open(explanation_path, "w", encoding="utf-8") as f:
        f.write(" ".join(all_explanations))
    print("saved %d frames to %s" % (len(final_frames), save_path))


def get_topk_from_B(folder_name, B_root):
    path = os.path.join(B_root, folder_name)
    if not os.path.exists(path):
        return None
    return len([f for f in os.listdir(path)
                if f.lower().endswith((".jpg", ".png", ".jpeg"))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a-root", required=True)
    ap.add_argument("--b-root", required=True)
    ap.add_argument("--model", default="gpt-4.1-mini")
    ap.add_argument("--batch-size", type=int, default=10)
    a = ap.parse_args()
    for folder_name in sorted(os.listdir(a.a_root)):
        folder_A = os.path.join(a.a_root, folder_name)
        if not os.path.isdir(folder_A):
            continue
        metadata_path = os.path.join(folder_A, "metadata_with_llm.json")
        if not os.path.exists(metadata_path):
            print("skip %s: metadata not found" % folder_name)
            continue
        save_path = os.path.join(folder_A, "fusion_selected.json")
        explanation_path = os.path.join(folder_A, "fusion_summary.txt")
        if os.path.exists(save_path):
            print("skip %s: already done" % folder_name)
            continue
        top_k = get_topk_from_B(folder_name, a.b_root)
        if top_k is None or top_k == 0:
            print("skip %s: no matching folder in B" % folder_name)
            continue
        print("processing %s, target frames %d" % (folder_name, top_k))
        try:
            run_fusion_agent_score_only(metadata_path, save_path, explanation_path,
                                        a.model, a.batch_size, top_k)
        except QuotaExceededError:
            print("terminated due to API quota exhaustion")
            break


if __name__ == "__main__":
    main()

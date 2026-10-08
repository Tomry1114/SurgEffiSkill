import argparse
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import cv2
import numpy as np
import scipy.signal
import torch
from glob import glob
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoImageProcessor


def plot_score_segments(score_log, segments, aks_keyframes, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs = [i for i, _ in score_log]
    ys = [s for _, s in score_log]
    plt.figure(figsize=(18, 4))
    plt.plot(xs, ys, linewidth=1.2, label="Merged Score", color="black")
    cmap = plt.colormaps.get_cmap("tab20")
    for i, seg in enumerate(segments):
        seg_x = [x for x, _ in seg]
        plt.axvspan(seg_x[0], seg_x[-1], color=cmap(i % 20), alpha=0.15)
    peak_x = [i for i, _ in aks_keyframes]
    peak_y = [s for _, s in aks_keyframes]
    plt.scatter(peak_x, peak_y, color="red", s=40)
    plt.xlabel("Frame Index")
    plt.ylabel("Merged Score")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def compute_semantic_motion_score(model, processor, frame_prev, frame_curr, device, alpha=0.5, beta=0.5):
    imgs = [
        Image.fromarray(cv2.cvtColor(frame_prev, cv2.COLOR_BGR2RGB)),
        Image.fromarray(cv2.cvtColor(frame_curr, cv2.COLOR_BGR2RGB)),
    ]
    inputs = processor(images=imgs, return_tensors="pt").to(device)
    with torch.no_grad():
        feats = model(**inputs).last_hidden_state.mean(dim=1)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        sim = torch.matmul(feats[0], feats[1]).item()
        semantic = float(1 - sim)
    g1 = cv2.resize(cv2.cvtColor(frame_prev, cv2.COLOR_BGR2GRAY), None, fx=0.25, fy=0.25)
    g2 = cv2.resize(cv2.cvtColor(frame_curr, cv2.COLOR_BGR2GRAY), None, fx=0.25, fy=0.25)
    flow = cv2.calcOpticalFlowFarneback(g1, g2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    motion = np.percentile(mag, 90)
    motion = motion / (motion + 1)
    motion = float(motion)
    merged = float(alpha * semantic + beta * motion)
    return semantic, motion, merged


def smooth_scores(scores, window=7):
    if len(scores) < window:
        return scores
    return np.convolve(scores, np.ones(window) / window, mode="same")


def score_derivative(scores):
    d = np.abs(np.diff(scores, prepend=scores[0]))
    return d / (d.max() + 1e-6)


def aks_judge_and_split_enhanced(score_list, base_threshold=0.02, min_len=10):
    if len(score_list) < min_len * 2:
        return [score_list]
    scores = np.array([s for _, s in score_list])
    smoothed = smooth_scores(scores)
    diff_signal = score_derivative(smoothed)
    combined = 0.6 * smoothed + 0.4 * diff_signal
    diffs = np.abs(np.diff(combined))
    split_points = np.where(diffs > base_threshold)[0] + 1
    if len(split_points) == 0:
        return [score_list]
    segments, prev = [], 0
    for sp in split_points:
        if sp - prev >= min_len:
            segments.append(score_list[prev:sp])
            prev = sp
    if len(score_list) - prev >= min_len:
        segments.append(score_list[prev:])
    return segments


def aks_recursive_enhanced(score_list, depth=0, max_depth=4):
    if depth >= max_depth or len(score_list) < 25:
        return [score_list]
    segments = aks_judge_and_split_enhanced(score_list)
    if len(segments) <= 1:
        return [score_list]
    result = []
    for seg in segments:
        result.extend(aks_recursive_enhanced(seg, depth + 1, max_depth))
    return result


def aks_top_k(seg, k=3):
    sorted_seg = sorted(seg, key=lambda x: x[1], reverse=True)
    return sorted_seg[:min(k, len(sorted_seg))]


def aks_select_keyframes_enhanced(segments, k=3):
    final = []
    for seg in segments:
        top1 = aks_top_k(seg, k)
        uniq = {idx: sc for idx, sc in top1}
        final.extend([(i, s) for i, s in uniq.items()])
    return sorted(final, key=lambda x: x[0])


def semantic_motion_sample_video(input_path, model, processor, device, base_tmp_dir,
                                 threshold=0.40, every_n=10, k=3, plot=True):
    video_name = os.path.splitext(os.path.basename(input_path))[0]
    out_dir = os.path.join(base_tmp_dir, video_name)
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(input_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    score_log = []
    metadata = []
    idx = saved = 0
    last_frame = None
    for _ in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1
        if idx % every_n != 0:
            continue
        if last_frame is None:
            semantic = motion = merged = 1.0
            save = True
        else:
            semantic, motion, merged = compute_semantic_motion_score(
                model, processor, last_frame, frame, device)
            save = merged > threshold
        score_log.append((idx, merged))
        if save:
            p = os.path.join(out_dir, "frame_%06d.jpg" % saved)
            cv2.imwrite(p, frame)
            metadata.append({"file": p, "orig_idx": idx, "semantic": semantic,
                             "motion": motion, "merged": merged})
            saved += 1
        last_frame = frame.copy()
    cap.release()
    segments = aks_recursive_enhanced(score_log)
    aks_kf = aks_select_keyframes_enhanced(segments, k=k)
    aks_idx = set(i for i, _ in aks_kf)
    keep_files = set()
    new_metadata = []
    for m in metadata:
        if m["orig_idx"] in aks_idx:
            new_metadata.append(m)
            keep_files.add(m["file"])
    all_images = glob(os.path.join(out_dir, "frame_*.jpg"))
    delete_count = 0
    for img in all_images:
        if img not in keep_files:
            try:
                os.remove(img)
                delete_count += 1
            except OSError:
                pass
    print("cleaned %s: saved %d, kept %d, deleted %d"
          % (video_name, len(all_images), len(keep_files), delete_count))
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(new_metadata, f, indent=2, ensure_ascii=False)
    if plot:
        plot_score_segments(score_log, segments, aks_kf,
                            os.path.join(out_dir, "score_segments.png"))
    return out_dir


def process_all_videos(input_dir, base_tmp_dir, model_dir, threshold=0.50,
                       every_n=10, k=3, plot=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("running DINO on %s" % device)
    model = AutoModel.from_pretrained(model_dir).to(device)
    processor = AutoImageProcessor.from_pretrained(model_dir)
    videos = sorted(glob(os.path.join(input_dir, "*.mp4")))
    print("found %d videos" % len(videos))
    for i, vp in enumerate(videos, 1):
        print("[%d/%d] %s" % (i, len(videos), vp))
        video_name = os.path.splitext(os.path.basename(vp))[0]
        out_dir = os.path.join(base_tmp_dir, video_name)
        if os.path.exists(out_dir):
            print("skip %s (folder exists)" % video_name)
            continue
        semantic_motion_sample_video(vp, model, processor, device, base_tmp_dir,
                                     threshold, every_n, k, plot)
    print("done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model-dir", default="facebook/dinov2-small")
    ap.add_argument("--threshold", type=float, default=0.50)
    ap.add_argument("--every-n", type=int, default=10)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--no-plot", dest="plot", action="store_false", default=True)
    a = ap.parse_args()
    process_all_videos(a.input_dir, a.out_dir, a.model_dir, a.threshold,
                       a.every_n, a.topk, a.plot)


if __name__ == "__main__":
    main()

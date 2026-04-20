from ..preprocessing import get_scene_images, get_scene_duration_ms, classify_scene_type
from ..scene_metrics import derive_valence
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

AOI_LEFT = {"x_min": 0.15, "x_max": 0.45, "y_min": 0.45, "y_max": 0.82}
AOI_RIGHT = {"x_min": 0.55, "x_max": 0.85, "y_min": 0.45, "y_max": 0.82}

VALENCE_COLORS = {
    "negative": "#d32f2f",
    "positive": "#388e3c",
    "neutral": "#757575",
}

_image_cache = {}

def _download_image(url):
    """
    Download image from URL, cache result
    """
    if url in _image_cache:
        return _image_cache[url]
    try:
        import urllib.request
        from io import BytesIO
        from PIL import Image
 
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        resp = urllib.request.urlopen(req, timeout=10)
        img = np.array(Image.open(BytesIO(resp.read())).convert("RGB"))
        _image_cache[url] = img
        return img
    except Exception:
        return None

def _get_valence_and_label(image_id, stimulus_config):
    """
    Get valence and label for an image
    """
    if image_id not in stimulus_config:
        return "unknown", image_id
    meta = stimulus_config[image_id]
    valence = derive_valence(meta.get("category", ""), meta.get("labels", []))
    label = ", ".join(meta.get("labels", []))
    return valence, label

def _find_image_sides(scene_df, images, stimulus_config):
    """
    Figure out which image is on the left and which is on the right
    by looking at the mean horizontal gaze position for each image.
    """
    left, right = None, None
 
    for img_id in images:
        rows = scene_df[scene_df["IMAGE"] == img_id]
        if len(rows) == 0:
            continue
 
        valence, label = _get_valence_and_label(img_id, stimulus_config)
        info = {"id": img_id, "valence": valence, "label": label}
 
        if rows["RX"].mean() < 0.5:
            left = info
        else:
            right = info
 
    for img_id in images:
        already = {(left or {}).get("id"), (right or {}).get("id")}
        if img_id in already:
            continue
        valence, label = _get_valence_and_label(img_id, stimulus_config)
        info = {"id": img_id, "valence": valence, "label": label}
        if left is None:
            left = info
        elif right is None:
            right = info
 
    return {"left": left, "right": right}


def _draw_aoi_box(ax, aoi, info, stimulus_config, show_image=True, filled=True):
    """
    Draw one AOI rectangle with its image and label
    """
    color = VALENCE_COLORS.get(info["valence"], "#bdbdbd")
 
    if show_image:
        url = stimulus_config.get(info["id"], {}).get("url")
        if url:
            img = _download_image(url)
            if img is not None:
                ax.imshow(img,
                    extent=[aoi["x_min"], aoi["x_max"], aoi["y_max"], aoi["y_min"]],
                    aspect="auto", alpha=0.4, zorder=0)
 
    if filled:
        rect = Rectangle(
            (aoi["x_min"], aoi["y_min"]),
            aoi["x_max"] - aoi["x_min"], aoi["y_max"] - aoi["y_min"],
            linewidth=2, edgecolor=color, facecolor=color, alpha=0.12)
    else:
        rect = Rectangle(
            (aoi["x_min"], aoi["y_min"]),
            aoi["x_max"] - aoi["x_min"], aoi["y_max"] - aoi["y_min"],
            linewidth=2.5, edgecolor=color, facecolor="none", linestyle="--")
    ax.add_patch(rect)
 
    cx = (aoi["x_min"] + aoi["x_max"]) / 2
    label = info["valence"].upper()
    if filled:
        label += f"\n{info['label'][:30]}"

    ax.text(cx, aoi["y_min"] - 0.03, label, ha="center", va="top", fontsize=8, fontweight="bold", color=color)


def plot_scene_exploration(scene_df, scene_index, fixations, stimulus_config):
    """
    Two-panel plot for one scene:
    - Left panel: scanpath with numbered fixations and arrows
    - Right panel: gaze density heatmap
    Both panels show AOI boxes and stimulus images
    """
    images = get_scene_images(scene_df)
    if not images:
        return
 
    sides = _find_image_sides(scene_df, images, stimulus_config)
    duration = get_scene_duration_ms(scene_df)
 
    fig, (ax_scanpath, ax_heatmap) = plt.subplots(1, 2, figsize=(18, 7))
  
    # scanpath panel

    for side, aoi in [("left", AOI_LEFT), ("right", AOI_RIGHT)]:
        if sides[side]:
            _draw_aoi_box(ax_scanpath, aoi, sides[side], stimulus_config, filled=True)
 
    if len(fixations) > 0:
        df_fixations = fixations.dropna(subset=["rx", "ry"]).copy()
        df_fixations["rx"] = df_fixations["rx"].clip(0, 1)
        df_fixations["ry"] = df_fixations["ry"].clip(0, 1)
 
        img_colors = {}
        for info in sides.values():
            if info:
                img_colors[info["id"]] = VALENCE_COLORS.get(info["valence"], "gray")
 
        sizes = df_fixations["duration_ms"].clip(lower=60) / 3
        colors = [img_colors.get(row["image"], "#90a4ae") for _, row in df_fixations.iterrows()]
 
        ax_scanpath.scatter(df_fixations["rx"], df_fixations["ry"], s=sizes, c=colors,
                        alpha=0.6, edgecolors="black", linewidth=0.5, zorder=3)
 
    ax_scanpath.set_xlabel("RX")
    ax_scanpath.set_ylabel("RY")
    ax_scanpath.set_title(f"Gaze scanpath | {duration:.0f}ms | {len(fixations)} fixations")
 
    # heatmap panel
 
    for side, aoi in [("left", AOI_LEFT), ("right", AOI_RIGHT)]:
        if sides[side]:
            _draw_aoi_box(ax_heatmap, aoi, sides[side], stimulus_config, show_image=True, filled=False)
 
    rx = scene_df["RX"].dropna().values
    ry = scene_df["RY"].dropna().values
    mask = (rx > -0.1) & (rx < 1.1) & (ry > -0.1) & (ry < 1.1)
    rx, ry = rx[mask], ry[mask]
 
    if len(rx) > 10:
        try:
            from scipy.stats import gaussian_kde
            xi, yi = np.linspace(0, 1, 200), np.linspace(0, 1, 200)
            xi_grid, yi_grid = np.meshgrid(xi, yi)
            kde = gaussian_kde(np.vstack([rx, ry]), bw_method=0.05)
            zi = kde(np.vstack([xi_grid.ravel(), yi_grid.ravel()])).reshape(xi_grid.shape)
            ax_heatmap.contourf(xi_grid, yi_grid, zi, levels=20, cmap="YlOrRd", alpha=0.7)
        except Exception:
            ax_heatmap.scatter(rx, ry, s=3, alpha=0.3, color="red")
    else:
        ax_heatmap.scatter(rx, ry, s=10, alpha=0.5, color="red")
 
    ax_heatmap.set_xlabel("RX")
    ax_heatmap.set_ylabel("RY")
    ax_heatmap.set_title("Gaze heatmap")
 
    for ax in [ax_scanpath, ax_heatmap]:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.invert_yaxis()
        ax.set_aspect("equal", adjustable="box")
 
    fig.suptitle(f"Scene {scene_index}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()

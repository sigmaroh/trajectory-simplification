"""
Export OSM comparison map data to JSON and generate an HTML viewer that reads it.
"""

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

# Add project root to path for module imports.
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.experiments.visualize_osm import (  # noqa: E402
    ALGORITHM_COLORS,
    downsample_path,
    extract_latlon,
    sample_indices,
    simplify_trajectory,
)
from src.metrics.evaluation_metrics import compute_all_metrics  # noqa: E402


def prepare_map_data(
    trajectories_file: str,
    algorithms: Sequence[str],
    compression_ratios: Sequence[float],
    max_trajectories: int,
    max_points_per_trajectory: int,
) -> Dict:
    with open(trajectories_file, "rb") as f:
        trajectories = pickle.load(f)

    if not isinstance(trajectories, list) or len(trajectories) == 0:
        raise ValueError("Trajectory file is empty or invalid.")

    selected_ids = sample_indices(len(trajectories), max_trajectories)
    selected_pairs = [(idx, trajectories[idx]) for idx in selected_ids]

    valid_pairs = []
    centers = []
    for traj_id, traj in selected_pairs:
        if not isinstance(traj, pd.DataFrame) or len(traj) < 2:
            continue
        try:
            lat_values, lon_values = extract_latlon(traj)
        except ValueError:
            continue
        finite_mask = np.isfinite(lat_values) & np.isfinite(lon_values)
        lat_values = lat_values[finite_mask]
        lon_values = lon_values[finite_mask]
        if len(lat_values) < 2:
            continue
        centers.append((float(np.mean(lat_values)), float(np.mean(lon_values))))
        valid_pairs.append((traj_id, traj.copy()))

    if not valid_pairs:
        raise ValueError("No valid trajectories found.")

    ratio_values = sorted(float(r) for r in compression_ratios)
    ratio_labels = [f"{r:.2f}x" for r in ratio_values]
    traj_labels = [f"T{traj_id:04d}" for traj_id, _ in valid_pairs]

    layers: List[Dict] = []
    metric_rows: List[Dict] = []

    for traj_id, traj in valid_pairs:
        traj["lat"] = pd.to_numeric(traj["lat"], errors="coerce")
        traj["lon"] = pd.to_numeric(traj["lon"], errors="coerce")
        traj = traj.dropna(subset=["lat", "lon"]).reset_index(drop=True)
        if len(traj) < 2:
            continue

        traj_label = f"T{traj_id:04d}"
        for ratio in ratio_values:
            ratio_label = f"{ratio:.2f}x"
            for algorithm in algorithms:
                algo_key = algorithm.lower().strip()
                try:
                    simplified = simplify_trajectory(traj, algorithm, compression_ratio=ratio)
                    if simplified.shape[0] < 2:
                        continue
                    path = downsample_path(
                        simplified[:, 0],
                        simplified[:, 1],
                        max_points=max_points_per_trajectory,
                    )
                    color = ALGORITHM_COLORS.get(algo_key, "black")
                    layer_label = f"{ratio_label} | {traj_label} - {algorithm}"
                    layers.append(
                        {
                            "label": layer_label,
                            "ratio_label": ratio_label,
                            "trajectory_id": traj_label,
                            "algorithm": algorithm,
                            "color": color,
                            "path": path,
                        }
                    )

                    if algo_key not in {"original", "none"}:
                        metrics = compute_all_metrics(traj, simplified, original_indices=None)
                        metric_rows.append(
                            {
                                "layer_label": layer_label,
                                "ratio_label": ratio_label,
                                "trajectory_id": traj_label,
                                "algorithm": algorithm,
                                "hausdorff_distance": metrics.get("hausdorff_distance"),
                                "frechet_distance": metrics.get("frechet_distance"),
                                "average_pte": metrics.get("average_pte"),
                                "ped": metrics.get("ped"),
                                "sed": metrics.get("sed"),
                                "dad": metrics.get("dad"),
                                "sad": metrics.get("sad"),
                                "issd": metrics.get("issd"),
                            }
                        )
                except Exception:
                    continue

    center = [float(np.mean([c[0] for c in centers])), float(np.mean([c[1] for c in centers]))]
    return {
        "meta": {
            "center": center,
            "zoom_start": 11,
            "ratio_labels": ratio_labels,
            "trajectory_labels": traj_labels,
            "algorithms": list(algorithms),
        },
        "layers": layers,
        "metrics": metric_rows,
    }


def render_html_template(json_filename: str, fallback_data: Dict) -> str:
    fallback_json = json.dumps(fallback_data, ensure_ascii=True)
    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Trajectory OSM Comparison (JSON)</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <style>
    html, body, #map {{ height: 100%; margin: 0; padding: 0; }}
    .panel {{
      position: fixed; z-index: 9999; background: rgba(255,255,255,0.92);
      border: 1px solid #888; border-radius: 6px; box-shadow: 0 1px 6px rgba(0,0,0,0.25);
      padding: 8px; font-size: 12px;
    }}
    #trajectory-panel {{ top: 18px; left: 18px; }}
    #ratio-panel {{ top: 72px; left: 18px; }}
    #layers-panel {{ top: 126px; left: 18px; max-height: 34vh; overflow-y: auto; min-width: 300px; font-size: 11px; }}
    #metrics-panel {{ bottom: 18px; left: 18px; max-height: 38vh; overflow-y: auto; max-width: 96vw; font-size: 11px; }}
    #legend-panel {{ top: 170px; right: 18px; min-width: 120px; font-size: 11px; }}
  </style>
</head>
<body>
  <div id="map"></div>



  <div id="ratio-panel" class="panel">
    <label for="ratio-select" style="font-weight:700; margin-right:6px;">Compression</label>
    <select id="ratio-select"></select>
  </div>

  <div id="trajectory-panel" class="panel">
    <label for="trajectory-select" style="font-weight:700; margin-right:6px;">Trajectory</label>
    <select id="trajectory-select"></select>
  </div>

  <div id="layers-panel" class="panel">
    <div style="font-weight:700; margin-bottom:4px;">Visible Layers</div>
    <div id="layer-list-card"></div>
  </div>

  <div id="metrics-panel" class="panel">
    <div style="font-weight:700; margin-bottom:6px;">Error Metrics</div>
    <table id="metrics-table" style="border-collapse: collapse;">
      <thead>
        <tr>
          <th style="padding:4px 6px; text-align:left;">Ratio</th>
          <th style="padding:4px 6px; text-align:left;">Traj</th>
          <th style="padding:4px 6px; text-align:left;">Algo</th>
          <th style="padding:4px 6px; text-align:right;">HD (m)</th>
          <th style="padding:4px 6px; text-align:right;">Fréchet (m)</th>
          <th style="padding:4px 6px; text-align:right;">APTE (m)</th>
          <th style="padding:4px 6px; text-align:right;">PED (m)</th>
          <th style="padding:4px 6px; text-align:right;">SED (m)</th>
          <th style="padding:4px 6px; text-align:right;">DAD (deg)</th>
          <th style="padding:4px 6px; text-align:right;">SAD (m/s)</th>
          <th style="padding:4px 6px; text-align:right;">ISSD (m*s)</th>
        </tr>
      </thead>
      <tbody id="metrics-body"></tbody>
    </table>
  </div>

  <div id="legend-panel" class="panel">
    <div style="font-weight:700; margin-bottom:4px;">Algorithm Legend</div>
    <div id="legend-body"></div>
  </div>

  <script>
    const DATA_URL = "{json_filename}";
    const FALLBACK_DATA = {fallback_json};
    const algorithmDisplay = {{
      original: "Original", dp: "DP", squish: "SQUISH", vw: "VW", sw: "SW", rw: "RW",
      greedy_policy: "Greedy Policy (RL)", proposed: "Proposed"
    }};

    function fmt(v) {{
      if (v === null || v === undefined || !Number.isFinite(v)) return "-";
      if (Math.abs(v) >= 1000) return v.toFixed(1);
      if (Math.abs(v) >= 100) return v.toFixed(2);
      return v.toFixed(3);
    }}

    async function loadData() {{
      try {{
        const r = await fetch(DATA_URL);
        if (!r.ok) throw new Error("fetch failed");
        return await r.json();
      }} catch (e) {{
        console.warn("Using embedded fallback data. To force JSON load, open via local web server.", e);
        return FALLBACK_DATA;
      }}
    }}

    function init(data) {{
      const map = L.map("map", {{ zoomControl: false }}).setView(data.meta.center, data.meta.zoom_start || 11);
      L.control.zoom({{ position: "bottomright" }}).addTo(map);

      const baseMaps = {{
        "OpenStreetMap": L.tileLayer("https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png", {{ attribution: "&copy; OpenStreetMap contributors" }}),
        "Carto Light": L.tileLayer("https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png", {{ attribution: "&copy; OpenStreetMap contributors &copy; CARTO" }}),
        "Stamen Toner": L.tileLayer("https://stamen-tiles.a.ssl.fastly.net/toner/{{z}}/{{x}}/{{y}}.png", {{ attribution: "Map tiles by Stamen Design; data by OpenStreetMap" }}),
        "Esri WorldGrayCanvas": L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{{z}}/{{y}}/{{x}}", {{ attribution: "Tiles &copy; Esri" }})
      }};
      baseMaps["OpenStreetMap"].addTo(map);
      L.control.layers(baseMaps, null, {{ position: "topright", collapsed: false }}).addTo(map);

      const ratioSelect = document.getElementById("ratio-select");
      const trajSelect = document.getElementById("trajectory-select");
      const layerCard = document.getElementById("layer-list-card");
      const metricsBody = document.getElementById("metrics-body");
      const legendBody = document.getElementById("legend-body");

      data.meta.ratio_labels.forEach(v => {{
        const o = document.createElement("option");
        o.value = v; o.textContent = v; ratioSelect.appendChild(o);
      }});
      data.meta.trajectory_labels.forEach(v => {{
        const o = document.createElement("option");
        o.value = v; o.textContent = v; trajSelect.appendChild(o);
      }});
      ratioSelect.value = data.meta.ratio_labels[0];
      trajSelect.value = data.meta.trajectory_labels[0];

      const layerRegistry = data.layers.map((layer, idx) => {{
        const group = L.layerGroup();
        const path = layer.path || [];
        if (path.length >= 2) {{
          L.polyline(path, {{
            color: layer.color || "black",
            weight: (layer.algorithm || "").toLowerCase() === "original" ? 3 : 2,
            opacity: (layer.algorithm || "").toLowerCase() === "original" ? 0.9 : 0.75
          }}).bindTooltip(layer.label || "").addTo(group);
          L.circleMarker(path[0], {{ radius: 3, color: "green", fill: true, fillOpacity: 0.8 }}).addTo(group);
          L.circleMarker(path[path.length - 1], {{ radius: 3, color: "red", fill: true, fillOpacity: 0.8 }}).addTo(group);
        }}
        return {{ key: "layer-" + idx, ...layer, group, checked: true }};
      }});

      const algSeen = new Set();
      layerRegistry.forEach(item => {{
        const k = (item.algorithm || "").toLowerCase();
        if (algSeen.has(k)) return;
        algSeen.add(k);
        const row = document.createElement("div");
        row.style.display = "flex";
        row.style.alignItems = "center";
        row.style.gap = "6px";
        row.style.margin = "2px 0";
        row.innerHTML = `<span style="display:inline-block;width:12px;height:12px;background:${{item.color || "black"}};border:1px solid #444;"></span><span>${{algorithmDisplay[k] || item.algorithm}}</span>`;
        legendBody.appendChild(row);
      }});

      function selectedLayers() {{
        const ratio = ratioSelect.value;
        const traj = trajSelect.value;
        return layerRegistry.filter(l => l.ratio_label === ratio && l.trajectory_id === traj);
      }}

      function refreshLayerCard() {{
        const layers = selectedLayers();
        if (!layers.length) {{
          layerCard.innerHTML = "<div>No matching layers.</div>";
          return;
        }}
        layerCard.innerHTML = "";
        layers.forEach(layer => {{
          const row = document.createElement("label");
          row.style.display = "flex";
          row.style.alignItems = "center";
          row.style.gap = "6px";
          row.style.margin = "2px 0";
          const cb = document.createElement("input");
          cb.type = "checkbox";
          cb.checked = layer.checked;
          cb.addEventListener("change", () => {{
            layer.checked = cb.checked;
            applyVisibility();
          }});
          const swatch = document.createElement("span");
          swatch.style.display = "inline-block";
          swatch.style.width = "10px";
          swatch.style.height = "10px";
          swatch.style.border = "1px solid #444";
          swatch.style.background = layer.color || "black";
          const text = document.createElement("span");
          text.textContent = layer.label;
          row.appendChild(cb);
          row.appendChild(swatch);
          row.appendChild(text);
          layerCard.appendChild(row);
        }});
      }}

      function refreshMetrics() {{
        const ratio = ratioSelect.value;
        const traj = trajSelect.value;
        const visibleLayerLabels = new Set(selectedLayers().filter(l => l.checked).map(l => l.label));
        const rows = data.metrics.filter(m => m.ratio_label === ratio && m.trajectory_id === traj && visibleLayerLabels.has(m.layer_label));
        metricsBody.innerHTML = "";
        rows.forEach(m => {{
          const tr = document.createElement("tr");
          tr.innerHTML =
            `<td style='padding:3px 6px;'>${{m.ratio_label}}</td>` +
            `<td style='padding:3px 6px;'>${{m.trajectory_id}}</td>` +
            `<td style='padding:3px 6px;'>${{m.algorithm}}</td>` +
            `<td style='padding:3px 6px; text-align:right;'>${{fmt(m.hausdorff_distance)}}</td>` +
            `<td style='padding:3px 6px; text-align:right;'>${{fmt(m.frechet_distance)}}</td>` +
            `<td style='padding:3px 6px; text-align:right;'>${{fmt(m.average_pte)}}</td>` +
            `<td style='padding:3px 6px; text-align:right;'>${{fmt(m.ped)}}</td>` +
            `<td style='padding:3px 6px; text-align:right;'>${{fmt(m.sed)}}</td>` +
            `<td style='padding:3px 6px; text-align:right;'>${{fmt(m.dad)}}</td>` +
            `<td style='padding:3px 6px; text-align:right;'>${{fmt(m.sad)}}</td>` +
            `<td style='padding:3px 6px; text-align:right;'>${{fmt(m.issd)}}</td>`;
          metricsBody.appendChild(tr);
        }});
      }}

      function applyVisibility() {{
        layerRegistry.forEach(l => {{
          if (map.hasLayer(l.group)) map.removeLayer(l.group);
        }});
        selectedLayers().forEach(l => {{
          if (l.checked) l.group.addTo(map);
        }});
        refreshMetrics();
      }}

      ratioSelect.addEventListener("change", () => {{
        selectedLayers().forEach(l => (l.checked = true));
        refreshLayerCard();
        applyVisibility();
      }});
      trajSelect.addEventListener("change", () => {{
        selectedLayers().forEach(l => (l.checked = true));
        refreshLayerCard();
        applyVisibility();
      }});

      refreshLayerCard();
      applyVisibility();
    }}

    loadData().then(init);
  </script>
</body>
</html>
"""


def export_json_and_html(
    trajectories_file: str,
    output_json: str,
    output_html: str,
    algorithms: Sequence[str],
    compression_ratios: Sequence[float],
    max_trajectories: int,
    max_points_per_trajectory: int,
) -> None:
    data = prepare_map_data(
        trajectories_file=trajectories_file,
        algorithms=algorithms,
        compression_ratios=compression_ratios,
        max_trajectories=max_trajectories,
        max_points_per_trajectory=max_points_per_trajectory,
    )

    output_json_path = Path(output_json)
    output_html_path = Path(output_html)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_html_path.parent.mkdir(parents=True, exist_ok=True)

    output_json_path.write_text(json.dumps(data, ensure_ascii=True), encoding="utf-8")
    html_content = render_html_template(output_json_path.name, data)
    output_html_path.write_text(html_content, encoding="utf-8")

    print(f"Saved JSON data to: {output_json_path}")
    print(f"Saved HTML viewer to: {output_html_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export OSM comparison JSON + HTML viewer")
    parser.add_argument("--trajectories-file", type=str, default="data/processed/trajectories.pkl")
    parser.add_argument("--output-json", type=str, default="results/figures/trajectories_osm_comparison_data.json")
    parser.add_argument("--output-html", type=str, default="results/figures/trajectories_osm_comparison_from_json.html")
    parser.add_argument("--algorithms", type=str, default="original,dp,vw,squish,rw,greedy_policy,proposed")
    parser.add_argument("--compression-ratios", type=str, default="2,5")
    parser.add_argument("--max-trajectories", type=int, default=1)
    parser.add_argument("--max-points-per-trajectory", type=int, default=1200)
    args = parser.parse_args()

    algorithms = [a.strip() for a in args.algorithms.split(",") if a.strip()]
    compression_ratios = [float(x.strip()) for x in args.compression_ratios.split(",") if x.strip()]

    export_json_and_html(
        trajectories_file=args.trajectories_file,
        output_json=args.output_json,
        output_html=args.output_html,
        algorithms=algorithms,
        compression_ratios=compression_ratios,
        max_trajectories=args.max_trajectories,
        max_points_per_trajectory=args.max_points_per_trajectory,
    )


if __name__ == "__main__":
    main()

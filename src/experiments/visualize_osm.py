"""
Export original and simplified trajectories to interactive OpenStreetMap maps.
"""

import argparse
import pickle
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import folium
import numpy as np
import pandas as pd
from branca.element import Element

# Add project root to path for module imports.
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.algorithms.baseline_algorithms import simplify_with_budget
from src.algorithms.proposed_method import proposed_simplification
from src.metrics.evaluation_metrics import compute_all_metrics

ALGORITHM_COLORS = {
    "none": "gray",
    "original": "gray",
    "dp": "blue",
    "rdp": "blue",
    "douglas-peucker": "blue",
    "douglas_peucker": "blue",
    "squish": "purple",
    "vw": "orange",
    "visvalingam-whyatt": "orange",
    "visvalingam_whyatt": "orange",
    "sw": "green",
    "sliding_window": "green",
    "sliding-window": "green",
    "rw": "red",
    "reumann-witkam": "red",
    "reumann_witkam": "red",
    "greedy_policy": "darkpurple",
    "greedy-policy": "darkpurple",
    "rl_inspired": "darkpurple",
    "proposed": "black",
}


def is_valid_latlon(lat_values: np.ndarray, lon_values: np.ndarray) -> bool:
    """Check whether arrays look like valid latitude/longitude coordinates."""
    if len(lat_values) < 2 or len(lon_values) < 2:
        return False
    finite_mask = np.isfinite(lat_values) & np.isfinite(lon_values)
    if finite_mask.sum() < 2:
        return False
    lat = lat_values[finite_mask]
    lon = lon_values[finite_mask]
    if np.any((lat < -90) | (lat > 90)):
        return False
    if np.any((lon < -180) | (lon > 180)):
        return False
    if np.nanstd(lat) < 1e-8 or np.nanstd(lon) < 1e-8:
        return False
    return True


def extract_latlon(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Extract latitude and longitude arrays from a trajectory DataFrame."""
    column_pairs: Sequence[Tuple[str, str]] = (
        ("lat", "lon"),
        ("latitude", "longitude"),
        ("Latitude", "Longitude"),
    )

    for lat_col, lon_col in column_pairs:
        if lat_col not in df.columns or lon_col not in df.columns:
            continue
        lat_values = pd.to_numeric(df[lat_col], errors="coerce").to_numpy(dtype=float)
        lon_values = pd.to_numeric(df[lon_col], errors="coerce").to_numpy(dtype=float)
        if is_valid_latlon(lat_values, lon_values):
            return lat_values, lon_values
        if is_valid_latlon(lon_values, lat_values):
            return lon_values, lat_values

    raise ValueError("No valid latitude/longitude columns found.")


def sample_indices(total_count: int, max_items: int) -> List[int]:
    """Get evenly spaced indices to keep map rendering lightweight."""
    if total_count <= max_items:
        return list(range(total_count))
    return np.linspace(0, total_count - 1, max_items, dtype=int).tolist()


def downsample_path(lat_values: np.ndarray, lon_values: np.ndarray, max_points: int) -> List[Tuple[float, float]]:
    """Create a lat/lon path with at most max_points."""
    if len(lat_values) > max_points:
        keep = np.linspace(0, len(lat_values) - 1, max_points, dtype=int)
        lat_values = lat_values[keep]
        lon_values = lon_values[keep]
    return list(zip(lat_values.tolist(), lon_values.tolist()))


def simplify_trajectory(
    trajectory: pd.DataFrame,
    algorithm: str,
    compression_ratio: float,
) -> np.ndarray:
    """Return simplified points as an Nx2 array (lat, lon)."""
    budget = max(2, int(len(trajectory) / compression_ratio))

    if algorithm in {"original", "none"}:
        return trajectory[["lat", "lon"]].to_numpy(dtype=float)
    if algorithm == "proposed":
        simplified, _ = proposed_simplification(trajectory, budget=budget)
        return simplified
    return simplify_with_budget(trajectory, algorithm=algorithm, budget=budget)


def format_metric_value(value: float) -> str:
    """Format metric values for overlay display."""
    if value is None or not np.isfinite(value):
        return "-"
    if abs(value) >= 1000:
        return f"{value:.1f}"
    if abs(value) >= 100:
        return f"{value:.2f}"
    return f"{value:.3f}"


def add_metrics_overlay(
    fmap: folium.Map,
    metric_rows: Sequence[dict],
) -> None:
    """Add a fixed bottom-left HTML overlay with per-layer metrics (no averaging)."""
    display_names = {
        "dp": "DP",
        "squish": "SQUISH",
        "vw": "VW",
        "sw": "SW",
        "rw": "RW",
        "proposed": "Proposed",
        "rdp": "DP",
        "sliding_window": "SW",
        "greedy_policy": "Greedy Policy (RL)",
        "greedy-policy": "Greedy Policy (RL)",
    }
    metric_columns = [
        ("hausdorff_distance", "HD (m)"),
        ("frechet_distance", "Fréchet (m)"),
        ("average_pte", "APTE (m)"),
        ("ped", "PED (m)"),
        ("sed", "SED (m)"),
        ("dad", "DAD (deg)"),
        ("sad", "SAD (m/s)"),
        ("issd", "ISSD (m*s)"),
    ]

    rows_html = []
    for row_data in metric_rows:
        algo_key = str(row_data.get("algorithm", "")).lower()
        traj_id = row_data.get("trajectory_id", "-")
        ratio_label = row_data.get("ratio_label", "-")
        row = [
            f"<td style='padding:3px 6px;'>{ratio_label}</td>",
            f"<td style='padding:3px 6px;'>{traj_id}</td>",
            f"<td style='padding:3px 6px; font-weight:600;'>{display_names.get(algo_key, row_data.get('algorithm', '-'))}</td>",
        ]
        for metric_name, _ in metric_columns:
            row.append(
                f"<td style='padding:3px 6px; text-align:right;'>{format_metric_value(row_data.get(metric_name, np.nan))}</td>"
            )
        layer_label = row_data.get("layer_label", "")
        rows_html.append(
            f"<tr data-ratio='{ratio_label}' data-traj='{traj_id}' data-layer='{layer_label}'>{''.join(row)}</tr>"
        )

    if not rows_html:
        return

    header_cells = "".join(
        [f"<th style='padding:4px 6px; text-align:right;'>{abbr}</th>" for _, abbr in metric_columns]
    )

    overlay_html = f"""
    <div style="
        position: fixed;
        bottom: 18px;
        left: 18px;
        z-index: 9999;
        background: rgba(255,255,255,0.92);
        border: 1px solid #888;
        border-radius: 6px;
        box-shadow: 0 1px 6px rgba(0,0,0,0.25);
        padding: 8px;
        max-width: 96vw;
        max-height: 38vh;
        overflow-x: auto;
        overflow-y: auto;
        font-size: 11px;
    ">
      <div style="font-weight: 700; margin-bottom: 6px;">Error Metrics (per trajectory & algorithm)</div>
      <table id="metrics-overlay-table" style="border-collapse: collapse;">
        <thead>
          <tr>
            <th style="padding:4px 6px; text-align:left;">Ratio</th>
            <th style="padding:4px 6px; text-align:left;">Traj</th>
            <th style="padding:4px 6px; text-align:left;">Algo</th>
            {header_cells}
          </tr>
        </thead>
        <tbody>
          {''.join(rows_html)}
        </tbody>
      </table>
    </div>
    """
    fmap.get_root().html.add_child(Element(overlay_html))


def add_selection_controls_overlay(
    fmap: folium.Map,
    ratios: Sequence[float],
    trajectory_labels: Sequence[str],
) -> None:
    """Add compression and trajectory dropdown controls with fixed positions."""
    if not ratios:
        return

    sorted_ratios = sorted(float(r) for r in ratios)
    ratio_labels = [f"{ratio:.2f}x" for ratio in sorted_ratios]
    options_html = "".join([f"<option value='{label}'>{label}</option>" for label in ratio_labels])
    sorted_trajectories = sorted(set(trajectory_labels))
    trajectory_options_html = "".join([f"<option value='{traj}'>{traj}</option>" for traj in sorted_trajectories])
    first_label = ratio_labels[0]
    first_trajectory = sorted_trajectories[0] if sorted_trajectories else ""

    controls_html = f"""
    <style>
      /* Keep basemap switcher on right-top, hide trajectory overlay checkboxes there. */
      .leaflet-control-layers-overlays {{
        display: none !important;
      }}
      .leaflet-control-layers-separator {{
        display: none !important;
      }}
    </style>
    <div style="
        position: fixed;
        top: 18px;
        left: 18px;
        z-index: 9999;
        background: rgba(255,255,255,0.92);
        border: 1px solid #888;
        border-radius: 6px;
        box-shadow: 0 1px 6px rgba(0,0,0,0.25);
        padding: 8px;
        font-size: 12px;
    ">
      <label for="ratio-select" style="font-weight:700; margin-right:6px;">Compression</label>
      <select id="ratio-select" style="padding:2px 4px;">{options_html}</select>
    </div>
    <div style="
        position: fixed;
        top: 72px;
        left: 18px;
        z-index: 9999;
        background: rgba(255,255,255,0.92);
        border: 1px solid #888;
        border-radius: 6px;
        box-shadow: 0 1px 6px rgba(0,0,0,0.25);
        padding: 8px;
        font-size: 12px;
    ">
      <label for="trajectory-select" style="font-weight:700; margin-right:6px;">Trajectory</label>
      <select id="trajectory-select" style="padding:2px 4px;">{trajectory_options_html}</select>
    </div>
    <div style="
        position: fixed;
        top: 126px;
        left: 18px;
        z-index: 9999;
        background: rgba(255,255,255,0.92);
        border: 1px solid #888;
        border-radius: 6px;
        box-shadow: 0 1px 6px rgba(0,0,0,0.25);
        padding: 8px;
        font-size: 11px;
        max-height: 34vh;
        overflow-y: auto;
        min-width: 260px;
    ">
      <div style="font-weight:700; margin-bottom:4px;">Visible Layers</div>
      <div id="layer-list-card"></div>
    </div>
    <script>
      function refreshLayerList(selectedRatioLabel, selectedTrajectory) {{
        var container = document.getElementById('layer-list-card');
        if (!container) return;
        var overlayList = document.querySelector('.leaflet-control-layers-overlays');
        if (!overlayList) {{
          container.innerHTML = '<div>No layers found.</div>';
          return;
        }}
        var labels = overlayList.querySelectorAll('label');
        var items = [];
        for (var i = 0; i < labels.length; i++) {{
          var labelText = (labels[i].textContent || '').trim();
          if (labelText.indexOf(' | ') === -1) continue;
          var parts = labelText.split(' | ');
          var ratioPrefix = parts[0].trim();
          var trajPart = parts.length > 1 ? parts[1].split(' - ')[0].trim() : '';
          var input = labels[i].querySelector('input[type=\"checkbox\"]');
          if (!input) continue;
          var layerKey = 'overlay-layer-' + i;
          input.setAttribute('data-layer-key', layerKey);
          if (ratioPrefix === selectedRatioLabel && trajPart === selectedTrajectory) {{
            items.push({{
              key: layerKey,
              label: labelText,
              checked: input.checked
            }});
          }}
        }}
        if (items.length === 0) {{
          container.innerHTML = '<div>No matching layers.</div>';
          return;
        }}
        var html = '<div style="display:flex; flex-direction:column; gap:4px;">';
        for (var j = 0; j < items.length; j++) {{
          var checkedAttr = items[j].checked ? 'checked' : '';
          html += '<label style="display:flex; align-items:center; gap:6px;">' +
                  '<input class="layer-card-checkbox" type="checkbox" data-layer-key="' + items[j].key + '" ' + checkedAttr + '>' +
                  '<span>' + items[j].label + '</span>' +
                  '</label>';
        }}
        html += '</div>';
        container.innerHTML = html;

        var boxList = container.querySelectorAll('.layer-card-checkbox');
        for (var b = 0; b < boxList.length; b++) {{
          boxList[b].addEventListener('change', function() {{
            var key = this.getAttribute('data-layer-key');
            var hiddenInput = overlayList.querySelector('input[data-layer-key=\"' + key + '\"]');
            if (!hiddenInput) return;
            if (hiddenInput.checked !== this.checked) {{
              hiddenInput.click();
            }}
          }});
        }}
      }}

      function applyVisibility() {{
        var ratioSelect = document.getElementById('ratio-select');
        var trajectorySelect = document.getElementById('trajectory-select');
        if (!ratioSelect || !trajectorySelect) return;
        var selectedRatioLabel = ratioSelect.value;
        var selectedTrajectory = trajectorySelect.value;

        // Toggle overlay layers by their label prefix "<ratio>x | ..."
        var overlayList = document.querySelector('.leaflet-control-layers-overlays');
        if (overlayList) {{
          var labels = overlayList.querySelectorAll('label');
          for (var i = 0; i < labels.length; i++) {{
            var labelText = labels[i].textContent || '';
            if (labelText.indexOf(' | ') === -1) continue;
            var parts = labelText.split(' | ');
            var ratioPrefix = parts[0].trim();
            var trajPart = parts.length > 1 ? parts[1].split(' - ')[0].trim() : '';
            var input = labels[i].querySelector('input[type=\"checkbox\"]');
            if (!input) continue;
            var trajMatch = (trajPart === selectedTrajectory);
            var shouldBeChecked = (ratioPrefix === selectedRatioLabel) && trajMatch;
            if (input.checked !== shouldBeChecked) {{
              input.click();
            }}
          }}
        }}

        var table = document.getElementById('metrics-overlay-table');
        if (table) {{
          var rows = table.querySelectorAll('tbody tr');
          for (var r = 0; r < rows.length; r++) {{
            var ratioMatch = (rows[r].getAttribute('data-ratio') === selectedRatioLabel);
            var rowTraj = rows[r].getAttribute('data-traj');
            var trajMatch = (rowTraj === selectedTrajectory);
            rows[r].style.display = (ratioMatch && trajMatch) ? '' : 'none';
          }}
        }}
        refreshLayerList(selectedRatioLabel, selectedTrajectory);
      }}
      window.addEventListener('load', function() {{
        var ratioSelect = document.getElementById('ratio-select');
        var trajectorySelect = document.getElementById('trajectory-select');
        if (!ratioSelect || !trajectorySelect) return;
        ratioSelect.value = '{first_label}';
        trajectorySelect.value = '{first_trajectory}';
        // Delay first toggle slightly so Leaflet layers/control are fully initialized.
        setTimeout(function() {{
          applyVisibility();
        }}, 50);
        ratioSelect.addEventListener('change', function() {{
          applyVisibility();
        }});
        trajectorySelect.addEventListener('change', function() {{
          applyVisibility();
        }});
      }});
    </script>
    """
    fmap.get_root().html.add_child(Element(controls_html))


def add_zoom_bottom_right(fmap: folium.Map) -> None:
    """Place map zoom controls at bottom-right."""
    map_name = fmap.get_name()
    script = f"""
    <script>
      window.addEventListener('load', function() {{
        if (typeof {map_name} !== 'undefined') {{
          L.control.zoom({{position: 'bottomright'}}).addTo({map_name});
        }}
      }});
    </script>
    """
    fmap.get_root().html.add_child(Element(script))


def add_algorithm_legend_overlay(
    fmap: folium.Map,
    algorithms: Sequence[str],
) -> None:
    """Show algorithm color legend on map."""
    display_names = {
        "original": "Original",
        "dp": "DP",
        "squish": "SQUISH",
        "vw": "VW",
        "sw": "SW",
        "rw": "RW",
        "greedy_policy": "Greedy Policy (RL)",
        "proposed": "Proposed",
    }

    seen = set()
    rows = []
    for algorithm in algorithms:
        key = algorithm.lower().strip()
        if key in seen:
            continue
        seen.add(key)
        color = ALGORITHM_COLORS.get(key, "black")
        label = display_names.get(key, algorithm)
        rows.append(
            f"<div style='display:flex; align-items:center; gap:6px; margin:2px 0;'>"
            f"<span style='display:inline-block; width:12px; height:12px; background:{color}; border:1px solid #444;'></span>"
            f"<span>{label}</span>"
            f"</div>"
        )

    if not rows:
        return

    legend_html = f"""
    <div style="
        position: fixed;
        top: 170px;
        right: 18px;
        z-index: 9998;
        background: rgba(255,255,255,0.92);
        border: 1px solid #888;
        border-radius: 6px;
        box-shadow: 0 1px 6px rgba(0,0,0,0.25);
        padding: 8px;
        font-size: 11px;
        min-width: 120px;
    ">
      <div style="font-weight:700; margin-bottom:4px;">Algorithm Legend</div>
      {''.join(rows)}
    </div>
    """
    fmap.get_root().html.add_child(Element(legend_html))


def add_basemap_layers(fmap: folium.Map) -> None:
    """Add selectable basemap layers."""
    folium.TileLayer(
        tiles="OpenStreetMap",
        name="OpenStreetMap",
        overlay=False,
        control=True,
    ).add_to(fmap)
    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
        name="Carto Light",
        attr='&copy; OpenStreetMap contributors &copy; <a href="https://carto.com/">CARTO</a>',
        overlay=False,
        control=True,
    ).add_to(fmap)
    folium.TileLayer(
        tiles="https://stamen-tiles.a.ssl.fastly.net/toner/{z}/{x}/{y}.png",
        name="Stamen Toner",
        attr='Map tiles by <a href="http://stamen.com">Stamen Design</a>, '
             'under <a href="http://creativecommons.org/licenses/by/3.0">CC BY 3.0</a>. '
             'Data by <a href="http://openstreetmap.org">OpenStreetMap</a>, under ODbL.',
        overlay=False,
        control=True,
    ).add_to(fmap)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        name="Esri WorldGrayCanvas",
        attr='Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ',
        overlay=False,
        control=True,
    ).add_to(fmap)


def create_osm_map(
    trajectories_file: str,
    output_file: str,
    max_trajectories: int = 30,
    max_points_per_trajectory: int = 1200,
) -> Path:
    """Load trajectories from pickle and save an OSM HTML map."""
    with open(trajectories_file, "rb") as f:
        trajectories = pickle.load(f)

    if not isinstance(trajectories, list) or len(trajectories) == 0:
        raise ValueError("Trajectory file is empty or not a list of trajectories.")

    selected_ids = sample_indices(len(trajectories), max_trajectories)
    selected_trajectories = [trajectories[i] for i in selected_ids]

    all_lat: List[float] = []
    all_lon: List[float] = []
    prepared_paths: List[List[Tuple[float, float]]] = []

    for traj in selected_trajectories:
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

        if len(lat_values) > max_points_per_trajectory:
            keep = np.linspace(0, len(lat_values) - 1, max_points_per_trajectory, dtype=int)
            lat_values = lat_values[keep]
            lon_values = lon_values[keep]

        path = list(zip(lat_values.tolist(), lon_values.tolist()))
        prepared_paths.append(path)
        all_lat.extend(lat_values.tolist())
        all_lon.extend(lon_values.tolist())

    if not prepared_paths:
        raise ValueError(
            "No plottable trajectories found in the pickle. "
            "If this file was generated before the loader fix, re-run preprocessing."
        )

    center = [float(np.mean(all_lat)), float(np.mean(all_lon))]
    fmap = folium.Map(location=center, zoom_start=11, tiles=None, control_scale=True, zoom_control=False)
    add_basemap_layers(fmap)

    palette = [
        "blue", "red", "green", "purple", "orange", "darkred", "darkblue",
        "cadetblue", "darkgreen", "black",
    ]
    for idx, path in enumerate(prepared_paths):
        color = palette[idx % len(palette)]
        folium.PolyLine(path, weight=3, opacity=0.85, color=color).add_to(fmap)
        folium.CircleMarker(path[0], radius=4, color="green", fill=True, fill_opacity=0.8).add_to(fmap)
        folium.CircleMarker(path[-1], radius=4, color="red", fill=True, fill_opacity=0.8).add_to(fmap)

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    add_zoom_bottom_right(fmap)
    fmap.save(str(output_path))
    return output_path


def create_osm_comparison_map(
    trajectories_file: str,
    output_file: str,
    algorithms: Sequence[str],
    compression_ratios: Sequence[float],
    max_trajectories: int = 5,
    max_points_per_trajectory: int = 1000,
) -> Path:
    """Save an OSM map with toggleable layers for original + simplified trajectories."""
    with open(trajectories_file, "rb") as f:
        trajectories = pickle.load(f)

    if not isinstance(trajectories, list) or len(trajectories) == 0:
        raise ValueError("Trajectory file is empty or not a list of trajectories.")

    selected_ids = sample_indices(len(trajectories), max_trajectories)
    selected_pairs = [(idx, trajectories[idx]) for idx in selected_ids]

    centers: List[Tuple[float, float]] = []
    valid_pairs: List[Tuple[int, pd.DataFrame]] = []

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
        raise ValueError("No valid trajectories available for comparison map.")

    center_lat = float(np.mean([c[0] for c in centers]))
    center_lon = float(np.mean([c[1] for c in centers]))
    fmap = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=11,
        tiles=None,
        control_scale=True,
        zoom_control=False,
    )
    add_basemap_layers(fmap)

    fallback_colors = ["darkblue", "darkpurple", "darkgreen", "darkred", "pink"]

    metric_rows: List[dict] = []
    default_ratio = compression_ratios[0] if len(compression_ratios) > 0 else None

    for traj_index, (traj_id, traj) in enumerate(valid_pairs):
        # Ensure numeric lat/lon for simplification algorithms.
        traj["lat"] = pd.to_numeric(traj["lat"], errors="coerce")
        traj["lon"] = pd.to_numeric(traj["lon"], errors="coerce")
        traj = traj.dropna(subset=["lat", "lon"]).reset_index(drop=True)
        if len(traj) < 2:
            continue

        for ratio in compression_ratios:
            for algorithm in algorithms:
                layer_name = f"{ratio:.2f}x | T{traj_id:04d} - {algorithm}"
                layer = folium.FeatureGroup(
                    name=layer_name,
                    # Only show first compression ratio on initial load.
                    show=(ratio == default_ratio),
                )
                try:
                    simplified = simplify_trajectory(traj, algorithm, compression_ratio=ratio)
                    if simplified.shape[0] < 2:
                        continue
                    path = downsample_path(simplified[:, 0], simplified[:, 1], max_points=max_points_per_trajectory)
                    color = ALGORITHM_COLORS.get(algorithm.lower(), fallback_colors[traj_index % len(fallback_colors)])
                    folium.PolyLine(
                        path,
                        weight=3 if algorithm == "original" else 2,
                        opacity=0.9 if algorithm == "original" else 0.75,
                        color=color,
                        tooltip=f"Trajectory {traj_id} | {algorithm} | {ratio:.2f}x | points={len(path)}",
                    ).add_to(layer)
                    folium.CircleMarker(
                        path[0], radius=3, color="green", fill=True, fill_opacity=0.8
                    ).add_to(layer)
                    folium.CircleMarker(
                        path[-1], radius=3, color="red", fill=True, fill_opacity=0.8
                    ).add_to(layer)
                    layer.add_to(fmap)

                    algo_key = algorithm.lower()
                    if algo_key not in {"original", "none"}:
                        metrics = compute_all_metrics(traj, simplified, original_indices=None)
                        metric_rows.append({
                            "ratio_label": f"{ratio:.2f}x",
                            "trajectory_id": f"T{traj_id:04d}",
                            "algorithm": algorithm,
                            "layer_label": layer_name,
                            "hausdorff_distance": metrics.get("hausdorff_distance"),
                            "frechet_distance": metrics.get("frechet_distance"),
                            "average_pte": metrics.get("average_pte"),
                            "ped": metrics.get("ped"),
                            "sed": metrics.get("sed"),
                            "dad": metrics.get("dad"),
                            "sad": metrics.get("sad"),
                            "issd": metrics.get("issd"),
                        })
                except Exception:
                    continue

    add_metrics_overlay(fmap, metric_rows)
    folium.LayerControl(position="topright", collapsed=False).add_to(fmap)
    trajectory_labels = [f"T{traj_id:04d}" for traj_id, _ in valid_pairs]
    add_selection_controls_overlay(fmap, compression_ratios, trajectory_labels)
    add_algorithm_legend_overlay(fmap, algorithms)
    add_zoom_bottom_right(fmap)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(output_path))
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize trajectories on OpenStreetMap")
    parser.add_argument(
        "--trajectories-file",
        type=str,
        default="data/processed/trajectories.pkl",
        help="Path to trajectories pickle file",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="results/figures/trajectories_osm.html",
        help="Output HTML path",
    )
    parser.add_argument(
        "--max-trajectories",
        type=int,
        default=30,
        help="Maximum number of trajectories to render",
    )
    parser.add_argument(
        "--max-points-per-trajectory",
        type=int,
        default=1200,
        help="Maximum number of points rendered per trajectory",
    )
    parser.add_argument(
        "--comparison",
        action="store_true",
        help="Create comparison map with original + simplified layers",
    )
    parser.add_argument(
        "--algorithms",
        type=str,
        default="original,dp,vw,squish,rw,greedy_policy,proposed",
        help="Comma-separated algorithms for comparison map",
    )
    parser.add_argument(
        "--compression-ratio",
        type=float,
        default=5.0,
        help="Compression ratio used to simplify trajectories in comparison mode",
    )
    parser.add_argument(
        "--compression-ratios",
        type=str,
        default="",
        help="Comma-separated compression ratios for dropdown map view (e.g., 2,5,10)",
    )
    args = parser.parse_args()

    if args.comparison:
        algorithms = [algo.strip() for algo in args.algorithms.split(",") if algo.strip()]
        if args.compression_ratios.strip():
            compression_ratios = [float(x.strip()) for x in args.compression_ratios.split(",") if x.strip()]
        else:
            compression_ratios = [args.compression_ratio]
        output_path = create_osm_comparison_map(
            trajectories_file=args.trajectories_file,
            output_file=args.output_file,
            algorithms=algorithms,
            compression_ratios=compression_ratios,
            max_trajectories=args.max_trajectories,
            max_points_per_trajectory=args.max_points_per_trajectory,
        )
    else:
        output_path = create_osm_map(
            trajectories_file=args.trajectories_file,
            output_file=args.output_file,
            max_trajectories=args.max_trajectories,
            max_points_per_trajectory=args.max_points_per_trajectory,
        )
    print(f"Saved OSM map to: {output_path}")


if __name__ == "__main__":
    main()

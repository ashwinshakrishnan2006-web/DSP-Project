"""
CrimeSense – Interactive Map Renderer
======================================
Generates an interactive HTML map using Folium that shows:

1. Choropleth-style risk colouring of grid cells (green/orange/red).
2. DBSCAN hotspot circles with cluster size labels.
3. (Optional) K-Means cluster centroids as markers.
4. A legend and layer control.

Output: output/crime_map.html
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Risk level → Folium fill colour
RISK_COLORS = {
    "Low": "#2ecc71",       # green
    "Medium": "#f39c12",    # amber/orange
    "High": "#e74c3c",      # red
}

CHICAGO_CENTER = [41.85, -87.65]   # approximate geographic centre

OUTPUT_DIR = Path("output")


class MapRenderer:
    """
    Renders the CrimeSense interactive crime map.

    Parameters
    ----------
    grid_df : pd.DataFrame
        Grid DataFrame from RiskLabeler, with columns:
        centroid_lat, centroid_lon, risk_label, incident_count, cell_lat, cell_lon.
    hotspot_df : pd.DataFrame, optional
        DBSCAN hotspot summary from DBSCANHotspotDetector.
    output_path : str or Path, optional
        Where to save the HTML file. Defaults to output/crime_map.html.
    """

    def __init__(
        self,
        grid_df: pd.DataFrame,
        hotspot_df: Optional[pd.DataFrame] = None,
        output_path: Optional[str | Path] = None,
    ):
        self.grid_df = grid_df
        self.hotspot_df = hotspot_df
        self.output_path = Path(output_path or OUTPUT_DIR / "crime_map.html")

    def render(self) -> Path:
        """
        Build and save the interactive map.

        Returns
        -------
        Path   Absolute path to the saved HTML file.
        """
        try:
            import folium
            from folium.plugins import MarkerCluster
        except ImportError:
            raise ImportError(
                "folium is required for map rendering. "
                "Install it with:  pip install folium"
            )

        logger.info("Rendering interactive crime map…")

        m = folium.Map(
            location=CHICAGO_CENTER,
            zoom_start=11,
            tiles="CartoDB positron",
        )

        # 1. Risk grid cells
        self._add_risk_layer(m)

        # 2. DBSCAN hotspot circles
        if self.hotspot_df is not None and not self.hotspot_df.empty:
            self._add_dbscan_layer(m)

        # 3. Legend
        self._add_legend(m)

        # 4. Layer control
        folium.LayerControl(collapsed=False).add_to(m)

        # Save
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        m.save(str(self.output_path))
        logger.info("Map saved → %s", self.output_path.resolve())
        return self.output_path.resolve()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _add_risk_layer(self, m) -> None:
        """Add coloured grid-cell rectangles to the map."""
        import folium

        cfg_lat = 0.004494  # ≈ GRID_LAT_STEP
        cfg_lon = 0.006014  # ≈ GRID_LON_STEP

        feature_group = folium.FeatureGroup(name="Risk Grid", show=True)

        for _, row in self.grid_df.iterrows():
            lat = row["cell_lat"]
            lon = row["cell_lon"]
            risk = row.get("risk_label", "Low")
            count = row.get("incident_count", "?")
            color = RISK_COLORS.get(risk, "#95a5a6")

            folium.Rectangle(
                bounds=[[lat, lon], [lat + cfg_lat, lon + cfg_lon]],
                color=None,
                fill=True,
                fill_color=color,
                fill_opacity=0.45,
                tooltip=f"Risk: {risk}<br>Incidents: {count}",
            ).add_to(feature_group)

        feature_group.add_to(m)

    def _add_dbscan_layer(self, m) -> None:
        """Add DBSCAN hotspot circles to the map."""
        import folium

        fg = folium.FeatureGroup(name="DBSCAN Hotspots", show=True)

        for _, row in self.hotspot_df.iterrows():
            lat = row["centroid_lat"]
            lon = row["centroid_lon"]
            size = row.get("size", 0)
            cid = row.get("cluster_id", "?")
            top_crime = row.get("top_crime", "Unknown")

            # Radius scaled by cluster size (max ~500 m visual)
            radius_m = min(500, max(100, size / 50 * 200))

            folium.Circle(
                location=[lat, lon],
                radius=radius_m,
                color="#c0392b",
                fill=True,
                fill_color="#e74c3c",
                fill_opacity=0.5,
                tooltip=(
                    f"Hotspot #{cid}<br>"
                    f"Incidents: {size}<br>"
                    f"Top crime: {top_crime}"
                ),
            ).add_to(fg)

        fg.add_to(m)

    @staticmethod
    def _add_legend(m) -> None:
        """Inject an HTML legend into the map."""
        legend_html = """
        <div style="
            position: fixed;
            bottom: 40px; left: 40px;
            background: white;
            border: 2px solid #555;
            border-radius: 6px;
            padding: 12px 18px;
            font-family: Arial, sans-serif;
            font-size: 13px;
            z-index: 9999;
            box-shadow: 3px 3px 6px rgba(0,0,0,0.3);
        ">
            <b>Risk Level</b><br>
            <span style="color:#2ecc71;">&#9632;</span> Low<br>
            <span style="color:#f39c12;">&#9632;</span> Medium<br>
            <span style="color:#e74c3c;">&#9632;</span> High<br>
            <hr style="margin:6px 0;">
            <span style="color:#c0392b;">&#9711;</span> DBSCAN Hotspot
        </div>
        """
        m.get_root().html.add_child(
            __import__("folium").Element(legend_html)
        )


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def render_map(
    grid_df: pd.DataFrame,
    hotspot_df: Optional[pd.DataFrame] = None,
    output_path: Optional[str | Path] = None,
) -> Path:
    """One-call map render."""
    return MapRenderer(grid_df, hotspot_df, output_path).render()

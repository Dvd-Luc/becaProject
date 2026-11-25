"""Build the local interactive page of the full-survey detections (map, season, night, presence).

    python src/survey/build_viz.py --dry-run     # counts and checks only, writes nothing
    python src/survey/build_viz.py               # writes the page

Reads the full-survey tables (drive_detection.dir), the audit results and the microphone
positions (viz.sources, GeoPackages), and writes ONE self-contained HTML file: data,
page code, d3 and the département outlines are inlined, so it opens by double-click
(file://), offline, with no server. Each view has SVG / PNG / CSV export buttons.

The microphone positions are sensitive (snipe breeding sites). They stay on this
computer: the GeoPackages and the page are gitignored, this script never prints a
position, and the page must not be uploaded anywhere. viz.round_coords (degrees)
blurs the positions written into the page.

What the page holds, per site: drive, split, département, hours recorded, kept events
and events per hour per class, presence status per class (results/survey/
presence_full_gen<K>.csv: present / not confirmed / not judged; classes not in
review.presence_labels are "not checked"); per site x evening: hours recorded and
events per class; per event: minutes from sunset; header: precision per class from the
audit (kept row when the class has a min_event_conf, class row otherwise).

Minutes from sunset come from the recorders' own schedule: two one-hour files from
sunset (viz.schedule_start_min, 0) to sunset + 2 h, so an event lies at
(event time - first file start of that evening) + schedule_start_min. This needs no
clock or time-zone correction (most recorder clocks stayed on UTC+1 after the March
time change, some follow legal time; either way the first file starts at sunset in
the recorder's own clock). An evening is a site and a date; its files are sorted by
start. Set-up tests are left out (files starting before noon or shorter than 30 min).
A single-file evening counts as file 1 if it starts with the site's neighbouring
evenings (+-3 days, 20 min), as file 2 if about 60 min later, otherwise it is left
out. The counts of each case are printed.

Positions are matched to sites.csv by the exact site name (viz.sources name_column),
otherwise by the recorder number (number_column "MIC011" / "Mic049" against
"SMMICRO_011" in the site name). Unmatched sites are listed by name. Sites are grouped
by département: the code in the site name ("_25_" = Doubs), or for names without one,
the département outline (viz.departements) holding the position.

The map's base layer (Plan IGN, aerial photo, relief, land cover, wetlands) is loaded
from IGN's Géoplateforme only when chosen in the page: IGN then sees which tiles are
viewed, never the positions. "Off" (default) needs no network.

Writes:
    <viz.out>   e.g. results/survey/viz/full_gen2.html (gitignored)

Run from code/.
"""
import argparse
import csv
import json
import re
import sqlite3
import struct
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import load

TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
NUMBER_IN_SITE = re.compile(r"SMMICRO_(\d+)", re.IGNORECASE)
MIN_EVENING_HOUR = 12           # files starting earlier are set-up tests, not evening recordings
MIN_FILE_S = 1800               # shorter files are tests
NEIGHBOUR_DAYS = 3              # single-file evenings: compared with the evenings this close
MATCH_MIN = 20                  # ...whose first file starts within this many minutes


def read_dicts(path):
    """Rows of a CSV file as dicts.

    Args:
        path (str | Path): CSV file with a header row.

    Returns:
        list[dict[str, str]]: One dict per row.
    """
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------- positions
def gpkg_point(blob):
    """(x, y) of a GeoPackage point geometry (first point of a multipoint).

    Args:
        blob (bytes): GeoPackage geometry: "GP" header, optional envelope, then WKB.

    Returns:
        tuple[float, float] | None: Coordinates in the layer's CRS, None if empty.

    Raises:
        ValueError: If the blob is not a GeoPackage point or multipoint.
    """
    if blob is None:
        return None
    if blob[:2] != b"GP":
        raise ValueError("not a GeoPackage geometry")
    flags = blob[3]
    if flags & 0b10000:                    # empty geometry
        return None
    envelope = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}[(flags >> 1) & 0b111]
    wkb = blob[8 + envelope:]
    order = "<" if wkb[0] == 1 else ">"
    geom_type = struct.unpack(order + "I", wkb[1:5])[0] % 1000
    if geom_type == 4:                     # multipoint: skip to its first point
        wkb = wkb[9:]
        order = "<" if wkb[0] == 1 else ">"
        geom_type = struct.unpack(order + "I", wkb[1:5])[0] % 1000
    if geom_type != 1:
        raise ValueError(f"geometry type {geom_type} is not a point")
    return struct.unpack(order + "dd", wkb[5:21])


def read_positions(viz_cfg):
    """Microphone points of every viz source, in WGS84.

    Args:
        viz_cfg (dict): The viz config section (sources).

    Returns:
        list[dict]: One dict per point: name (str or ""), number (int or None),
            lat, lon (float), source (str).

    Raises:
        SystemExit: If a GeoPackage or its layer is missing.
    """
    from pyproj import Transformer
    points = []
    for source in viz_cfg["sources"]:
        path = Path(source["path"])
        if not path.exists():
            raise SystemExit(f"{path} not found (viz.sources): copy the GeoPackage there")
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        layer = source["layer"]
        found = con.execute("select srs_id from gpkg_contents where table_name = ?", (layer,)).fetchone()
        if not found:
            raise SystemExit(f"layer {layer} not in {path}")
        geom_column = con.execute("select column_name from gpkg_geometry_columns where table_name = ?",
                                  (layer,)).fetchone()[0]
        to_wgs84 = Transformer.from_crs(f"EPSG:{found[0]}", "EPSG:4326", always_xy=True)
        columns = [geom_column] + [column for column in (source.get("name_column"), source.get("number_column"))
                                   if column]
        query = "select " + ", ".join(f'"{column}"' for column in columns) + f' from "{layer}"'
        for row in con.execute(query):
            xy = gpkg_point(row[0])
            if xy is None:
                continue
            values = dict(zip(columns[1:], row[1:], strict=True))
            name = values.get(source.get("name_column")) or ""
            number_text = str(values.get(source.get("number_column")) or "")
            digits = re.search(r"(\d+)", number_text)
            lon, lat = to_wgs84.transform(*xy)
            points.append({"name": str(name).strip(), "number": int(digits.group(1)) if digits else None,
                           "lat": lat, "lon": lon, "source": path.name})
        con.close()
    return points


def match_positions(site_names, points):
    """Position of each site: exact name first, otherwise a unique recorder number.

    Args:
        site_names (list[str]): Site names (sites.csv).
        points (list[dict]): Points from read_positions.

    Returns:
        tuple[dict[str, dict], list[str], int]: (site -> point, sites without a
            position, number of points used by no site).
    """
    by_name = {point["name"]: point for point in points if point["name"]}
    by_number = defaultdict(list)
    for point in points:
        if point["number"] is not None:
            by_number[point["number"]].append(point)
    matched, missing = {}, []
    for site in site_names:
        if site in by_name:
            matched[site] = by_name[site]
            continue
        number = NUMBER_IN_SITE.search(site)
        candidates = by_number.get(int(number.group(1)), []) if number else []
        if len(candidates) == 1:
            matched[site] = candidates[0]
        else:
            missing.append(site)
    used = {id(point) for point in matched.values()}
    return matched, missing, sum(1 for point in points if id(point) not in used)


# ---------------------------------------------------------------- tables
def presence_status(cfg):
    """Presence status per (site, class) from the presence results.

    Args:
        cfg (dict): The whole project config.

    Returns:
        dict[tuple[str, str], str]: (site, label) -> "present" / "not confirmed" / "not judged".
    """
    path = Path("results/survey") / f"presence_full_gen{cfg['drive_detection']['gen']}.csv"
    if not path.exists():
        return {}
    return {(row["site"], row["label"]): row["status"] for row in read_dicts(path)}


def precision_header(cfg):
    """Measured precision per class, for the page header.

    Args:
        cfg (dict): The whole project config.

    Returns:
        dict[str, dict]: label -> {precision, low, high, judged, scope, drives}.
    """
    gen = cfg["drive_detection"]["gen"]
    path = Path("results/survey") / f"audit_full_gen{gen}.csv"
    if not path.exists():
        return {}
    rows = read_dicts(path)
    review_dir = Path(cfg["review"]["dir"])
    drives = sorted(p.name[len("audit_"):-len("_queue.csv")] for p in review_dir.glob("audit_*_queue.csv"))
    out = {}
    for row in rows:
        if row["scope"] not in ("class", "kept"):
            continue
        if row["scope"] == "class" and row["label"] in out and out[row["label"]]["scope"] == "kept":
            continue
        out[row["label"]] = {"precision": float(row["precision"]), "low": float(row["ci_low"]),
                             "high": float(row["ci_high"]), "judged": int(row["judged"]),
                             "scope": row["scope"], "bin": row["bin"], "drives": drives}
    return out


def build_data(cfg, matched, round_coords, depts):
    """Everything the page shows, as one JSON-ready dict.

    Args:
        cfg (dict): The whole project config.
        matched (dict[str, dict]): site -> position.
        round_coords (float | None): Round positions to this many degrees (None: exact).
        depts (dict): Département outlines (GeoJSON, "code" and "nom" properties).

    Returns:
        dict: meta, classes, sites, evenings, events (see the module docstring).
    """
    detection_cfg, viz_cfg = cfg["drive_detection"], cfg["viz"]
    out_dir = Path(detection_cfg["dir"])
    gen = detection_cfg["gen"]
    labels = [line.strip() for line in (Path(cfg["generations"]["dir"]) / f"gen_{gen}" / "model_Labels.txt")
              .read_text(encoding="utf-8").splitlines() if line.strip()]
    order = [label for label in ("Drumming", "Song", "Call") if label in labels] + \
            [label for label in labels if label not in ("Drumming", "Song", "Call")]
    sites_csv = {row["site"]: row for row in read_dicts(Path(cfg["survey"]["dir"]) / "sites.csv")}
    presence = presence_status(cfg)
    presence_labels = set(cfg["review"].get("presence_labels", []))

    files = {}
    for list_path in sorted((out_dir / "files").glob("*_files.csv")):
        for row in read_dicts(list_path):
            files[row["file_id"]] = row
    done = {row["file_id"] for row in read_dicts(out_dir / "file_summary.csv")}
    hours = Counter()
    evening_hours = defaultdict(float)
    for file_id in done:
        row = files[file_id]
        duration_h = float(row["duration_s"]) / 3600
        hours[row["site"]] += duration_h
        evening_hours[(row["site"], row["start"][:10])] += duration_h
    first_start, kept_files, evening_counts = evening_starts(files, done)
    for case, n in sorted(evening_counts.items()):
        print(f"    {case}: {n}")

    events = read_dicts(out_dir / "events.csv")
    n_events = Counter((event["site"], event["label"]) for event in events)
    evening_events = Counter((event["site"], event["start"][:10], event["label"]) for event in events)

    def rounded(value):
        """Position rounded to viz.round_coords degrees.

        Args:
            value (float): Latitude or longitude.

        Returns:
            float: Rounded value (5 decimals when no rounding is asked).
        """
        if round_coords:
            return round(round(value / round_coords) * round_coords, 6)
        return round(value, 5)

    dept_names = {feature["properties"]["code"]: feature["properties"]["nom"] for feature in depts["features"]}
    dept_site, from_position, disagree = {}, [], []
    for site in hours:
        if site not in matched:
            continue
        located = dept_at(matched[site]["lon"], matched[site]["lat"], depts)
        dept_site[site] = dept_of(site) or located
        if not dept_of(site):
            from_position.append(short_name(site))
        elif located and located != dept_of(site):
            disagree.append(short_name(site))
    if from_position:
        print(f"    département from the position (no code in the name): {', '.join(sorted(from_position))}")
    if disagree:
        print(f"warning: position outside the département of the name (name kept): {', '.join(sorted(disagree))}")

    site_rows, site_index = [], {}
    for site in sorted(dept_site, key=lambda name: (dept_site[name], short_name(name))):
        info = sites_csv.get(site, {})
        dept = dept_site[site]
        statuses = {}
        for label in order:
            if not n_events[(site, label)]:
                statuses[label] = "none"
            elif label in presence_labels:
                statuses[label] = presence.get((site, label), "not judged")
            else:
                statuses[label] = "not checked"
        site_index[site] = len(site_rows)
        site_rows.append({"site": site, "short": short_name(site), "drive": info.get("drive", ""),
                          "split": info.get("split", ""), "dept": dept,
                          "dept_name": f"{dept} {dept_names.get(dept, '')}".strip() if dept else "unknown",
                          "lat": rounded(matched[site]["lat"]), "lon": rounded(matched[site]["lon"]),
                          "hours": round(hours[site], 2),
                          "events": [n_events[(site, label)] for label in order],
                          "status": [statuses[label] for label in order]})

    evenings = []
    for (site, day), duration_h in sorted(evening_hours.items()):
        if site in site_index:
            evenings.append([site_index[site], day, round(duration_h, 3)]
                            + [evening_events[(site, day, label)] for label in order])

    schedule_start = viz_cfg["schedule_start_min"]
    event_rows, left_out, beyond = [], 0, 0
    minutes_of = defaultdict(list)
    for event in events:
        if event["site"] not in site_index:
            continue
        evening = kept_files.get(event["file_id"])
        if evening is None or evening not in first_start:
            left_out += 1
            continue
        when = datetime.strptime(event["event_time"], TIME_FORMAT)
        minutes = (when - first_start[evening]).total_seconds() / 60 + schedule_start
        beyond += not schedule_start <= minutes < schedule_start + 120
        minutes_of[event["label"]].append(minutes)
        event_rows.append([site_index[event["site"]], order.index(event["label"]), event["start"][:10],
                           round(minutes, 1), float(event["max_conf"])])
    if left_out:
        print(f"note: {left_out} events in files left out above: not in the time-from-sunset view")
    if beyond:
        print(f"note: {beyond} events outside the 2 h after the evening's first file (extra files)")
    print("median minutes from sunset: " + ", ".join(
        f"{label} {sorted(values)[len(values) // 2]:.0f}" for label, values in sorted(minutes_of.items())))

    detection_note = f"windows >= {detection_cfg['min_conf']}"
    if detection_cfg.get("min_event_conf"):
        detection_note += ", best window >= " + ", ".join(
            f"{label} {value}" for label, value in detection_cfg["min_event_conf"].items())
    return {"meta": {"gen": gen, "built": datetime.now().strftime("%Y-%m-%d %H:%M"),
                     "drives": sorted({row["drive"] for row in site_rows}),
                     "hours": round(sum(row["hours"] for row in site_rows), 1),
                     "detection": detection_note, "schedule_start_min": schedule_start,
                     "rounded": round_coords, "precision": precision_header(cfg),
                     "presence_labels": sorted(presence_labels)},
            "classes": order, "sites": site_rows, "evenings": evenings, "events": event_rows}


def dept_of(site):
    """Département code from a site name.

    Args:
        site (str): e.g. "2026_Beca_SMMICRO_049_25_Houtaud" or "BECA_SMMINI01_15_Coltines".

    Returns:
        str: The département code ("25"), "" if not found.
    """
    match = re.search(r"_(\d{2})_[^_\d]", site)
    return match.group(1) if match else ""


def in_ring(lon, lat, ring):
    """True if a point lies inside a polygon ring (ray casting).

    Args:
        lon (float): Longitude.
        lat (float): Latitude.
        ring (list[list[float]]): [lon, lat] vertices.

    Returns:
        bool: Inside or not.
    """
    inside = False
    for (x1, y1), (x2, y2) in zip(ring, ring[1:] + ring[:1], strict=True):
        if (y1 > lat) != (y2 > lat) and lon < (x2 - x1) * (lat - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def dept_at(lon, lat, depts):
    """Département whose outline contains a point.

    Args:
        lon (float): Longitude.
        lat (float): Latitude.
        depts (dict): GeoJSON FeatureCollection with "code" properties.

    Returns:
        str: The département code, "" if the point is in none.
    """
    for feature in depts["features"]:
        geometry = feature["geometry"]
        polygons = [geometry["coordinates"]] if geometry["type"] == "Polygon" else geometry["coordinates"]
        for rings in polygons:
            if in_ring(lon, lat, rings[0]) and not any(in_ring(lon, lat, hole) for hole in rings[1:]):
                return feature["properties"]["code"]
    return ""


def short_name(site):
    """Readable site label: commune first, then the recorder.

    Args:
        site (str): Site name.

    Returns:
        str: e.g. "Sainte Colombe · 041", "Coltines · MINI01", "Coualle Basse".
    """
    name = re.sub(r"^(2026_Beca_|BECA_)", "", site)
    match = re.match(r"(?:SMMICRO_(\d+)|SM(MINI\d+)|SM_([A-Za-z]+))_\d{2}_(.+)$", name)
    if not match:
        return name.replace("_", " ")
    recorder = next(group for group in match.groups()[:3] if group)
    return f"{match.group(4).replace('_', ' ')} · {recorder}"


def evening_starts(files, done):
    """Start of the first file of each site-evening, with the files kept or left out.

    The recorders start at sunset and record two one-hour files. An evening is a site and
    a date; its files are sorted by start. Files starting before MIN_EVENING_HOUR or
    shorter than MIN_FILE_S are set-up tests and left out. A single-file evening is
    "file 1" if it starts within MATCH_MIN minutes of the first-file start of the site's
    evenings within NEIGHBOUR_DAYS, "file 2" if about 60 min later, otherwise left out.

    Args:
        files (dict[str, dict]): file_id -> files/<drive>_files.csv row.
        done (set[str]): file_ids with BirdNET results.

    Returns:
        tuple[dict, dict, Counter]: ((site, date) -> first-file start datetime;
            file_id -> (site, date) for the files kept; counts per case).
    """
    counts = Counter()
    by_evening = defaultdict(list)
    for file_id in done:
        row = files[file_id]
        start = datetime.strptime(row["start"], TIME_FORMAT)
        if start.hour < MIN_EVENING_HOUR:
            counts["files left out: before noon"] += 1
        elif float(row["duration_s"]) < MIN_FILE_S:
            counts["files left out: shorter than 30 min"] += 1
        else:
            by_evening[(row["site"], row["start"][:10])].append((start, file_id))
    first, kept = {}, {}
    for key, items in by_evening.items():
        if len(items) >= 2:
            first[key] = min(start for start, _ in items)
            counts["evenings with 2+ files"] += 1
    minute_of_day = {key: start.hour * 60 + start.minute for key, start in first.items()}
    for (site, day), items in by_evening.items():
        if (site, day) in first:
            kept.update({file_id: (site, day) for _, file_id in items})
            continue
        start, file_id = items[0]
        neighbours = [(site, (start + timedelta(days=delta)).strftime("%Y-%m-%d"))
                      for delta in range(-NEIGHBOUR_DAYS, NEIGHBOUR_DAYS + 1) if delta]
        near = [minute_of_day[key] for key in neighbours if key in minute_of_day]
        if not near:
            counts["evenings left out: 1 file, no neighbour"] += 1
            continue
        usual = sorted(near)[len(near) // 2]
        own = start.hour * 60 + start.minute
        if abs(own - usual) <= MATCH_MIN:
            first[(site, day)] = start
            counts["evenings with 1 file: file 1"] += 1
        elif abs(own - usual - 60) <= MATCH_MIN:
            first[(site, day)] = start - timedelta(minutes=60)
            counts["evenings with 1 file: file 2"] += 1
        else:
            counts["evenings left out: 1 file off schedule"] += 1
            continue
        kept[file_id] = (site, day)
    return first, kept, counts


def main():
    """CLI entry point: match positions, check, build the page."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--config", default="config/project.yaml")
    parser.add_argument("--dry-run", action="store_true", help="counts and checks only, write nothing")
    args = parser.parse_args()
    cfg = load(args.config)
    if "viz" not in cfg:
        raise SystemExit(f"no viz section in {args.config}")
    viz_cfg = cfg["viz"]

    points = read_positions(viz_cfg)
    surveyed = sorted({row["site"] for row in read_dicts(Path(cfg["drive_detection"]["dir"]) / "file_summary.csv")})
    all_sites = [row["site"] for row in read_dicts(Path(cfg["survey"]["dir"]) / "sites.csv")]
    matched, missing, unused = match_positions(all_sites, points)
    print(f"positions: {len(points)} points in {len(viz_cfg['sources'])} GeoPackages; "
          f"{len(all_sites) - len(missing)}/{len(all_sites)} sites of sites.csv matched; {unused} points unused")
    missing_surveyed = [site for site in missing if site in surveyed]
    if missing:
        print(f"warning: {len(missing)} sites without a position ({len(missing_surveyed)} with full-survey results): "
              + ", ".join(missing))
    depts_text = Path(viz_cfg["departements"]).read_text(encoding="utf-8")
    data = build_data(cfg, matched, viz_cfg.get("round_coords"), json.loads(depts_text))
    events_per_class = dict(zip(data["classes"], [sum(site["events"][i] for site in data["sites"])
                                                  for i in range(len(data["classes"]))], strict=True))
    print(f"page data: {len(data['sites'])} sites ({', '.join(data['meta']['drives'])}), "
          f"{data['meta']['hours']:.0f} h, {len(data['evenings'])} site-evenings, events {events_per_class}, "
          f"{len(data['events'])} with a sunset time")
    status = Counter((data["classes"][i], site["status"][i]) for site in data["sites"] for i in range(len(data["classes"])))
    print("site status: " + ", ".join(f"{label} {state} {n}" for (label, state), n in sorted(status.items())))
    if args.dry_run:
        print(f"nothing written (--dry-run); the page would be {viz_cfg['out']}")
        return

    template = Path(viz_cfg["template"]).read_text(encoding="utf-8")
    page = (template.replace("/*__D3__*/", Path(viz_cfg["d3"]).read_text(encoding="utf-8"))
            .replace("/*__D3TILE__*/", Path(viz_cfg["d3_tile"]).read_text(encoding="utf-8"))
            .replace("/*__DEPTS__*/", depts_text)
            .replace("/*__DATA__*/", json.dumps(data, separators=(",", ":"), ensure_ascii=False)))
    out = Path(viz_cfg["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB): open it in a browser; keep it on this computer")


if __name__ == "__main__":
    main()

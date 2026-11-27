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

What the page holds: every event of every class at drive_detection.min_conf (0.5),
rebuilt from the raw results with the class floors ignored, so the viewer picks the
threshold per class (0.5 / 0.7 / 0.9, the audit bins; default = the survey thresholds,
min_event_conf) and every view is recomputed in the page. Per site: drive, split,
département, hours recorded, presence status per class from the verdicts (results/
survey/presence_full_gen<K>.csv; classes not in review.presence_labels are "not
checked"). Per site x evening: hours recorded. Per event: class, date, max_conf,
minutes from sunset. The audit precision per class x bin for train and heldout sites
(results/survey/audit_full_gen<K>_{train,heldout}.csv, audit_events.py report --split),
so the page shows the precision of the chosen threshold (heldout first).

Audio examples: one per site x class with events, EXAMPLE_S (8) s from 1 s before the
event, 16 kHz mono, peak-normalised MP3 embedded in the page (decoded only when played;
the page also draws its spectrogram). Chosen from the snippets of every generation's
audit and presence queues: an event judged correct ("verified") first, then at or above
the survey threshold, then the current generation, then the highest score. The MP3s are
cached next to the page (audio_gen<K>/). The audio holds no position.

The page works on phones: filters in a drawer, a sticky section bar, tap tooltips,
+/- buttons to zoom the charts, a presence table with a fixed name column.

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
    <viz.out>                 e.g. results/survey/viz/full_gen3.html (gitignored)
    <viz.out dir>/audio_gen<K>/<item_id>.mp3   audio example cache (gitignored)

Run from code/.
"""
import argparse
import base64
import csv
import json
import re
import sqlite3
import struct
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_events import generation_queues
from detect import post
from detect_drive import KEYS
from review import load_judgments

from common.config import load

TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
NUMBER_IN_SITE = re.compile(r"SMMICRO_(\d+)", re.IGNORECASE)
MIN_EVENING_HOUR = 12           # files starting earlier are set-up tests, not evening recordings
MIN_FILE_S = 1800               # shorter files are tests
NEIGHBOUR_DAYS = 3              # single-file evenings: compared with the evenings this close
MATCH_MIN = 20                  # ...whose first file starts within this many minutes
EXAMPLE_S = 8.0                 # audio example length, from 1 s before the event
EXAMPLE_RATE = 16000            # Hz: keeps sound up to 8 kHz (drumming, song, calls)


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


def build_data(cfg, matched, round_coords, depts, dry_run=False):
    """Everything the page shows, as one JSON-ready dict.

    Args:
        cfg (dict): The whole project config.
        matched (dict[str, dict]): site -> position.
        round_coords (float | None): Round positions to this many degrees (None: exact).
        depts (dict): Département outlines (GeoJSON, "code" and "nom" properties).
        dry_run (bool): Choose the audio examples but encode none.

    Returns:
        dict: meta, classes, sites, evenings, events, examples (see the module docstring).
    """
    detection_cfg, viz_cfg = cfg["drive_detection"], cfg["viz"]
    out_dir = Path(detection_cfg["dir"])
    gen = detection_cfg["gen"]
    labels = [line.strip() for line in (Path(cfg["generations"]["dir"]) / f"gen_{gen}" / "model_Labels.txt")
              .read_text(encoding="utf-8").splitlines() if line.strip()]
    order = [label for label in ("Drumming", "Song", "Call") if label in labels] + \
            [label for label in labels if label not in ("Drumming", "Song", "Call", "Noise")]
    sites_csv = {row["site"]: row for row in read_dicts(Path(cfg["survey"]["dir"]) / "sites.csv")}
    presence = presence_status(cfg)
    presence_labels = set(cfg["review"].get("presence_labels", []))
    floors = {label: floor_of(cfg, label) for label in order}

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
    events = [event for event in base_events(cfg, files, done, labels) if event["label"] in order]

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
        statuses = [presence.get((site, label), "not judged") if label in presence_labels else "not checked"
                    for label in order]                      # "none" is decided in the page, per threshold
        site_index[site] = len(site_rows)
        site_rows.append({"site": site, "short": short_name(site), "drive": info.get("drive", ""),
                          "split": info.get("split", ""), "dept": dept,
                          "dept_name": f"{dept} {dept_names.get(dept, '')}".strip() if dept else "unknown",
                          "lat": rounded(matched[site]["lat"]), "lon": rounded(matched[site]["lon"]),
                          "hours": round(hours[site], 2), "status": statuses})

    evenings = [[site_index[site], day, round(duration_h, 3)]
                for (site, day), duration_h in sorted(evening_hours.items()) if site in site_index]

    schedule_start = viz_cfg["schedule_start_min"]
    event_rows, left_out, beyond = [], 0, 0
    minutes_of = defaultdict(list)
    for event in events:
        if event["site"] not in site_index:
            continue
        evening = kept_files.get(event["file_id"])
        minutes = None
        if evening is None or evening not in first_start:
            left_out += 1
        else:
            minutes = round((event["event_time"] - first_start[evening]).total_seconds() / 60 + schedule_start, 1)
            beyond += not schedule_start <= minutes < schedule_start + 120
            if event["max_conf"] >= floors[event["label"]]:
                minutes_of[event["label"]].append(minutes)
        event_rows.append([site_index[event["site"]], order.index(event["label"]), event["start"][:10],
                           minutes, round(event["max_conf"], 4)])
    if left_out:
        print(f"note: {left_out} events in files left out above: no time from sunset")
    if beyond:
        print(f"note: {beyond} events outside the 2 h after the evening's first file (extra files)")
    print("median minutes from sunset (survey thresholds): " + ", ".join(
        f"{label} {sorted(values)[len(values) // 2]:.0f}" for label, values in sorted(minutes_of.items())))
    at_floor = Counter(event["label"] for event in events if event["max_conf"] >= floors[event["label"]])
    print(f"events at {detection_cfg['min_conf']} (page data): {dict(Counter(event['label'] for event in events))}; "
          f"at the survey thresholds {floors}: {dict(at_floor)}")

    wanted = {(event["site"], event["label"]) for event in events if event["site"] in site_index}
    audio_dir = Path(viz_cfg["out"]).with_name(f"audio_gen{gen}")
    examples = audio_examples(cfg, wanted, audio_dir, dry_run)
    example_rows = [[site_index[site], order.index(label), ex["verified"], ex["gen"], ex["date"], ex["time"],
                     ex["score"], ex["audio"]] for (site, label), ex in sorted(examples.items())]
    n_verified = sum(ex["verified"] for ex in examples.values())
    size = sum(len(ex["audio"]) for ex in examples.values()) / 1e6
    print(f"audio examples: {len(examples)} of {len(wanted)} site x class with events at {detection_cfg['min_conf']} "
          f"({n_verified} verified by a verdict, {len(examples) - n_verified} not)"
          + ("" if dry_run else f", {size:.1f} MB in the page"))

    choices = sorted(set(cfg["review"]["audit_bins"][:-1]) | set(floors.values()))
    return {"meta": {"gen": gen, "built": datetime.now().strftime("%Y-%m-%d %H:%M"),
                     "drives": sorted({row["drive"] for row in site_rows}),
                     "hours": round(sum(row["hours"] for row in site_rows), 1),
                     "base": detection_cfg["min_conf"], "floors": floors, "choices": choices,
                     "schedule_start_min": schedule_start, "rounded": round_coords,
                     "audit": audit_bins(cfg), "presence_labels": sorted(presence_labels),
                     "example_s": EXAMPLE_S},
            "classes": order, "sites": site_rows, "evenings": evenings, "events": event_rows,
            "examples": example_rows}


def audit_bins(cfg):
    """Audited precision per class x max_conf bin, for train and heldout sites.

    Args:
        cfg (dict): The whole project config.

    Returns:
        dict: split -> label -> list of {lo, hi, population, judged, precision}
            (precision None when the bin has no verdict), from results/survey/
            audit_full_gen<K>_<split>.csv (audit_events.py report --split ...);
            "survey": split -> label -> {precision, low, high, judged, scope} at the
            survey thresholds (kept row, else class row).
    """
    gen = cfg["drive_detection"]["gen"]
    out = {"train": {}, "heldout": {}, "survey": {"train": {}, "heldout": {}}}
    for split in ("train", "heldout"):
        path = Path("results/survey") / f"audit_full_gen{gen}_{split}.csv"
        if not path.exists():
            print(f"note: {path} not found (audit_events.py report --split {split}): no precision for {split} sites")
            continue
        for row in read_dicts(path):
            value = None if row["precision"] in ("nan", "") else float(row["precision"])
            if row["scope"] == "class|bin":
                lo, hi = (float(edge) for edge in row["bin"].split("-"))
                out[split].setdefault(row["label"], []).append(
                    {"lo": lo, "hi": hi, "population": int(row["population"]), "judged": int(row["judged"]),
                     "precision": value})
            elif row["scope"] in ("kept", "class") and value is not None:
                if row["scope"] == "class" and out["survey"][split].get(row["label"], {}).get("scope") == "kept":
                    continue
                out["survey"][split][row["label"]] = {"precision": value, "low": float(row["ci_low"]),
                                                      "high": float(row["ci_high"]), "judged": int(row["judged"]),
                                                      "scope": row["scope"]}
    return out


def base_events(cfg, files, done, labels):
    """Events of every class at the base floor (drive_detection.min_conf), class floors ignored.

    The page applies the class thresholds itself, so a viewer can try others.

    Args:
        cfg (dict): The whole project config.
        files (dict[str, dict]): file_id -> files/<drive>_files.csv row.
        done (set[str]): file_ids with BirdNET results.
        labels (list[str]): Model classes.

    Returns:
        list[dict]: file_id, site, start, label, event_time (datetime), max_conf.
    """
    detection_cfg = cfg["drive_detection"]
    rows = [files[file_id] for file_id in sorted(done)]
    raw_events, _summary = post(rows, Path(detection_cfg["dir"]) / "raw", {**detection_cfg, "min_event_conf": {}},
                                labels, keys=KEYS)
    events = []
    for file_id, site, _split, _drive, start, _duration, label, start_s, _end_s, _n, max_conf, _mean in raw_events:
        events.append({"file_id": file_id, "site": site, "start": start, "label": label,
                       "event_time": datetime.strptime(start, TIME_FORMAT) + timedelta(seconds=float(start_s)),
                       "max_conf": float(max_conf)})
    return events


def floor_of(cfg, label):
    """Survey threshold of a class: max(min_conf, min_event_conf of the class).

    Args:
        cfg (dict): The whole project config.
        label (str): Class name.

    Returns:
        float: The threshold.
    """
    detection_cfg = cfg["drive_detection"]
    return max(detection_cfg["min_conf"], (detection_cfg.get("min_event_conf") or {}).get(label, 0.0))


def audio_examples(cfg, wanted, audio_dir, dry_run):
    """One short MP3 per site x class: the best verified event, else the best unverified one.

    Candidates are the events of every generation's audit and presence queues that have
    a snippet and were not judged as something else. Preference: judged correct, then at
    or above the survey threshold, then from the current generation, then the highest
    score. The clip is EXAMPLE_S seconds from 1 s before the event, resampled to
    EXAMPLE_RATE Hz mono, peak-normalised, saved as MP3 in audio_dir (kept and reused).

    Args:
        cfg (dict): The whole project config.
        wanted (set[tuple[str, str]]): (site, label) pairs with events.
        audio_dir (Path): Cache folder of the MP3s.
        dry_run (bool): Choose only, encode nothing.

    Returns:
        dict[tuple[str, str], dict]: (site, label) -> verified, gen, date, time, score,
            audio (base64 MP3, "" in a dry run).
    """
    gen_now = cfg["drive_detection"]["gen"]
    best = {}
    for _name, gen, queue_path, judgments_path in generation_queues(cfg):
        verdicts = load_judgments(judgments_path)
        for row in read_dicts(queue_path):
            key = (row["site"], row["predicted"])
            if key not in wanted or not Path(row["clip_path"]).exists():
                continue
            verdict = verdicts.get(row["item_id"])
            if verdict not in (None, row["predicted"]):
                continue                                    # judged as something else (or Unsure)
            score = float(row["max_conf"])
            rank = (verdict == row["predicted"], score >= floor_of(cfg, row["predicted"]), gen == gen_now, score)
            if key not in best or rank > best[key]["rank"]:
                best[key] = {"rank": rank, "row": row, "gen": gen, "verified": verdict == row["predicted"]}
    examples = {}
    for key, choice in best.items():
        row = choice["row"]
        offset = float(row["offset_s"])
        stamp = re.search(r"(\d{8}_\d{6})$", row["clip_id"])
        when = (datetime.strptime(stamp.group(1), "%Y%m%d_%H%M%S") + timedelta(seconds=offset + float(row["start_s"]))
                if stamp else None)
        example = {"verified": choice["verified"], "gen": choice["gen"], "score": round(float(row["max_conf"]), 3),
                   "date": when.strftime("%Y-%m-%d") if when else "", "time": when.strftime("%H:%M") if when else "",
                   "audio": ""}
        if not dry_run:
            mp3 = audio_dir / f"{row['item_id']}.mp3"
            if not mp3.exists():
                audio_dir.mkdir(parents=True, exist_ok=True)
                info = sf.info(row["clip_path"])
                first = max(0.0, float(row["start_s"]) - 1.0)
                audio, rate = sf.read(row["clip_path"], start=int(first * info.samplerate),
                                      frames=int(EXAMPLE_S * info.samplerate), dtype="float32", always_2d=True)
                audio = audio.mean(axis=1)
                divisor = gcd(EXAMPLE_RATE, rate)
                audio = resample_poly(audio, EXAMPLE_RATE // divisor, rate // divisor).astype("float32")
                audio = audio / max(1e-9, float(np.abs(audio).max())) * 0.89
                sf.write(mp3, audio, EXAMPLE_RATE, format="MP3", subtype="MPEG_LAYER_III", compression_level=0.5)
            example["audio"] = base64.b64encode(mp3.read_bytes()).decode("ascii")
        examples[key] = example
    return examples


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
    data = build_data(cfg, matched, viz_cfg.get("round_coords"), json.loads(depts_text), args.dry_run)
    print(f"page data: {len(data['sites'])} sites ({', '.join(data['meta']['drives'])}), "
          f"{data['meta']['hours']:.0f} h, {len(data['evenings'])} site-evenings, {len(data['events'])} events "
          f"at {data['meta']['base']} (thresholds applied in the page)")
    status = Counter((data["classes"][i], site["status"][i]) for site in data["sites"] for i in range(len(data["classes"])))
    print("site status (verdicts): " + ", ".join(f"{label} {state} {n}" for (label, state), n in sorted(status.items())))
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

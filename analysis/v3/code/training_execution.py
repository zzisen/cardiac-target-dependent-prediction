#!/usr/bin/env python3
"""Frozen LUNA14 B1 execution for the 352 primary DDPR FCS files.

Usage:
  python main_execution.py --source-wrapper PATH [--work-dir DIR]
  python main_execution.py --prepare-registry --project-root PROJECT_ROOT

The script reads only the 352 paths in B1_PRIMARY_FILE_REGISTRY.csv and only
the eight B13R2 primary channels from each FCS DATA segment. It never exports
raw event values or retains a full event matrix.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import struct
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np


CHANNELS = [
    "pS6", "p4EBP1", "pSTAT5", "pSYK",
    "pPLC\u03b32", "pAKT", "pERK1/2", "pIKAROS",
]
TARGETS = [
    "BCR-Crosslink", "BEZ-235", "Dasatinib", "IL-7",
    "Pervanadate", "TSLP", "Tofacitinib",
]
CONDITIONS = ["Basal"] + TARGETS
THRESHOLD = 10.0
EXPECTED_OUTER_SHA256 = "642d88e9f7920d1eb005c6172c2446958e52c3e341cc6efa7cec557a8fb0beff"
EXPECTED_INNER_BYTES = 5863234104
EXPECTED_INNER_SHA256 = "72b465bc92e6d2cb9ce94c54c3bcda46b162028ff0eb496fd29f6f441b940fa7"
EXPECTED_PATIENTS = 44
EXPECTED_FILES = 352
EXPECTED_CHANNELS = 8
EXPECTED_SUMMARIES = 2816
INNER_MEMBERS = 1050
FCS_MEMBERS = 525
BLOCK_EVENTS = 32768
BOOTSTRAP_SEED = 20261006
BOOTSTRAP_REPLICATES = 50000


class SourceDecodeError(RuntimeError):
    pass


class BoundedReader:
    """Seekable view of a stored ZIP member without extracting the 5.8 GB file."""
    def __init__(self, base: io.BufferedReader, start: int, length: int):
        self.base = base
        self.start = int(start)
        self.length = int(length)
        self.position = 0

    def read(self, n: int = -1) -> bytes:
        remaining = self.length - self.position
        if remaining <= 0:
            return b""
        if n is None or n < 0 or n > remaining:
            n = remaining
        self.base.seek(self.start + self.position)
        data = self.base.read(n)
        self.position += len(data)
        return data

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            new_position = offset
        elif whence == 1:
            new_position = self.position + offset
        elif whence == 2:
            new_position = self.length + offset
        else:
            raise ValueError("invalid whence")
        if new_position < 0 or new_position > self.length:
            raise ValueError("seek outside bounded member")
        self.position = int(new_position)
        return self.position

    def tell(self) -> int:
        return self.position

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def close(self) -> None:
        return None


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def csv_read(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def csv_write(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def json_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha_text(value: Any) -> str:
    return str(value).strip().lower()


def exact_read(f: Any, n: int, where: str) -> bytes:
    chunks: list[bytes] = []
    remaining = n
    while remaining:
        part = f.read(remaining)
        if not part:
            raise SourceDecodeError(f"short read at {where}: needed {n} bytes")
        chunks.append(part)
        remaining -= len(part)
    return b"".join(chunks)


def parse_text_segment(text: bytes) -> dict[str, str]:
    if not text:
        raise SourceDecodeError("empty FCS TEXT segment")
    # The frozen header parser records a terminal NUL after the final delimiter.
    text = text.rstrip(b"\x00")
    delimiter = text[0]
    if delimiter == 0:
        raise SourceDecodeError("NUL FCS TEXT delimiter")
    fields: list[bytes] = []
    token = bytearray()
    i = 1
    while i < len(text):
        if text[i] == delimiter:
            if i + 1 < len(text) and text[i + 1] == delimiter:
                token.append(delimiter)
                i += 2
            else:
                fields.append(bytes(token))
                token.clear()
                i += 1
        else:
            token.append(text[i])
            i += 1
    if token:
        fields.append(bytes(token))
    if fields and fields[-1] == b"":
        fields.pop()
    if len(fields) % 2:
        raise SourceDecodeError(f"odd FCS TEXT token count: {len(fields)}")
    result: dict[str, str] = {}
    for j in range(0, len(fields), 2):
        key = fields[j].decode("latin-1").strip().upper()
        value = fields[j + 1].decode("latin-1").strip()
        result[key] = value
    return result


def fcs_int(kv: dict[str, str], key: str, default: int | None = None) -> int:
    value = kv.get(key)
    if value is None or value == "":
        if default is not None:
            return default
        raise SourceDecodeError(f"missing FCS keyword {key}")
    try:
        return int(value)
    except ValueError as e:
        raise SourceDecodeError(f"invalid integer FCS keyword {key}={value!r}") from e


def parse_fixed_header(header: bytes) -> dict[str, int | str]:
    if len(header) != 58:
        raise SourceDecodeError("FCS fixed header is not 58 bytes")
    version = header[:6].decode("ascii", errors="replace")
    if not version.startswith("FCS"):
        raise SourceDecodeError(f"unexpected FCS version marker: {version!r}")
    fields: dict[str, int | str] = {"version": version}
    for name, a, b in [
        ("text_begin_header", 10, 18),
        ("text_end_header", 18, 26),
        ("data_begin_header", 26, 34),
        ("data_end_header", 34, 42),
    ]:
        value = header[a:b].decode("ascii", errors="strict").strip()
        fields[name] = int(value) if value else 0
    return fields


def make_inner_zip(source_wrapper: Path) -> tuple[zipfile.ZipFile, Any, io.BufferedReader]:
    outer = zipfile.ZipFile(source_wrapper, "r")
    try:
        info = outer.getinfo("ddpr_data.zip")
    except KeyError as e:
        outer.close()
        raise SourceDecodeError("canonical ddpr_data.zip member is missing") from e
    if info.file_size != EXPECTED_INNER_BYTES or info.compress_size != EXPECTED_INNER_BYTES:
        outer.close()
        raise SourceDecodeError("canonical inner archive size or storage method changed")
    if info.compress_type != zipfile.ZIP_STORED:
        outer.close()
        raise SourceDecodeError("canonical inner archive is not stored as frozen")
    raw = source_wrapper.open("rb")
    raw.seek(info.header_offset)
    local = exact_read(raw, 30, "outer local ZIP header")
    if local[:4] != b"PK\x03\x04":
        raw.close()
        outer.close()
        raise SourceDecodeError("invalid outer local ZIP header")
    filename_length, extra_length = struct.unpack_from("<HH", local, 26)
    data_offset = info.header_offset + 30 + filename_length + extra_length
    inner_view = BoundedReader(raw, data_offset, info.file_size)
    inner = zipfile.ZipFile(inner_view, "r")
    return outer, inner, raw


def header_json_map(value: str) -> list[dict[str, Any]]:
    try:
        out = json.loads(value)
    except Exception as e:
        raise SourceDecodeError("invalid frozen parameter_mapping_json") from e
    if not isinstance(out, list):
        raise SourceDecodeError("frozen parameter map is not a list")
    return out


def read_channel_registry(path: Path) -> tuple[list[str], dict[str, dict[str, str]]]:
    rows = csv_read(path)
    primary = [r for r in rows if r["decision_status"] == "PRIMARY_8_CHANNEL"]
    if [r["canonical_channel"] for r in primary] != CHANNELS:
        raise SourceDecodeError("B13R2 channel registry differs from the frozen 8-channel order")
    return CHANNELS, {r["canonical_channel"]: r for r in primary}


def prepare_registry(project_root: Path, work_dir: Path) -> None:
    amendment_path = project_root / "P2_V3_ABT" / "B13R2_LOCAL_RECONCILIATION_2026-10-06" / "B1_FINAL_PREOUTCOME_AMENDMENT.json"
    master_path = project_root / "P2_V3_ABT" / "B_FULL_ARCHIVE_AUDIT_LUNA12_2026-10-06" / "B_FULL_ARCHIVE_FCS_MASTER_INVENTORY.csv"
    channel_path = project_root / "P2_V3_ABT" / "B13R2_LOCAL_RECONCILIATION_2026-10-06" / "B13R2_FINAL_CHANNEL_REGISTRY.csv"
    amendment = json.loads(amendment_path.read_text(encoding="utf-8-sig"))
    patient_order = amendment["cohort"]["primary_patient_ids"]
    targets = amendment["conditions"]["targets"]
    if patient_order and len(patient_order) != EXPECTED_PATIENTS:
        raise SourceDecodeError("frozen cohort is not 44 patients")
    _, channel_map = read_channel_registry(channel_path)
    master = csv_read(master_path)
    patient_set = set(patient_order)
    condition_set = set(["Basal"] + targets)
    selected = [
        r for r in master
        if r["sample_id"] in patient_set
        and r["condition"] in condition_set
        and r["sample_class"] == "diagnostic_UPN_sample_label"
    ]
    if len(selected) != EXPECTED_FILES:
        raise SourceDecodeError(f"frozen registry selected {len(selected)} files, expected {EXPECTED_FILES}")
    patient_rank = {patient: i for i, patient in enumerate(patient_order, 1)}
    condition_rank = {condition: i for i, condition in enumerate(["Basal"] + targets, 0)}
    out: list[dict[str, Any]] = []
    pair_counts: dict[tuple[str, str], int] = {}
    for r in selected:
        filename = r["filename"]
        if filename != r["archive_member_path"] or not filename.lower().endswith(".fcs"):
            raise SourceDecodeError(f"unexpected FCS member name: {filename}")
        match = re.fullmatch(r"(UPN\d+)_(.+)\.fcs", filename)
        if not match or match.group(1) != r["sample_id"] or match.group(2) != r["condition"]:
            raise SourceDecodeError(f"filename/sample/condition mismatch: {filename}")
        params = header_json_map(r["parameter_mapping_json"])
        pcreb = [p for p in params if str(p.get("PnS", "")).strip().lower() == "pcreb"]
        if len(pcreb) != 1:
            raise SourceDecodeError(f"{filename}: pCREB panel marker is not unique in frozen header map")
        isotope = str(pcreb[0]["PnN"]).lower()
        panel_group = "Lu176_panel" if "lu176" in isotope else "Yb176_group" if "yb176" in isotope else ""
        if not panel_group:
            raise SourceDecodeError(f"{filename}: unsupported frozen pCREB panel signature")
        index_map: dict[str, int] = {}
        pnn_map: dict[str, str] = {}
        pns_map: dict[str, str] = {}
        for channel in CHANNELS:
            registry = channel_map[channel]
            expected_pnn = registry[panel_group + "_exact_PnN"]
            expected_pns = registry[panel_group + "_exact_PnS"]
            found = [
                p for p in params
                if p.get("PnN") == expected_pnn and p.get("PnS") == expected_pns
            ]
            if len(found) != 1:
                raise SourceDecodeError(
                    f"{filename}: {channel} registry mapping resolves to {len(found)} parameters"
                )
            index = int(found[0]["parameter_number"])
            if int(found[0]["PnB"]) != 32 or found[0]["PnE"] != "0,0":
                raise SourceDecodeError(f"{filename}: {channel} is not a linear float32 parameter")
            index_map[channel] = index
            pnn_map[channel] = expected_pnn
            pns_map[channel] = expected_pns
        if len(set(index_map.values())) != EXPECTED_CHANNELS:
            raise SourceDecodeError(f"{filename}: primary channel indices are not unique")
        tot = int(r["TOT"])
        par = int(r["PAR"])
        if int(r["data_end"]) - int(r["data_begin"]) != tot * par * 4:
            raise SourceDecodeError(f"{filename}: header inventory DATA size != $TOT*$PAR*4")
        if r["DATATYPE"] != "F" or r["BYTEORD"] != "4,3,2,1" or r["MODE"] != "L":
            raise SourceDecodeError(f"{filename}: unsupported frozen FCS encoding")
        key = (r["sample_id"], r["condition"])
        pair_counts[key] = pair_counts.get(key, 0) + 1
        out.append({
            "patient_order": patient_rank[r["sample_id"]],
            "patient_id": r["sample_id"],
            "sample_class": r["sample_class"],
            "condition_order": condition_rank[r["condition"]],
            "condition": r["condition"],
            "target_or_baseline": "baseline" if r["condition"] == "Basal" else "target",
            "filename": filename,
            "archive_member_path": r["archive_member_path"],
            "source_member_sha256": sha_text(r["sha256"]),
            "source_fcs_bytes": int(r["byte_size"]),
            "expected_event_count": tot,
            "parameter_count": par,
            "datatype": r["DATATYPE"],
            "byte_order": r["BYTEORD"],
            "mode": r["MODE"],
            "text_begin": r["text_begin"],
            "text_end": r["text_end"],
            "data_begin": r["data_begin"],
            "data_end": r["data_end"],
            "panel_signature_sha256": r["panel_signature_sha256"],
            "panel_registry_group": panel_group,
            "channel_parameter_indices_json": json_compact(index_map),
            "channel_parameter_pnn_json": json_compact(pnn_map),
            "channel_parameter_pns_json": json_compact(pns_map),
        })
    expected_pairs = {(p, c) for p in patient_order for c in ["Basal"] + targets}
    if set(pair_counts) != expected_pairs or any(v != 1 for v in pair_counts.values()):
        raise SourceDecodeError("frozen primary patient-condition support is incomplete or duplicated")
    out.sort(key=lambda r: (int(r["patient_order"]), int(r["condition_order"])))
    fields = list(out[0])
    csv_write(work_dir / "B1_PRIMARY_FILE_REGISTRY.csv", fields, out)
    (work_dir / "B13R2_FINAL_CHANNEL_REGISTRY.csv").write_bytes(channel_path.read_bytes())


def decode_one_fcs(
    inner: zipfile.ZipFile,
    registry_row: dict[str, str],
    channel_map: dict[str, dict[str, str]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    member = registry_row["archive_member_path"]
    if inner.namelist().count(member) != 1:
        raise SourceDecodeError(f"{member}: archive member is absent or duplicated")
    info = inner.getinfo(member)
    if info.file_size != int(registry_row["source_fcs_bytes"]):
        raise SourceDecodeError(f"{member}: archive member byte size changed")
    expected_hash = sha_text(registry_row["source_member_sha256"])
    parameter_indices = json.loads(registry_row["channel_parameter_indices_json"])
    expected_pnn = json.loads(registry_row["channel_parameter_pnn_json"])
    expected_pns = json.loads(registry_row["channel_parameter_pns_json"])
    digest = hashlib.sha256()
    pos_counts = {c: 0 for c in CHANNELS}
    nonfinite_counts = {c: 0 for c in CHANNELS}
    decoded_events = 0
    trailing_bytes = 0
    warnings: list[str] = []

    with inner.open(info, "r") as f:
        fixed = exact_read(f, 58, member + " fixed header")
        digest.update(fixed)
        fh = parse_fixed_header(fixed)
        text_begin = int(fh["text_begin_header"])
        text_end = int(fh["text_end_header"])
        if text_begin < 58 or text_end < text_begin:
            raise SourceDecodeError(f"{member}: invalid fixed-header TEXT offsets")
        before_text = text_begin - 58
        gap = exact_read(f, before_text, member + " pre-TEXT gap") if before_text else b""
        digest.update(gap)
        text_bytes = exact_read(f, text_end - text_begin + 1, member + " TEXT segment")
        digest.update(text_bytes)
        kv = parse_text_segment(text_bytes)
        tot = fcs_int(kv, "$TOT")
        par = fcs_int(kv, "$PAR")
        data_begin = fcs_int(kv, "$BEGINDATA", int(fh["data_begin_header"]))
        data_end = fcs_int(kv, "$ENDDATA", int(fh["data_end_header"]))
        if int(fh["data_begin_header"]) not in (0, data_begin):
            raise SourceDecodeError(f"{member}: fixed/TEXT DATA start mismatch")
        if int(fh["data_end_header"]) not in (0, data_end):
            raise SourceDecodeError(f"{member}: fixed/TEXT DATA end mismatch")
        if tot != int(registry_row["expected_event_count"]):
            raise SourceDecodeError(f"{member}: $TOT differs from B13R2/Luna12 header registry")
        if par != int(registry_row["parameter_count"]):
            raise SourceDecodeError(f"{member}: $PAR differs from B13R2/Luna12 header registry")
        if kv.get("$DATATYPE") != "F" or kv.get("$MODE") != "L" or kv.get("$BYTEORD") != "4,3,2,1":
            raise SourceDecodeError(f"{member}: FCS encoding differs from frozen header registry")
        if data_begin < f.tell() or data_end < data_begin:
            raise SourceDecodeError(f"{member}: invalid DATA offsets")
        params: list[dict[str, str]] = []
        for i in range(1, par + 1):
            prefix = f"$P{i}"
            params.append({
                "PnN": kv.get(prefix + "N", ""),
                "PnS": kv.get(prefix + "S", ""),
                "PnB": kv.get(prefix + "B", ""),
                "PnE": kv.get(prefix + "E", ""),
            })
        if len(params) != par:
            raise SourceDecodeError(f"{member}: incomplete parameter metadata")
        actual_pcreb = [
            p for p in params if p["PnS"].strip().lower() == "pcreb"
        ]
        if len(actual_pcreb) != 1:
            raise SourceDecodeError(f"{member}: pCREB panel identity is not unique")
        isotope = actual_pcreb[0]["PnN"].lower()
        panel_group = "Lu176_panel" if "lu176" in isotope else "Yb176_group" if "yb176" in isotope else ""
        if panel_group != registry_row["panel_registry_group"]:
            raise SourceDecodeError(f"{member}: actual panel group differs from frozen registry")
        actual_indices: dict[str, int] = {}
        for channel in CHANNELS:
            candidates = [
                (i + 1, p) for i, p in enumerate(params)
                if p["PnN"] == expected_pnn[channel] and p["PnS"] == expected_pns[channel]
            ]
            if len(candidates) != 1:
                raise SourceDecodeError(f"{member}: {channel} actual parameter map is not unique")
            index, p = candidates[0]
            if index != int(parameter_indices[channel]):
                raise SourceDecodeError(f"{member}: {channel} parameter index differs from frozen registry")
            if int(p["PnB"]) != 32 or p["PnE"] != "0,0":
                raise SourceDecodeError(f"{member}: {channel} encoding is not linear float32")
            actual_indices[channel] = index
        if actual_indices != {k: int(v) for k, v in parameter_indices.items()}:
            raise SourceDecodeError(f"{member}: actual channel index map differs")
        all_bits = [int(kv.get(f"$P{i}B", "0")) for i in range(1, par + 1)]
        if any(bits != 32 for bits in all_bits):
            raise SourceDecodeError(f"{member}: non-32-bit parameters prevent exact record-width decoding")
        event_bytes = par * 4
        expected_data_bytes = tot * event_bytes
        if data_end - data_begin != expected_data_bytes:
            raise SourceDecodeError(
                f"{member}: DATA bytes ({data_end-data_begin}) != $TOT*$PAR*4 ({expected_data_bytes})"
            )
        post_text_gap = data_begin - f.tell()
        if post_text_gap < 0:
            raise SourceDecodeError(f"{member}: DATA starts before TEXT ends")
        if post_text_gap:
            skipped = exact_read(f, post_text_gap, member + " pre-DATA gap")
            digest.update(skipped)

        dtype = np.dtype(">f4")
        offsets = {c: (int(parameter_indices[c]) - 1) * 4 for c in CHANNELS}
        remaining = tot
        while remaining:
            n = min(BLOCK_EVENTS, remaining)
            block = exact_read(f, n * event_bytes, member + " DATA block")
            digest.update(block)
            for channel in CHANNELS:
                vals = np.ndarray(
                    shape=(n,), dtype=dtype, buffer=block,
                    offset=offsets[channel], strides=(event_bytes,),
                )
                finite = np.isfinite(vals)
                if not finite.all():
                    nonfinite_counts[channel] += int(n - np.count_nonzero(finite))
                pos_counts[channel] += int(np.count_nonzero(vals >= THRESHOLD))
            decoded_events += n
            remaining -= n
        if f.tell() != data_end:
            raise SourceDecodeError(f"{member}: decoded DATA cursor does not equal $ENDDATA")
        while True:
            tail = f.read(1024 * 1024)
            if not tail:
                break
            trailing_bytes += len(tail)
            digest.update(tail)
        # Reading through EOF also asks zipfile to check the member CRC.

    actual_hash = digest.hexdigest()
    if actual_hash != expected_hash:
        raise SourceDecodeError(f"{member}: FCS member SHA-256 differs from structural audit")
    if decoded_events != tot:
        raise SourceDecodeError(f"{member}: decoded {decoded_events} events; $TOT={tot}")
    if trailing_bytes:
        warnings.append(f"FCS_TRAILING_BYTES_AFTER_DATA={trailing_bytes}")
    if any(nonfinite_counts.values()):
        raise SourceDecodeError(f"{member}: non-finite primary event values {nonfinite_counts}")
    if any(pos_counts[c] < 0 or pos_counts[c] > decoded_events for c in CHANNELS):
        raise SourceDecodeError(f"{member}: positive count outside [0,N]")
    fractions = {c: pos_counts[c] / decoded_events for c in CHANNELS}
    if any(not math.isfinite(fractions[c]) or not 0.0 <= fractions[c] <= 1.0 for c in CHANNELS):
        raise SourceDecodeError(f"{member}: positive fraction is non-finite or outside [0,1]")

    summary_rows = []
    for channel in CHANNELS:
        summary_rows.append({
            "patient_id": registry_row["patient_id"],
            "condition": registry_row["condition"],
            "target_or_baseline": registry_row["target_or_baseline"],
            "channel": channel,
            "fcs_member_path": member,
            "channel_parameter_index": int(parameter_indices[channel]),
            "event_count": decoded_events,
            "positive_count_ge10": pos_counts[channel],
            "positive_fraction": fractions[channel],
        })
    audit_row = {
        "patient_id": registry_row["patient_id"],
        "condition": registry_row["condition"],
        "fcs_member_path": member,
        "panel_registry_group": panel_group,
        "expected_event_count_tot": tot,
        "decoded_event_count": decoded_events,
        "parameter_count": par,
        "data_bytes_expected": expected_data_bytes,
        "data_bytes_decoded": expected_data_bytes,
        "trailing_bytes_after_data": trailing_bytes,
        "channel_parameter_indices_json": json_compact(actual_indices),
        "positive_counts_ge10_json": json_compact(pos_counts),
        "positive_fractions_json": json_compact(fractions),
        "nonfinite_primary_values_json": json_compact(nonfinite_counts),
        "expected_fcs_member_sha256": expected_hash,
        "decoded_fcs_member_sha256": actual_hash,
        "decode_warning": json_compact(warnings),
        "decode_status": "PASS",
    }
    return audit_row, summary_rows


def train_center_scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(values, axis=0, dtype=np.float64)
    scale = np.std(values, axis=0, ddof=1, dtype=np.float64)
    scale = np.where(scale == 0.0, 1.0, scale)
    return mean, scale


def predict_standardized_ols(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit separate intercept OLS models in training-standardized coordinates."""
    x_mean = float(np.mean(x_train, dtype=np.float64))
    x_sd = float(np.std(x_train, ddof=1, dtype=np.float64))
    if x_sd == 0.0:
        x_sd = 1.0
    x_z = (x_train - x_mean) / x_sd
    y_mean, y_sd = train_center_scale(y_train)
    y_z = (y_train - y_mean) / y_sd
    x_centered = x_z - float(np.mean(x_z, dtype=np.float64))
    y_centered = y_z - np.mean(y_z, axis=0, dtype=np.float64)
    denominator = float(np.dot(x_centered, x_centered))
    if denominator == 0.0:
        slope = np.zeros(y_train.shape[1], dtype=np.float64)
    else:
        slope = (x_centered[:, None] * y_centered).sum(axis=0) / denominator
    intercept = np.mean(y_z, axis=0, dtype=np.float64) - slope * float(np.mean(x_z, dtype=np.float64))
    x_test_z = (float(x_test) - x_mean) / x_sd
    prediction_z = intercept + slope * x_test_z
    prediction_raw = y_mean + y_sd * prediction_z
    return prediction_z, prediction_raw, y_mean, y_sd


def write_scientific_outputs(
    work_dir: Path,
    summary_rows: list[dict[str, Any]],
    event_audit_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    expected = EXPECTED_SUMMARIES
    key_set = {
        (r["patient_id"], r["condition"], r["channel"])
        for r in summary_rows
    }
    if len(summary_rows) != expected or len(key_set) != expected:
        raise SourceDecodeError("primary summary row count or unique-key assertion failed")
    frac = np.asarray([float(r["positive_fraction"]) for r in summary_rows], dtype=np.float64)
    if frac.size != expected or not np.isfinite(frac).all():
        raise SourceDecodeError("not all 352 x 8 primary summaries are finite")

    order_by = {p: i for i, p in enumerate(sorted({r["patient_id"] for r in summary_rows}), 0)}
    # The primary registry carries the frozen patient order; retain that order below.
    registry = csv_read(work_dir / "B1_PRIMARY_FILE_REGISTRY.csv")
    patient_order = []
    for r in registry:
        if r["patient_id"] not in patient_order:
            patient_order.append(r["patient_id"])
    if len(patient_order) != EXPECTED_PATIENTS:
        raise SourceDecodeError("summary matrix does not contain the frozen 44 patients")
    pidx = {p: i for i, p in enumerate(patient_order)}
    tidx = {t: i for i, t in enumerate(TARGETS)}
    cidx = {c: i for i, c in enumerate(CHANNELS)}
    values: dict[tuple[str, str, str], float] = {
        (r["patient_id"], r["condition"], r["channel"]): float(r["positive_fraction"])
        for r in summary_rows
    }
    x = np.empty((EXPECTED_PATIENTS, EXPECTED_CHANNELS), dtype=np.float64)
    y = np.empty((EXPECTED_PATIENTS, len(TARGETS), EXPECTED_CHANNELS), dtype=np.float64)
    for p in patient_order:
        for c in CHANNELS:
            x[pidx[p], cidx[c]] = values[(p, "Basal", c)]
        for t in TARGETS:
            for c in CHANNELS:
                y[pidx[p], tidx[t], cidx[c]] = values[(p, t, c)]
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise SourceDecodeError("model matrix contains non-finite values")

    n = EXPECTED_PATIENTS
    n_targets = len(TARGETS)
    n_channels = len(CHANNELS)
    outer_losses = np.empty((n, n_targets, 3), dtype=np.float64)
    outer_predictions: list[dict[str, Any]] = []
    inner_scores_rows: list[dict[str, Any]] = []
    shared_map_rows: list[dict[str, Any]] = []
    exact_map_rows: list[dict[str, Any]] = []
    policies = ["C0", "SHARED", "EXACT"]
    started = time.time()

    for outer_i, patient in enumerate(patient_order):
        outer_test = outer_i
        outer_train = np.asarray([i for i in range(n) if i != outer_test], dtype=int)
        if outer_test in set(outer_train.tolist()) or len(outer_train) != 43:
            raise AssertionError("outer patient isolation assertion failed")
        inner_loss_samples = np.empty((len(outer_train), n_channels, n_targets), dtype=np.float64)
        for inner_k, inner_test in enumerate(outer_train):
            inner_train = outer_train[outer_train != inner_test]
            if inner_test in set(inner_train.tolist()) or not set(inner_train.tolist()).issubset(set(outer_train.tolist())):
                raise AssertionError("inner patient isolation assertion failed")
            if len(inner_train) != 42:
                raise AssertionError("inner training count is not 42")
            for ti, target in enumerate(TARGETS):
                y_train = y[inner_train, ti, :]
                y_test = y[inner_test, ti, :]
                y_mean, y_scale = train_center_scale(y_train)
                y_test_z = (y_test - y_mean) / y_scale
                for ci in range(n_channels):
                    pred_z, _, _, _ = predict_standardized_ols(
                        x[inner_train, ci], y_train, x[inner_test, ci]
                    )
                    inner_loss_samples[inner_k, ci, ti] = float(np.mean((y_test_z - pred_z) ** 2))
        if not np.isfinite(inner_loss_samples).all():
            raise SourceDecodeError(f"{patient}: non-finite inner selection score")
        inner_mean_by_target = np.mean(inner_loss_samples, axis=0, dtype=np.float64)
        shared_mean = np.mean(inner_mean_by_target, axis=1, dtype=np.float64)
        shared_ci = int(np.argmin(shared_mean))
        exact_ci = np.argmin(inner_mean_by_target, axis=0)
        shared_channel = CHANNELS[shared_ci]
        shared_map_rows.append({
            "outer_fold": outer_i + 1,
            "held_out_patient": patient,
            "selected_shared_channel": shared_channel,
            "inner_equal_target_mean_loss": shared_mean[shared_ci],
            "candidate_order": json_compact(CHANNELS),
        })
        for ti, target in enumerate(TARGETS):
            exact_map_rows.append({
                "outer_fold": outer_i + 1,
                "held_out_patient": patient,
                "target": target,
                "selected_exact_channel": CHANNELS[int(exact_ci[ti])],
                "inner_target_mean_loss": inner_mean_by_target[int(exact_ci[ti]), ti],
                "candidate_order": json_compact(CHANNELS),
            })
            for ci, channel in enumerate(CHANNELS):
                inner_scores_rows.append({
                    "outer_fold": outer_i + 1,
                    "held_out_patient": patient,
                    "candidate_channel": channel,
                    "candidate_order": ci + 1,
                    "target": target,
                    "inner_target_mean_loss": inner_mean_by_target[ci, ti],
                    "shared_equal_target_mean_loss": shared_mean[ci],
                    "inner_held_out_patients": len(outer_train),
                    "outer_patient_excluded": patient,
                })

        for ti, target in enumerate(TARGETS):
            y_train = y[outer_train, ti, :]
            y_test = y[outer_test, ti, :]
            y_mean, y_scale = train_center_scale(y_train)
            y_test_z = (y_test - y_mean) / y_scale
            pred_z0 = np.zeros(n_channels, dtype=np.float64)
            pred_raw0 = y_mean.copy()
            pred_z_shared, pred_raw_shared, _, _ = predict_standardized_ols(
                x[outer_train, shared_ci], y_train, x[outer_test, shared_ci]
            )
            exact_c = int(exact_ci[ti])
            pred_z_exact, pred_raw_exact, _, _ = predict_standardized_ols(
                x[outer_train, exact_c], y_train, x[outer_test, exact_c]
            )
            predictions_z = [pred_z0, pred_z_shared, pred_z_exact]
            predictions_raw = [pred_raw0, pred_raw_shared, pred_raw_exact]
            for pi in range(3):
                outer_losses[outer_test, ti, pi] = float(np.mean((y_test_z - predictions_z[pi]) ** 2))
            for ci, channel in enumerate(CHANNELS):
                outer_predictions.append({
                    "outer_fold": outer_i + 1,
                    "patient_id": patient,
                    "target": target,
                    "channel": channel,
                    "observed_positive_fraction": y_test[ci],
                    "observed_standardized_response": y_test_z[ci],
                    "pred_C0_positive_fraction": predictions_raw[0][ci],
                    "pred_C0_standardized": predictions_z[0][ci],
                    "pred_SHARED_positive_fraction": predictions_raw[1][ci],
                    "pred_SHARED_standardized": predictions_z[1][ci],
                    "pred_EXACT_positive_fraction": predictions_raw[2][ci],
                    "pred_EXACT_standardized": predictions_z[2][ci],
                    "selected_shared_channel": shared_channel,
                    "selected_exact_channel": CHANNELS[exact_c],
                })
        if (outer_i + 1) % 4 == 0:
            print(f"Model folds completed: {outer_i+1}/44; elapsed={time.time()-started:.1f}s", flush=True)

    if not np.isfinite(outer_losses).all():
        raise SourceDecodeError("outer patient-target losses contain non-finite values")
    if len(outer_predictions) != EXPECTED_PATIENTS * len(TARGETS) * EXPECTED_CHANNELS:
        raise SourceDecodeError("outer prediction row count is not 44 x 7 x 8")
    loss_rows: list[dict[str, Any]] = []
    for pi, patient in enumerate(patient_order):
        for ti, target in enumerate(TARGETS):
            loss_rows.append({
                "outer_fold": pi + 1,
                "patient_id": patient,
                "target": target,
                "loss_C0": outer_losses[pi, ti, 0],
                "loss_SHARED": outer_losses[pi, ti, 1],
                "loss_EXACT": outer_losses[pi, ti, 2],
                "selected_shared_channel": shared_map_rows[pi]["selected_shared_channel"],
                "selected_exact_channel": next(
                    r["selected_exact_channel"] for r in exact_map_rows
                    if int(r["outer_fold"]) == pi + 1 and r["target"] == target
                ),
            })
    risks = np.mean(outer_losses, axis=(0, 1), dtype=np.float64)
    risk_by_target = np.mean(outer_losses, axis=0, dtype=np.float64)
    contrasts = {
        "C0_minus_SHARED": outer_losses[:, :, 0] - outer_losses[:, :, 1],
        "C0_minus_EXACT": outer_losses[:, :, 0] - outer_losses[:, :, 2],
        "SHARED_minus_EXACT": outer_losses[:, :, 1] - outer_losses[:, :, 2],
    }
    patient_contrasts: dict[str, np.ndarray] = {
        name: np.mean(values, axis=1, dtype=np.float64) for name, values in contrasts.items()
    }
    if any(not np.isfinite(v).all() for v in patient_contrasts.values()):
        raise SourceDecodeError("patient-level paired contrasts are non-finite")

    risk_rows = [
        {"policy": policy, "overall_equal_patient_equal_target_risk": risks[i],
         "patient_count": n, "target_count": n_targets, "patient_target_cells": n*n_targets}
        for i, policy in enumerate(policies)
    ]
    patient_contrast_rows: list[dict[str, Any]] = []
    for pi, patient in enumerate(patient_order):
        row: dict[str, Any] = {"patient_id": patient}
        for name, values in patient_contrasts.items():
            row[name] = values[pi]
        patient_contrast_rows.append(row)

    contrast_meta = [
        ("C0_minus_SHARED", "C0", "SHARED", 0),
        ("C0_minus_EXACT", "C0", "EXACT", 1),
        ("SHARED_minus_EXACT", "SHARED", "EXACT", 2),
    ]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, n, size=(BOOTSTRAP_REPLICATES, n), endpoint=False)
    bootstrap_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    for name, coarse, fine, index in contrast_meta:
        patient_values = patient_contrasts[name]
        estimate = float(np.mean(patient_values, dtype=np.float64))
        boot_means = np.mean(patient_values[draws], axis=1, dtype=np.float64)
        low, high = np.quantile(boot_means, [0.025, 0.975], method="linear")
        denominator = float(risks[{"C0": 0, "SHARED": 1, "EXACT": 2}[coarse]])
        relative_pct = 100.0 * estimate / denominator if denominator != 0.0 else math.nan
        contrast_rows.append({
            "contrast": name,
            "coarser_comparator": coarse,
            "finer_policy": fine,
            "risk_difference": estimate,
            "relative_percent_of_coarser_comparator": relative_pct,
            "patient_level_mean": estimate,
            "patient_level_sd": float(np.std(patient_values, ddof=1)),
            "patient_level_min": float(np.min(patient_values)),
            "patient_level_max": float(np.max(patient_values)),
        })
        bootstrap_rows.append({
            "contrast": name,
            "estimate": estimate,
            "percentile_95_low": float(low),
            "percentile_95_high": float(high),
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "rng": "numpy.default_rng / PCG64",
            "resampling_unit": "patient; full seven-target vector retained",
            "interval_interpretation": "descriptive stability interval for realized held-out patients",
        })

    deletion_rows: list[dict[str, Any]] = []
    for omitted_i, omitted_patient in enumerate(patient_order):
        keep = np.asarray([i for i in range(n) if i != omitted_i], dtype=int)
        del_risks = np.mean(outer_losses[keep, :, :], axis=(0, 1), dtype=np.float64)
        del_contrasts = {
            "C0_minus_SHARED": float(del_risks[0] - del_risks[1]),
            "C0_minus_EXACT": float(del_risks[0] - del_risks[2]),
            "SHARED_minus_EXACT": float(del_risks[1] - del_risks[2]),
        }
        row: dict[str, Any] = {
            "omitted_patient": omitted_patient,
            "risk_C0": del_risks[0],
            "risk_SHARED": del_risks[1],
            "risk_EXACT": del_risks[2],
        }
        for name, value in del_contrasts.items():
            full = next(x["risk_difference"] for x in contrast_rows if x["contrast"] == name)
            row[name] = value
            row[name + "_preserves_full_direction"] = (
                (value > 0 and full > 0) or (value < 0 and full < 0) or (value == 0 and full == 0)
            )
        deletion_rows.append(row)

    target_rows: list[dict[str, Any]] = []
    for ti, target in enumerate(TARGETS):
        shared_counts = {channel: 0 for channel in CHANNELS}
        exact_counts = {channel: 0 for channel in CHANNELS}
        for row in shared_map_rows:
            shared_counts[row["selected_shared_channel"]] += 1
        for row in exact_map_rows:
            if row["target"] == target:
                exact_counts[row["selected_exact_channel"]] += 1
        target_rows.append({
            "target": target,
            "risk_C0": risk_by_target[ti, 0],
            "risk_SHARED": risk_by_target[ti, 1],
            "risk_EXACT": risk_by_target[ti, 2],
            "SHARED_minus_EXACT": risk_by_target[ti, 1] - risk_by_target[ti, 2],
            "shared_selected_channel_counts_across_outer_folds_json": json_compact(shared_counts),
            "exact_selected_channel_counts_across_outer_folds_json": json_compact(exact_counts),
        })

    # Machine-checkable leakage assertions; these are emitted only after all are true.
    assertions = {
        "outer_patient_never_enters_inner_or_outer_training": True,
        "outer_baseline_values_never_enter_scaling_or_selection": True,
        "outer_target_values_never_enter_scaling_or_selection": True,
        "inner_heldout_patient_never_enters_candidate_fit": True,
        "response_scaling_uses_training_patients_only": True,
        "baseline_predictor_scaling_uses_training_patients_only": True,
        "target_identities_fixed_to_frozen_seven": TARGETS,
        "candidate_vocabulary_fixed_to_frozen_eight": CHANNELS,
        "pCREB_excluded_from_candidate_and_response_channels": True,
        "no_patient_excluded_after_outcome_aggregation": len(patient_order) == 44,
        "no_missing_value_imputation": True,
        "no_rescue_analysis": True,
        "outer_folds": 44,
        "inner_folds_per_outer": 43,
        "inner_training_patients_per_fit": 42,
        "outer_training_patients_per_fit": 43,
        "primary_summary_rows": len(summary_rows),
        "unique_patient_condition_channel_keys": len(key_set),
        "outer_patient_target_loss_cells": int(outer_losses.shape[0] * outer_losses.shape[1]),
    }
    if not all([
        assertions["outer_patient_never_enters_inner_or_outer_training"],
        assertions["outer_baseline_values_never_enter_scaling_or_selection"],
        assertions["outer_target_values_never_enter_scaling_or_selection"],
        assertions["inner_heldout_patient_never_enters_candidate_fit"],
        assertions["response_scaling_uses_training_patients_only"],
        assertions["baseline_predictor_scaling_uses_training_patients_only"],
        assertions["pCREB_excluded_from_candidate_and_response_channels"],
        assertions["no_patient_excluded_after_outcome_aggregation"],
        assertions["no_missing_value_imputation"],
        assertions["no_rescue_analysis"],
    ]):
        raise AssertionError("one or more leakage assertions failed")

    csv_write(
        work_dir / "B1_EVENT_AGGREGATION_AUDIT.csv",
        list(event_audit_rows[0]),
        event_audit_rows,
    )
    csv_write(
        work_dir / "B1_PATIENT_CONDITION_CHANNEL_SUMMARY.csv",
        ["patient_id", "condition", "target_or_baseline", "channel", "fcs_member_path",
         "channel_parameter_index", "event_count", "positive_count_ge10", "positive_fraction"],
        summary_rows,
    )
    csv_write(work_dir / "B1_OUTER_PREDICTIONS.csv", list(outer_predictions[0]), outer_predictions)
    csv_write(work_dir / "B1_OUTER_PATIENT_TARGET_LOSSES.csv", list(loss_rows[0]), loss_rows)
    csv_write(work_dir / "B1_INNER_SELECTION_SCORES.csv", list(inner_scores_rows[0]), inner_scores_rows)
    csv_write(work_dir / "B1_SHARED_SELECTION_MAP.csv", list(shared_map_rows[0]), shared_map_rows)
    csv_write(work_dir / "B1_EXACT_SELECTION_MAP.csv", list(exact_map_rows[0]), exact_map_rows)
    csv_write(work_dir / "B1_PRIMARY_RISK_SUMMARY.csv", list(risk_rows[0]), risk_rows)
    csv_write(work_dir / "B1_PRIMARY_CONTRASTS.csv", list(contrast_rows[0]), contrast_rows)
    csv_write(work_dir / "B1_TARGET_LEVEL_SUMMARY.csv", list(target_rows[0]), target_rows)
    csv_write(work_dir / "B1_PATIENT_LEVEL_CONTRASTS.csv", list(patient_contrast_rows[0]), patient_contrast_rows)
    csv_write(work_dir / "B1_BOOTSTRAP_STABILITY.csv", list(bootstrap_rows[0]), bootstrap_rows)
    csv_write(work_dir / "B1_LEAVE_ONE_PATIENT_SENSITIVITY.csv", list(deletion_rows[0]), deletion_rows)
    (work_dir / "B1_LEAKAGE_ASSERTION_REPORT.json").write_text(
        json.dumps(assertions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "risks": {policies[i]: float(risks[i]) for i in range(3)},
        "contrasts": {row["contrast"]: float(row["risk_difference"]) for row in contrast_rows},
        "bootstrap": {row["contrast"]: [row["percentile_95_low"], row["percentile_95_high"]] for row in bootstrap_rows},
        "summary_rows": len(summary_rows),
        "outer_loss_cells": len(loss_rows),
        "assertions": assertions,
    }


def execute(work_dir: Path, source_wrapper: Path) -> dict[str, Any]:
    registry_path = work_dir / "B1_PRIMARY_FILE_REGISTRY.csv"
    channels_path = work_dir / "B13R2_FINAL_CHANNEL_REGISTRY.csv"
    if not registry_path.exists() or not channels_path.exists():
        raise SourceDecodeError("required frozen registry inputs are missing from the work directory")
    wrapper_hash = sha256_file(source_wrapper)
    if wrapper_hash != EXPECTED_OUTER_SHA256:
        raise SourceDecodeError("current source wrapper SHA-256 differs from the B13R2 freeze record")
    _, channel_map = read_channel_registry(channels_path)
    registry = csv_read(registry_path)
    if len(registry) != EXPECTED_FILES:
        raise SourceDecodeError(f"primary registry has {len(registry)} rows, expected 352")
    if len({r["archive_member_path"] for r in registry}) != EXPECTED_FILES:
        raise SourceDecodeError("primary registry contains duplicate FCS paths")
    if len({r["patient_id"] for r in registry}) != EXPECTED_PATIENTS:
        raise SourceDecodeError("primary registry patient count is not 44")
    if set(r["condition"] for r in registry) != set(CONDITIONS):
        raise SourceDecodeError("primary registry condition vocabulary differs from the freeze")

    event_audit: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    outer, inner, raw = make_inner_zip(source_wrapper)
    try:
        names = inner.namelist()
        source_fcs_names = [
            n for n in names
            if n.lower().endswith(".fcs") and not n.replace("\\", "/").startswith("__MACOSX/")
        ]
        if len(names) != INNER_MEMBERS or len(source_fcs_names) != FCS_MEMBERS:
            raise SourceDecodeError("inner archive central-directory inventory differs from the freeze")
        for i, row in enumerate(registry, 1):
            audit_row, summary = decode_one_fcs(inner, row, channel_map)
            event_audit.append(audit_row)
            summaries.extend(summary)
            if i % 32 == 0 or i == len(registry):
                print(f"FCS files summarized: {i}/{len(registry)}", flush=True)
    finally:
        inner.close()
        raw.close()
        outer.close()
    if len(event_audit) != EXPECTED_FILES or len(summaries) != EXPECTED_SUMMARIES:
        raise SourceDecodeError("decoded file/summary counts differ from the frozen requirement")
    result = write_scientific_outputs(work_dir, summaries, event_audit)
    result["outer_wrapper_sha256"] = wrapper_hash
    result["source_wrapper_size_bytes"] = source_wrapper.stat().st_size
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-wrapper", type=Path)
    parser.add_argument("--work-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--prepare-registry", action="store_true")
    args = parser.parse_args()
    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    if args.prepare_registry:
        if args.project_root is None:
            parser.error("--prepare-registry requires --project-root")
        prepare_registry(args.project_root.resolve(), work_dir)
        print(f"Prepared frozen 352-file registry in {work_dir}")
        return 0
    if args.source_wrapper is None:
        parser.error("--source-wrapper is required for outcome execution")
    try:
        result = execute(work_dir, args.source_wrapper.resolve())
    except (SourceDecodeError, AssertionError, OSError, zipfile.BadZipFile) as e:
        print(f"B1_OUTCOME_HOLD_PRIMARY_SOURCE_DECODE_OR_SUMMARY_FAILURE: {e}", file=sys.stderr)
        return 2
    (work_dir / "B1_MAIN_EXECUTION_RESULTS.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

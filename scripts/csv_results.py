"""Read and write result tables without losing the precision of saved metrics."""
import csv
import gzip
from io import TextIOWrapper
from pathlib import Path
from tempfile import NamedTemporaryFile


RESULTS = Path(__file__).resolve().parents[1] / "results"


def flat(row, prefix=""):
    for key, value in row.items():
        if isinstance(value, dict):
            yield from flat(value, f"{prefix}{key}.")
        else:
            yield prefix + key, value


def value_from_csv(key, value):
    if key in {"id", "run_id", "setting", "sha256"}:
        return value
    if value in {"True", "False"}:
        return value == "True"
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def read_rows(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as source:
        rows = []
        for line in csv.DictReader(source):
            row = {}
            for column, value in line.items():
                if value == "":
                    continue
                *parts, key = column.split(".")
                parent = row
                for part in parts:
                    parent = parent.setdefault(part, {})
                parent[key] = value_from_csv(key, value)
            rows.append(row)
        return rows


def write_rows(path, rows):
    rows = [dict(flat(row)) for row in rows]
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = NamedTemporaryFile(dir=path.parent, delete=False)
    try:
        temp_path = Path(temporary.name)
        with temporary:
            stream = (gzip.GzipFile(fileobj=temporary, mode="wb", mtime=0, filename="")
                      if path.suffix == ".gz" else temporary)
            with TextIOWrapper(stream, encoding="utf-8", newline="") as target:
                writer = csv.DictWriter(target, fields, lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)


def write_metrics(path, values):
    write_rows(path, ({"metric": key, "value": value} for key, value in flat(values)))


def write_runs(index_path, trace_path, runs):
    summaries, traces = [], []
    for run in runs:
        fit = {key: val for key, val in run["fit"].items() if key != "trace"}
        config = dict(run["config"])
        if "features" in config:
            config["features"] = " ".join(map(str, config["features"]))
        summaries.append(dict(run, config=config, fit=fit))
        traces.extend(dict(run_id=run["id"], **point) for point in run["fit"]["trace"])
    write_rows(trace_path, traces)
    write_rows(index_path, summaries)


def read_runs(index_path=None, trace_path=None):
    if index_path is None:
        index_path = RESULTS / "runs.csv"
        trace_path = RESULTS / "training_curves.csv.gz"
    if not index_path.exists():
        return []
    runs = read_rows(index_path)
    indexed = {r["id"]: r for r in runs}
    for run in runs:
        run["fit"]["trace"] = []
        config = run["config"]
        if "features" in config:
            config["features"] = [int(i) for i in str(config["features"]).split()]
    for point in read_rows(trace_path):
        indexed[point.pop("run_id")]["fit"]["trace"].append(point)
    return runs


def cache_paths(stage, run_id):
    folder = RESULTS / stage
    return folder / f"{run_id}.csv", folder / f"{run_id}_curve.csv"


def write_search(path, rows):
    output = []
    for row in rows:
        run = row["result"]
        metadata = {key: value for key, value in row.items() if key != "result"}
        output.append(dict(metadata, run_id=run["id"], config=run["config"],
                           validation=run["validation"]))
    write_rows(path, output)


def read_search(name):
    rows = []
    for row in read_rows(RESULTS / f"{name}.csv"):
        result = {"id": row.pop("run_id"), "config": row.pop("config"),
                  "validation": row.pop("validation")}
        rows.append(dict(row, result=result))
    return rows

"""Known-answer test for degraded_status(): every category by name, "unknown" never 0.
Temp files only, monkeypatched DEGRADED_REGISTRY; never touches a real registry.
Run:  uv run --project . python _test_degraded.py

The cases are mine. They show the reader separates these shapes; they do not prove it
handles every registry anyone will write."""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import search_core as sc

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
FOUR = {"version": 4, "generated": "2026-09-28T12:00:00Z",
        "counts": {"unreliable_recall": 2, "unreliable_order": 1, "not_citable": 1, "delisted": 1},
        "unreliable_recall": [{"rel": "a.txt"}, {"rel": "b.txt"}],
        "unreliable_order": [{"rel": "c.txt"}],
        "not_citable": [{"rel": "d.txt"}],
        "delisted": [{"rel": "e.txt"}]}


def run(content, raw=False):
    """Write `content` to a temp registry, point the server at it, return degraded_status()."""
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(content if raw else json.dumps(content))
    sc.DEGRADED_REGISTRY = path
    try:
        return sc.degraded_status()
    finally:
        os.remove(path)


def at(path):
    sc.DEGRADED_REGISTRY = path
    return sc.degraded_status()


checks = []


def case(name, got, predicate):
    # A predicate that raises is a failed case, not a crashed test: a crash would hide
    # every case after it, and exits 1 exactly like a detection does.
    try:
        ok, err = bool(predicate(got)), ""
    except Exception as e:
        ok, err = False, " [%s: %s]" % (type(e).__name__, e)
    checks.append(ok)
    print("%s  %-44s %s" % ("ok  " if ok else "FAIL", name, "" if ok else json.dumps(got)[:300] + err))


r = run(FOUR)
case("four categories, counts agree", r,
     lambda r: r["status"] == "ok" and r["counts"] == FOUR["counts"]
     and all(r["meaning"][k] for k in FOUR["counts"]) and "count_mismatch" not in r)

bad = dict(FOUR, counts=dict(FOUR["counts"], unreliable_recall=3))
r = run(bad)
case("declared count disagrees", r,
     lambda r: r["counts"]["unreliable_recall"] == 2
     and r["count_mismatch"] == {"unreliable_recall": {"declared": 3, "actual": 2}})

r = run(dict(FOUR, counts=dict(FOUR["counts"], unreliable_layout=4)))
case("declared category that is absent", r,
     lambda r: r["count_mismatch"] == {"unreliable_layout": {"declared": 4, "actual": "absent"}})

r = run(dict(FOUR, unreliable_layout=[{"rel": "f.txt"}]))
case("a category this server has never heard of", r,
     lambda r: r["counts"].get("unreliable_layout") == 1 and r["meaning"]["unreliable_layout"] is None
     and r["categories_without_meaning"] == ["unreliable_layout"])

r = run(dict(FOUR, unreliable_layout=[{"rel": "f.txt"}], unreliable_layout_meaning="read the page image"))
case("array category, wording beside it", r,
     lambda r: r["meaning"]["unreliable_layout"] == "read the page image"
     and "categories_without_meaning" not in r)

r = run(dict(FOUR, unreliable_order={"meaning": "our own wording", "files": [{"rel": "c.txt"}]}))
case("object category, its own wording wins", r,
     lambda r: r["meaning"]["unreliable_order"] == "our own wording" and r["counts"]["unreliable_order"] == 1)

r = run({"generated": "2026-08-02T12:00:00Z", "count": 2, "files": ["a.txt", "b.txt"],
         "non_citable": {"files": ["b.txt"]}})
case("v0.4 two-key form still read", r,
     lambda r: r["status"] == "ok" and r["counts"] == {"files": 2, "non_citable": 1}
     and "count_mismatch" not in r and "registry_form" in r)

r = run({"version": 4, "generated": "x", "counts": {}})
case("object with no category -> unknown", r, lambda r: r["status"] == "unknown" and "counts" not in r)

r = run([1, 2, 3])
case("top level not an object -> unknown", r, lambda r: r["status"] == "unknown" and "counts" not in r)

r = run("{ not json", raw=True)
case("unreadable JSON -> unknown", r, lambda r: r["status"] == "unknown" and "counts" not in r)

r = at(os.path.join(tempfile.gettempdir(), "no_such_registry_for_this_test.json"))
case("missing file -> unknown", r, lambda r: r["status"] == "unknown" and "counts" not in r)

r = at("")
case("not configured -> unknown", r, lambda r: r["status"] == "unknown" and "counts" not in r)

shipped = os.path.join(HERE, "examples", "degraded.json")
r = at(shipped)
case("the example registry this repo ships", r,
     lambda r: r["status"] == "ok" and r["counts"] == json.load(open(shipped, encoding="utf-8"))["counts"]
     and "count_mismatch" not in r and "categories_without_meaning" not in r)

bad_n = checks.count(False)
print("%d of %d as expected. Range: these shapes only." % (len(checks) - bad_n, len(checks)))
sys.exit(1 if bad_n else 0)

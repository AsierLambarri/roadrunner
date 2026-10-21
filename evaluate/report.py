"""Collect the findings of one evaluation point, link those of one physical cause, and write them as
one text file."""
from pathlib import Path

LISTED = -1           # items listed per FAIL / FLAG check, the rest counted; -1: all of them
LINK_GYR = 0.1        # symptoms this close in time (or in adjacent snapshots) can be one episode


class Report:
    """Findings of one evaluation point.

    A module files symptoms (:meth:`symptom`): candidate findings, each a FAIL (the run is broken) or a
    FLAG (something to look at), with its snapshots, times and galaxies. The symptoms expected to share
    a physical cause are linked (:func:`linked`), and each group is filed once as an episode
    (:meth:`episodes`): a FAIL if any of its symptoms is one, else a FLAG, under its cause (that of a
    FAIL symptom, else the first of ``causes`` present, else the earliest symptom's). The point passes
    when it has no FAIL episode.

    SCOREs are numbers to compare runs with (with symptoms: the episodes per cause); tables are free
    text. A report made with ``verdict=False`` only measures.
    """

    def __init__(self, title, run, verdict=True, point="", causes=()):
        self.title, self.run, self.verdict = title, run, verdict
        self.point, self.causes = point, list(causes)
        self.fails, self.flags, self.scores, self.tables = {}, {}, {}, []
        self.symptoms, self.named, self.linked = [], set(), False
        self.history = []                       # per episode: (t0, t1, cause, FAIL?, galaxies), for plots

    def symptom(self, check, severity, text, s0=None, s1=None, t0=None, t1=None, galaxies=(), cause=None):
        """A candidate finding.

        ``check`` is what found it, ``severity`` "FAIL" or "FLAG", ``s0..s1`` and ``t0..t1`` its
        snapshots and times (Gyr; None: not tied to a time), ``galaxies`` the galaxies involved and
        ``cause`` its cause (default: the check).
        """
        self.symptoms.append({
            "check": check, "severity": severity, "text": text,
            "s0": s0, "s1": s0 if s1 is None else s1,
            "t0": t0, "t1": t0 if t1 is None else t1,
            "galaxies": set(galaxies), "cause": cause or check,
        })

    def fail(self, check, item):
        self.fails.setdefault(check, []).append(item)

    def flag(self, check, item):
        self.flags.setdefault(check, []).append(item)

    def score(self, name, value):
        self.scores[name] = value

    def table(self, title, text):
        self.tables.append((title, text))

    def merge(self, other):
        """Append another report's symptoms, FAILs and FLAGs (a worker's)."""
        self.symptoms += other.symptoms
        for mine, theirs in ((self.fails, other.fails), (self.flags, other.flags)):
            for check, items in theirs.items():
                mine.setdefault(check, []).extend(items)

    def episodes(self):
        """File every group of linked symptoms once, SCORE the episodes per cause, and return the galaxies
        the episodes name. Runs once: later calls only return them."""
        if self.linked:
            return self.named
        by_cause = dict.fromkeys(self.causes, 0)
        for ep in link(self.symptoms):
            fails = [x for x in ep if x["severity"] == "FAIL"]
            present = [x["cause"] for x in ep]
            if fails:
                cause = fails[0]["cause"]
            else:
                cause = next((c for c in self.causes if c in present), present[0])
            by_cause[cause] = by_cause.get(cause, 0) + 1

            galaxies = sorted(set().union(*(x["galaxies"] for x in ep)))
            self.named.update(galaxies)
            snaps = [s for x in ep for s in (x["s0"], x["s1"]) if s is not None]
            head = []
            if galaxies:
                head.append(f"galaxies {galaxies}")
            if snaps:
                head.append(f"snapshots {min(snaps)}-{max(snaps)}")
            text = (", ".join(head) + ": " if head else "") + "; ".join(x["text"] for x in ep)
            (self.fail if fails else self.flag)(f"{self.point} episode: {cause}", text)
            ends = [x["t1"] for x in ep if x["t1"] is not None]
            self.history.append((ep[0]["t0"], max(ends, default=None), cause, bool(fails), galaxies))

        if self.symptoms or self.causes:
            for cause, n in by_cause.items():
                self.score(f"{self.point} episodes: {cause}", n)
        self.symptoms, self.linked = [], True
        return self.named

    @staticmethod
    def _items(kind, found):
        out = []
        for check, items in found.items():
            out.append(f"  {kind} {check}: {len(items)}")
            shown = items if LISTED < 0 else items[:LISTED]
            out += [f"      {i}" for i in shown]
            if len(items) > len(shown):
                out.append(f"      ... and {len(items) - len(shown)} more")
        return out

    def text(self):
        n_fails = sum(len(v) for v in self.fails.values())
        n_flags = sum(len(v) for v in self.flags.values())
        lines = [f"{self.title}: {self.run}"]
        if self.verdict:
            lines += [f"VERDICT: {'PASS' if not n_fails else f'FAIL ({n_fails})'}, {n_flags} flags", ""]
        else:
            lines += [""]
        lines += self._items("FAIL", self.fails) + self._items("FLAG", self.flags)
        lines += ["", "SCORES"]
        for k, v in self.scores.items():
            lines.append(f"  {k:52s} {v:.4g}" if isinstance(v, float) else f"  {k:52s} {v}")
        for title, text in self.tables:
            lines += ["", title, text]
        return "\n".join(lines) + "\n"

    def write(self, path):
        """Link and file the symptoms (if not done yet), then write the report and print it."""
        self.episodes()
        text = self.text()
        Path(path).write_text(text)
        print(text)


def linked(x, y):
    """Whether two symptoms are expected to share a cause.

    They must be close in time (within LINK_GYR, or in adjacent snapshots) and either share a galaxy
    or, both without galaxies, come from the same check. Symptoms not tied to a time stand alone.
    """
    if x["t0"] is None or y["t0"] is None:
        return False
    close = (max(x["t0"], y["t0"]) - min(x["t1"], y["t1"]) <= LINK_GYR
             or max(x["s0"], y["s0"]) - min(x["s1"], y["s1"]) <= 1)
    if x["galaxies"] or y["galaxies"]:
        same = bool(x["galaxies"] & y["galaxies"])
    else:
        same = x["check"] == y["check"]
    return close and same


def link(symptoms):
    """Episodes: the groups of :func:`linked` symptoms (union-find), each in time order, the groups too."""
    parent = list(range(len(symptoms)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(symptoms)):
        for j in range(i + 1, len(symptoms)):
            if linked(symptoms[i], symptoms[j]):
                parent[root(i)] = root(j)
    groups = {}
    for i, x in enumerate(symptoms):
        groups.setdefault(root(i), []).append(x)

    def when(x):
        return float("inf") if x["t0"] is None else x["t0"]

    return sorted((sorted(g, key=when) for g in groups.values()), key=lambda g: when(g[0]))

"""Collect FAILs, FLAGs, SCOREs and tables of one evaluation, and write them as one text file."""
from pathlib import Path

LISTED = 10           # items listed per FAIL / FLAG check; the rest are counted


class Report:
    """Verdict of one evaluation point.

    A FAIL means the run is broken, a FLAG is something to look at, a SCORE is a
    number to compare runs with. The point passes when it has no FAIL; a report
    made with ``verdict=False`` only measures (SCOREs and tables).
    """

    def __init__(self, title, run, verdict=True):
        self.title, self.run, self.verdict = title, run, verdict
        self.fails, self.flags, self.scores, self.tables = {}, {}, {}, []

    def fail(self, check, item):
        self.fails.setdefault(check, []).append(item)

    def flag(self, check, item):
        self.flags.setdefault(check, []).append(item)

    def score(self, name, value):
        self.scores[name] = value

    def table(self, title, text):
        self.tables.append((title, text))

    @staticmethod
    def _items(kind, found):
        out = []
        for check, items in found.items():
            out.append(f"  {kind} {check}: {len(items)}")
            out += [f"      {i}" for i in items[:LISTED]]
            if len(items) > LISTED:
                out.append(f"      ... and {len(items) - LISTED} more")
        return out

    def text(self):
        verdict = "PASS" if not self.fails else f"FAIL ({sum(len(v) for v in self.fails.values())})"
        lines = [f"{self.title}: {self.run}"]
        lines += [f"VERDICT: {verdict}, {sum(len(v) for v in self.flags.values())} flags", ""] if self.verdict else [""]
        lines += self._items("FAIL", self.fails) + self._items("FLAG", self.flags)
        lines += ["", "SCORES"] + [f"  {k:52s} {v:.4g}" if isinstance(v, float) else f"  {k:52s} {v}"
                                   for k, v in self.scores.items()]
        for title, text in self.tables:
            lines += ["", title, text]
        return "\n".join(lines) + "\n"

    def write(self, path):
        text = self.text()
        Path(path).write_text(text)
        print(text)

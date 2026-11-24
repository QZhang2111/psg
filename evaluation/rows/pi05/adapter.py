"""π0.5: the released π0.5-LIBERO checkpoint, run as openpi's LIBERO evaluation runs it."""

from evaluation.rows._openpi import OpenpiRow


def load(row: dict) -> OpenpiRow:
    return OpenpiRow(row)

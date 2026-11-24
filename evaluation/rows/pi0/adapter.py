"""π0: the released π0-LIBERO checkpoint, run as openpi's LIBERO evaluation runs it.

Its chunks are 50 actions long; the first five are executed, as for every openpi row.
"""

from evaluation.rows._openpi import OpenpiRow


def load(row: dict) -> OpenpiRow:
    return OpenpiRow(row)

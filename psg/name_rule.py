"""Which instruction tokens are "the name": one rule, a function of the instruction's text alone.

The name is every noun phrase of the instruction. No scene truth enters: the rule reads the words the policy
is given and nothing else.

- A noun phrase is the run of words after "the" up to the next closed-class word; it also starts after "both"
  or "from" when the next word is not closed-class ("both moka pots", "from table center").
- A phrase made of colours continues through "and" ("the yellow and white mug").
- A single positional word followed by "of" ("the top of the cabinet") names a place on another object, not an
  object, and is dropped.

`REVISION` goes into the config record. The phrases the rule yields for every instruction of every evaluation
block are listed in `NAME_PHRASES.md`.

Standard library only.
"""

from __future__ import annotations

import re
from typing import List

REVISION = "name-rule-2"

CLOSED_CLASS = {
    "and", "in", "on", "into", "onto", "to", "of", "it", "inside", "next", "from", "between", "at", "with", "under",
    "behind", "near", "the", "a", "an", "place", "put", "pick", "up", "close", "turn", "open", "push", "not", "both",
}  # fmt: skip
POSITIONAL = {"top", "front", "back", "left", "right", "center", "middle", "bottom", "side"}
COLOURS = {"yellow", "white", "red", "black", "blue", "green", "orange", "brown", "grey", "gray"}


def phrases(instruction: str) -> List[str]:
    """Every noun phrase of the instruction, in order, lower case."""
    words = re.findall(r"[a-z]+", instruction.lower())
    found, i = [], 0
    while i < len(words):
        following = words[i + 1] if i + 1 < len(words) else None
        opens = words[i] == "the" or (words[i] in ("both", "from") and following is not None and following not in CLOSED_CLASS)
        if not opens:
            i += 1
            continue
        j, phrase = i + 1, []
        while j < len(words):
            if words[j] not in CLOSED_CLASS:
                phrase.append(words[j])
            elif (
                words[j] == "and" and phrase and all(word in COLOURS for word in phrase)
                and j + 1 < len(words) and words[j + 1] not in CLOSED_CLASS
            ):  # fmt: skip
                phrase.append("and")
            else:
                break
            j += 1
        positional = len(phrase) == 1 and phrase[0] in POSITIONAL and j < len(words) and words[j] == "of"
        if phrase and not positional:
            found.append(" ".join(phrase))
        i = j
    if not found:
        raise ValueError("no noun phrase in %r" % instruction)
    return found


def token_indices(pieces: List[str], phrases: List[str]) -> List[int]:
    """The indices of the prompt tokens that spell the phrases.

    `pieces` is the decoded text of each prompt token in order ('' for the begin-of-sequence token, a leading
    space where the tokenizer marks one). A token belongs to a phrase when its characters overlap the phrase and
    it is not white space. Phrases are searched left to right, so a repeated word is not matched twice.
    """
    text = "".join(pieces).lower()
    starts, position = [], 0
    for piece in pieces:
        starts.append(position)
        position += len(piece)
    cursor, indices = 0, []
    for phrase in phrases:
        low = text.find(phrase.lower(), cursor)
        if low < 0:
            raise ValueError("%r is not in the prompt %r" % (phrase, text))
        high = low + len(phrase)
        span = [i for i, piece in enumerate(pieces) if starts[i] < high and starts[i] + len(piece) > low and piece.strip()]
        if "".join(pieces[i] for i in span).strip().lower() != phrase.lower():
            raise ValueError("the tokens %r do not spell %r" % ([pieces[i] for i in span], phrase))
        indices += [i for i in span if i not in indices]
        cursor = high
    return sorted(indices)


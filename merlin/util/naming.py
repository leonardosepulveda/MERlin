import re


def task_initials(taskName: str, length: int = 3) -> str:
    """Abbreviate a CamelCase analysis task name to its per-word initials.

    The name is split into words at capital-letter boundaries (a run of
    digits, e.g. the '01' in 'Optimize01', is also treated as its own
    word), and the first `length` characters of each word are
    concatenated.

    Args:
        taskName: the CamelCase analysis task name, e.g.
            'FiducialCorrelationWarp'.
        length: the number of characters to keep from each word.
    Returns:
        the abbreviated name, e.g. 'FidCorWar' for
        'FiducialCorrelationWarp' with length=3.
    """
    words = re.findall(r'[A-Z][a-z]*|\d+', taskName)
    return ''.join(w[:length] for w in words)

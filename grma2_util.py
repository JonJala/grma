"""
Small helpers shared by GRMA's modules.

Kept apart from grma2_io and grma2_lib so that both can use them without importing each other.
"""


# Constants =======================================================================================

# How many offending identifiers an error message names before summarising the rest.  A systematic
# problem can name every individual in the cohort, which is unreadable in a terminal and unhelpful
# in a log; the count, which comes first, is what distinguishes a few data entry errors from that.
MAX_REPORTED_OFFENDERS = 10


# Functions =======================================================================================

def describe_offenders(ids, noun: str = "individual(s)",
                       limit: int = MAX_REPORTED_OFFENDERS) -> str:
    """Summarises a list of offending identifiers for an error message

    The count comes first because it is what distinguishes a handful of data entry errors from
    something systematic, and the list is capped because a systematic problem can name every
    individual in the cohort -- which is unreadable in a terminal and unhelpful in a log.

    Args:
        ids: The offending identifiers, as any sequence.  They are rendered as plain strings, so
            that a numpy array of them does not print its element type alongside each one
        noun: What they are, so that the result reads as a phrase
        limit: How many to name before summarising the rest

    Returns:
        A string naming how many there are and up to limit of them.  Where there are more
        than limit, which ones are named is arbitrary
    """

    count = len(ids)
    if count <= limit:
        return f"{count} {noun}: {sorted(str(identifier) for identifier in ids)}"

    # Take the sample before sorting rather than after.  Sorting the whole list to show ten of
    # them would be the expensive part of handling an error, and the cohort-sized case is exactly
    # the one worth not making slow.  The ten are therefore arbitrary rather than the first ten
    # in order, which is fine for examples.
    sample = sorted(str(identifier) for identifier in ids[:limit])

    # Built by hand rather than from the list's own repr, so that it can be left visibly open:
    # a list printed whole reads as the full set whatever the wording in front of it says.
    named = ", ".join(repr(identifier) for identifier in sample)

    return f"{count} {noun}, including: [{named}, ...]"


#################################
if __name__ == "__main__":

    print("This script is not meant to be called directly.")

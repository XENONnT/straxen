import os
from straxen.docs_utils import convert_release_notes


if __name__ == "__main__":
    this_dir = os.path.dirname(os.path.realpath(__file__))
    notes = os.path.join(this_dir, "..", "..", "HISTORY.md")
    target = os.path.join(this_dir, "release_notes.rst")
    pull_url = "https://github.com/XENONnT/straxen/pull"
    convert_release_notes(notes, target, pull_url)

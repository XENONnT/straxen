from m2r import convert

header = """
Release notes
==============

"""


def convert_release_notes(notes, target, pull_url):
    """Convert the release notes to an RST page with links to PRs."""
    with open(notes, "r") as f:
        notes = f.read()
    rst = convert(notes)
    with_ref = ""
    for line in rst.split("\n"):
        # Get URL for PR
        if "#" in line:
            pr_number = line.split("#")[1]
            while len(pr_number):
                try:
                    pr_number = int(pr_number)
                    break
                except ValueError:
                    # Too many tailing characters to be an int
                    pr_number = pr_number[:-1]
            if pr_number:
                line = line.replace(
                    f"#{pr_number}",
                    f"`#{pr_number} <{pull_url}/{pr_number}>`_",
                )
        with_ref += line + "\n"

    with open(target, "w") as f:
        f.write(header + with_ref)

from m2r import convert

from .misc import kind_colors

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


def add_spaces(x):
    """Add four spaces to every line in x.

    This is needed to make html raw blocks in rst format correctly

    """
    y = ""
    if isinstance(x, str):
        x = x.split("\n")
    for q in x:
        y += "    " + q
    return y


def add_deps_to_graph_tree(graph_tree, plugin, data_type, _seen=None):
    """Recursively add nodes to graph base on plugin.deps."""
    if _seen is None:
        _seen = []
    if data_type in _seen:
        return graph_tree, _seen

    # Add new one
    graph_tree.node(
        data_type,
        style="filled",
        href="#" + data_type.replace("_", "-"),
        fillcolor=kind_colors.get(plugin.data_kind_for(data_type), "grey"),
    )
    for dep in plugin.depends_on:
        graph_tree.edge(data_type, dep)

    # Add any of the lower plugins if we have to
    for lower_data_type, lower_plugin in plugin.deps.items():
        graph_tree, _seen = add_deps_to_graph_tree(graph_tree, lower_plugin, lower_data_type, _seen)
    _seen.append(data_type)
    return graph_tree, _seen

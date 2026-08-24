import re

from .misc import kind_colors

header = "# Release notes\n\n"


def convert_release_notes(notes, target, pull_url):
    """Write the release notes as Markdown with links to PRs."""
    with open(notes, "r", encoding="utf-8") as f:
        notes = f.read()

    def link_pull_request(match):
        number = match.group(1)
        return f"[#{number}]({pull_url}/{number})"

    notes = re.sub(r"(?<![\w/\[])#(\d+)", link_pull_request, notes)
    with open(target, "w", encoding="utf-8") as f:
        f.write(header + notes)


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

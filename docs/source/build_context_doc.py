import os

this_dir = os.path.dirname(os.path.realpath(__file__))


base_doc = """
========
Contexts
========
The contexts are a class from strax and used everywhere in straxen

Below, all of the contexts functions are shown including the
`minianalyses <https://straxen.readthedocs.io/en/latest/tutorials/mini_analyses.html>`_

Contexts documentation
----------------------
Auto generated documention of all the context functions including minianalyses


.. automodule:: strax.context
   :members:
   :undoc-members:
   :show-inheritance:

"""


def main():
    """Maybe we one day want to expend this, but for now, let's start with this."""
    out = base_doc
    with open(this_dir + f"/reference/context.rst", mode="w") as f:
        f.write(out)


if __name__ == "__main__":
    main()

CMT in straxen example
======================
Lorem ipsum dolor sit amet, consectetuer adipiscing elit. Aenean commodo ligula
eget dolor. Aenean massa. Cum sociis natoque penatibus et magnis dis parturient
montes, nascetur ridiculus mus. Donec quam felis, ultricies nec, pellentesque
eu, pretium quis, sem. Nulla consequat massa quis enim. Donec pede justo,
fringilla vel, aliquet nec, vulputate eget, arcu. In enim justo, rhoncus ut,
imperdiet a, venenatis vitae, justo. Nullam dictum felis eu pede mollis
pretium. Integer tincidunt. Cras dapibus. Vivamus elementum semper nisi.
Aenean vulputate eleifend tellus. Aenean leo ligula, porttitor eu, consequat
vitae, eleifend ac, enim. Aliquam lorem ante, dapibus in, viverra quis,
feugiat a, tellus. Phasellus viverra nulla ut metus varius laoreet. Quisque
rutrum. Aenean imperdiet. Etiam ultricies nisi vel augue. Curabitur
ullamcorper ultricies nisi. Nam eget dui.


Chapter
-----------------------------------
Lorem ipsum dolor sit amet, consectetuer adipiscing elit. Aenean commodo ligula
eget dolor. Aenean massa. Cum sociis natoque penatibus et magnis dis parturient
montes, nascetur ridiculus mus. Donec quam felis, ultricies nec, pellentesque
eu, pretium quis, sem. Nulla consequat massa quis enim. Donec pede justo,
fringilla vel, aliquet nec, vulputate eget, arcu. In enim justo, rhoncus ut,
imperdiet a, venenatis vitae, justo. Nullam dictum felis eu pede mollis
pretium. Integer tincidunt. Cras dapibus. Vivamus elementum semper nisi.
Aenean vulputate eleifend tellus. Aenean leo ligula, porttitor eu, consequat
vitae, eleifend ac, enim. Aliquam lorem ante, dapibus in, viverra quis,
feugiat a, tellus. Phasellus viverra nulla ut metus varius laoreet. Quisque
rutrum. Aenean imperdiet. Etiam ultricies nisi vel augue. Curabitur
ullamcorper ultricies nisi. Nam eget dui.


# NB code bocks do need some empty lines to render properly

.. code-block:: python

    import straxen
    elife = straxen.get_correction_from_cmt(
        run_id,
        ("elife", "ONLINE", True))


Done exemplary code block


Figures can be added as so (upload to the straxen/docs/figures some svg):

.. image:: figures/online_monitor.svg


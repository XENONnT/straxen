from straxen.misc import TimeWidgets


def test_widgets():
    tw = TimeWidgets()
    wig = tw.create_widgets()
    start, end = tw.get_start_end()

    assert isinstance(start, int) and isinstance(end, int), "Should have returned unix time in ns as integer!"
    assert end > start, "By default end should be larger than start"

    # Now manually change time zone and compare:
    wig.children[0].children[0].value = 1
    start_utc, end_utc = tw.get_start_end()

    h_in_ns_unix = 60*60*10**9
    unix_conversion_worked = start_utc - start == h_in_ns_unix or start_utc - start == 2 * h_in_ns_unix
    assert unix_conversion_worked
    unix_conversion_worked = start_utc - end == h_in_ns_unix or start_utc - end == 2 * h_in_ns_unix
    assert unix_conversion_worked


def test_change_in_fields():
    tw = TimeWidgets()
    wig = tw.create_widgets()
    start, end = tw.get_start_end()

    # Modify the nano-second value:
    wig.children[1].children[2].value = '20'
    wig.children[2].children[2].value = '20'

    start20, end20 = tw.get_start_end()
    assert start20 - start == 20, 'Start nano-second field did not update.'
    assert end20 - end == 20, 'End nano-second field did not update.'

    # Modify Minutes:
    time = wig.children[1].children[1].value
    minutes = int(time[-2:])
    minutes *= 60*10**9
    wig.children[1].children[1].value = time[:-2] + '00'  # .value is a string "HH:MM"

    start00, _ = tw.get_start_end()
    assert start20 - start00 == minutes, 'Time field did not update its value!'

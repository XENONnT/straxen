from straxen.misc import time_widgets

def test_widgets():
    tw = time_widgets()
    wig = tw.create_widgets()
    start, end = tw.get_start_end()

    assert isinstance(start, int) and isinstance(end, int), "Should have returned unix time in ns as integer!"
    assert end > start, "By default end should be larger than start"

    # Now manually change time zone and compare:
    wig.children[0].value = 1
    start_utc, end_utc = tw.get_start_end()

    h_in_ns_unix = 60*60*10**9
    unix_conversion_worked = start_utc - start == h_in_ns_unix or start_utc - start == 2* h_in_ns_unix
    assert unix_conversion_worked
    unix_conversion_worked = start_utc - end == h_in_ns_unix or start_utc - end == 2 * h_in_ns_unix
    assert unix_conversion_worked
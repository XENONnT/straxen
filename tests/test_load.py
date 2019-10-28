import nEXO_strax as st
import tempfile
import os

with tempfile.TemporaryDirectory() as temp_dir:
    print("Temporary directory is ", temp_dir)
    os.chdir(temp_dir)


    con = st.contexts.MC_test()
    con.search_field('energy')
    print(con.data_info('nest_hits'))

    con.get_array('1','photons')
    con.get_array('1','test_consumer')
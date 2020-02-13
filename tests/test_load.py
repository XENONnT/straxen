import nEXO_strax as st
import tempfile
import os

with tempfile.TemporaryDirectory() as temp_dir:
    print("Temporary directory is ", temp_dir)
    os.chdir(temp_dir)


    con = st.contexts.MC_test()

    MCfactory = st.nEXO_MC_reader.MCreader_factory()

    con.register(MCfactory.make_MCreader('U','/home/brodsky3/nexo/mc_build/test2.root',10,1))
    con.register(MCfactory.make_MCreader('Th', '/home/brodsky3/nexo/mc_build/test2.root', 100, 1))
    con.register(MCfactory.make_MCmerger())
    con.register(st.nEXO_MC_reader.MCReader_test_consumer)

    con.search_field('energy')
    print(con.data_info('nest_hits'))

    con.get_array('1','photons')
    con.get_array('1','test_consumer')


# con = st.contexts.MC_test()
# con.search_field('energy')
# print(con.data_info('nest_hits'))
#
# con.get_array('1','photons')
# con.get_array('1','test_consumer')
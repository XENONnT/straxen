import unittest
import straxen
import os
import pymongo


def mongo_uri_not_set():
    return 'TEST_MONGO_URI' not in os.environ


class TestMongoDownloader(unittest.TestCase):
    """
    Test the saving behavior of the context with the mogno downloader

    Requires write access to some pymongo server, the URI of witch is to be set
    as an environment variable under:

        TEST_MONGO_URI

    At the moment this is just an empty database but you can also use some free
    ATLAS mongo server.
    """
    _run_test = True

    def setUp(self):
        # Just to make sure we are running some mongo server, see test-class docstring
        if 'TEST_MONGO_URI' not in os.environ:
            self._run_test = False
            return
        uri = os.environ.get('TEST_MONGO_URI')
        db_name = 'test_rundb'
        collection_name = 'fs.files'
        client = pymongo.MongoClient(uri)
        database = client[db_name]
        collection = database[collection_name]
        self.downloader = straxen.MongoDownloader(collection=collection,
                                                  readonly=True,
                                                  file_database=None,
                                                  _test_on_init=False,
                                                  )
        self.uploader = straxen.MongoUploader(collection=collection,
                                              readonly=False,
                                              file_database=None,
                                              _test_on_init=False,
                                              )
        self.collection = collection

    @unittest.skipIf(mongo_uri_not_set(), "No access to test database")
    def tearDown(self):
        self.collection.drop()

    @unittest.skipIf(mongo_uri_not_set(), "No access to test database")
    def test_upload(self):
        with self.assertRaises(ConnectionError):
            # Should be empty!
            self.downloader.test_find()
        file_name = 'test.txt'
        file_content = 'This is a test'
        with open(file_name, 'w') as f:
            f.write(file_content)
        assert os.path.exists(file_name)
        self.uploader.upload_from_dict({file_name: os.path.abspath(file_name)})
        assert self.uploader.md5_stored(file_name)
        assert self.downloader.config_exists(file_name)
        download_path = self.downloader.download_single(file_name)
        assert os.path.exists(download_path)
        read_file = straxen.get_resource(download_path)
        assert file_content == read_file
        os.remove(file_name)
        assert not os.path.exists(file_name)
        self.downloader.test_find()

        with self.assertRaises(NotImplementedError):
            self.downloader.download_all()

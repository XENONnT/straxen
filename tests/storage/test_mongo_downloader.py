import unittest

import utilix
import straxen
import utilix
import os
import pymongo
from straxen import mongo_uri_not_set


@unittest.skipIf(mongo_uri_not_set(), "No access to test database")
class TestMongoDownloader(unittest.TestCase):
    """Test the saving behavior of the context with the mogno downloader.

    Requires write access to some pymongo server, the URI of witch is to be set
    as an environment variable under:

        TEST_MONGO_URI

    At the moment this is just an empty database but you can also use some free
    ATLAS mongo server.

    """

    _run_test = True

    def setUp(self):
        # Just to make sure we are running some mongo server, see test-class docstring
        if "TEST_MONGO_URI" not in os.environ:
            self._run_test = False
            return
        uri = os.environ.get("TEST_MONGO_URI")
        db_name = "test_rundb"
        collection_name = "fs.files"
        client = pymongo.MongoClient(uri)
        database = client[db_name]
        collection = database[collection_name]
        self.downloader = straxen.MongoDownloader(
            collection=collection,
            readonly=True,
            file_database=None,
            _test_on_init=False,
        )
        self.uploader = utilix.mongo_storage.MongoUploader(
            collection=collection,
            readonly=False,
            file_database=None,
            _test_on_init=False,
        )
        self.collection = collection

    def tearDown(self):
        self.collection.drop()

    def test_up_and_download(self):
        with self.assertRaises(ConnectionError):
            # Should be empty!
            self.downloader.test_find()
        file_name = "test.txt"
        self.assertFalse(self.downloader.md5_stored(file_name))
        self.assertEqual(self.downloader.compute_md5(file_name), "")
        file_content = "This is a test"
        with open(file_name, "w") as f:
            f.write(file_content)
        self.assertTrue(os.path.exists(file_name))
        self.uploader.upload_from_dict({file_name: os.path.abspath(file_name)})
        self.assertTrue(self.uploader.md5_stored(file_name))
        self.assertTrue(self.downloader.config_exists(file_name))
        path = self.downloader.download_single(file_name)
        path_hr = self.downloader.download_single(file_name, human_readable_file_name=True)
        abs_path = self.downloader.get_abs_path(file_name)

        for p in [path, path_hr, abs_path]:
            self.assertTrue(os.path.exists(p))
        read_file = straxen.get_resource(path)
        self.assertTrue(file_content == read_file)
        os.remove(file_name)
        self.assertFalse(os.path.exists(file_name))
        self.downloader.test_find()
        self.downloader.download_all()
        # Now the test on init should work, let's double try
        straxen.MongoDownloader(
            collection=self.collection,
            file_database=None,
            _test_on_init=True,
        )

    def test_invalid_methods(self):
        """The following examples should NOT work, let's make sure the right errors are raised."""
        with self.assertRaises(ValueError):
            straxen.MongoDownloader(
                collection=self.collection,
                file_database="NOT NONE",
            )
        with self.assertRaises(ValueError):
            straxen.MongoDownloader(
                collection="invalid type",
            )
        with self.assertRaises(PermissionError):
            utilix.mongo_storage.MongoUploader(readonly=True)

        with self.assertRaises(ValueError):
            self.uploader.upload_from_dict("A string is not a dict")

        with self.assertRaises(utilix.mongo_storage.CouldNotLoadError):
            self.uploader.upload_single("no_such_file", "no_such_file")

        with self.assertWarns(UserWarning):
            self.uploader.upload_from_dict({"something": "no_such_file"})

        with self.assertRaises(ValueError):
            straxen.MongoDownloader(
                collection=self.collection,
                file_database=None,
                _test_on_init=False,
                store_files_at=False,
            )
        with self.assertRaises(ValueError):
            self.downloader.download_single("no_existing_file")

        with self.assertRaises(ValueError):
            self.downloader._check_store_files_at("some_str")

        with self.assertRaises(PermissionError):
            self.downloader._check_store_files_at([])

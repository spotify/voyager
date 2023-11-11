import os
from voyager import Index, Space
from voyager.stringindex import StringIndex

INDEX_FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "indices")

def test_stringindex_creation():
    index = StringIndex()
import io
import struct

import numpy as np
import pytest

from voyager import Index, Space, StorageDataType, StringIndex


@pytest.mark.parametrize("storage", list(StorageDataType))
def test_updates_and_round_trip(storage, tmp_path):
    index = StringIndex(Space.Euclidean, 2, storage_data_type=storage)
    names = ["", "café", "東京🛰️\0end"]
    index.add_items(names, [[0.0, 0.0], [0.5, 0.0], [0.0, 0.5]])
    index.add_item(names[1], np.array([0.75, 0.0], dtype=np.float32))
    index.add_items([names[1], names[1]], [[0.5, 0.0], [1.0, 0.0]])
    assert len(index) == 3
    np.testing.assert_allclose(index.get_vector(names[1]), [1.0, 0.0])
    assert index.query([1.0, 0.0])[0] == [names[1]]
    assert index.names == names
    assert names[2] in index
    assert "absent" not in index

    filename = str(tmp_path / "strings.voyager")
    index.save(filename)
    stream = io.BytesIO()
    index.save(stream)
    assert stream.getvalue() == index.as_bytes()
    assert struct.unpack_from("=i", stream.getvalue(), 4) == (2,)
    for loaded in [
        StringIndex.load(filename),
        StringIndex.load(io.BytesIO(stream.getvalue())),
    ]:
        assert loaded.names == names
        loaded.add_item(names[1], [0.5, 0.5])
        loaded.add_item("new", [1.0, 1.0])
        assert len(loaded) == 4
        assert loaded.query([0.5, 0.5])[0] == [names[1]]
        loaded.mark_deleted(names[1])
        assert names[1] not in loaded.query([0.5, 0.5], k=3)[0]
        loaded.unmark_deleted(names[1])
        assert loaded.query([0.5, 0.5])[0] == [names[1]]


def test_empty_and_failed_inserts():
    index = StringIndex(Space.Euclidean, 2)
    index.add_items([], [])
    loaded_empty = StringIndex.load(io.BytesIO(index.as_bytes()))
    assert len(loaded_empty) == 0
    loaded_empty.add_item("first", [0.0, 0.0])
    assert loaded_empty.query([0.0, 0.0])[0] == ["first"]
    with pytest.raises(Exception, match="dimensions"):
        index.add_item("bad", [1.0])
    assert "bad" not in index
    assert len(index) == 0
    with pytest.raises(Exception, match="length"):
        index.add_items(["bad"], [])
    with pytest.raises(Exception, match="dimensionality"):
        index.add_items(["a", "bad"], [[0.0, 0.0], [0.0]])
    assert len(index) == 0
    with pytest.raises(Exception, match="Unknown string identifier"):
        index.get_vector("absent")
    index.add_item("good", [0.0, 0.0])
    assert index.names == ["good"]


def test_invalid_format_and_mapping():
    index = StringIndex(Space.Euclidean, 2)
    index.add_item("a", [0.0, 0.0])
    data = bytearray(index.as_bytes())
    unknown = bytearray(data)
    struct.pack_into("=i", unknown, 4, 19)
    with pytest.raises(Exception, match="newer version"):
        Index.load(io.BytesIO(unknown))
    # Format 2: 19-byte v1 header, kind, count, label, length, UTF-8, graph.
    assert data[19] == 1
    for end in (20, 24, 34, 44):
        with pytest.raises(Exception):
            StringIndex.load(io.BytesIO(data[:end]))
    struct.pack_into("=Q", data, 28, 999)
    with pytest.raises(Exception, match="unknown graph label"):
        StringIndex.load(io.BytesIO(data))


def test_numeric_v3_format():
    index = Index(Space.Euclidean, 2)
    index.add_item([0.0, 0.0], 123)
    data = index.as_bytes()
    assert struct.unpack_from("=i", data, 4) == (2,)
    loaded = Index.load(io.BytesIO(data))
    np.testing.assert_allclose(loaded.get_vector(123), [0.0, 0.0])
    with pytest.raises(Exception, match="numeric identifiers"):
        StringIndex.load(io.BytesIO(data))


def test_batches_and_legacy_mapping():
    numeric = Index(Space.Euclidean, 2)
    numeric.add_items(np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32), [0, 1])
    index = StringIndex.from_index(numeric, ["a", "b"])
    with pytest.raises(Exception, match="string identifiers"):
        numeric.add_item([0.0, 0.0], 2)
    index.add_items(["a", "c"], np.array([[0.5, 0.5], [0.0, 1.0]], dtype=np.float32))
    names, distances = index.query(np.array([[0.5, 0.5], [0.0, 1.0]], dtype=np.float32))
    assert names == [["a"], ["c"]]
    np.testing.assert_allclose(distances, [[0], [0]])
    np.testing.assert_allclose(index.get_vectors(["a", "b"]), [[0.5, 0.5], [1, 1]])
    with pytest.raises(Exception, match="Duplicate"):
        StringIndex.from_index(numeric, ["x", "x", "z"])
    assert index.names == ["a", "b", "c"]


def test_parallel_batch_and_failed_quantization():
    index = StringIndex(Space.Euclidean, 2)
    names = [str(i) for i in range(100)]
    index.add_items(names, [[float(i), 0.0] for i in range(100)], num_threads=4)
    index.add_items(names, [[float(i), 1.0] for i in range(100)], num_threads=4)
    assert len(index) == 100
    assert index.query([57.0, 1.0])[0] == ["57"]
    quantized = StringIndex(
        Space.Euclidean, 2, storage_data_type=StorageDataType.Float8
    )
    with pytest.raises(Exception):
        quantized.add_items(["good", "bad"], [[0.5, 0.5], [10.0, 0.0]])
    assert "bad" not in quantized
    loaded = StringIndex.load(io.BytesIO(quantized.as_bytes()))
    loaded.add_item("next", [1.0, 1.0])
    assert "next" in loaded


def test_java_interoperability(tmp_path):
    import os
    import subprocess

    classpath = os.environ.get("VOYAGER_JAVA_TEST_CLASSPATH")
    if not classpath:
        pytest.skip(
            "Set VOYAGER_JAVA_TEST_CLASSPATH to java/target/classes:java/target/test-classes"
        )
    java = os.environ.get("VOYAGER_JAVA", "java")
    command = [java, "-cp", classpath, "com.spotify.voyager.jni.StringIndexInterop"]
    filename = str(tmp_path / "shared.voy")
    subprocess.run(command + ["write", filename], check=True)
    index = StringIndex.load(filename)
    name = "東京🛰️\0café"
    assert index.names == [name, ""]
    index.add_item(name, [0.5, 0.5])
    index.add_item("python", [0.0, 1.0])
    index.save(filename)
    subprocess.run(command + ["read", filename], check=True)

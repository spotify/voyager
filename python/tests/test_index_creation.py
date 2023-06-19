from io import BytesIO

import pytest

import numpy as np

import voyager


@pytest.mark.parametrize(
    "num_dimensions,num_elements",
    [
        (4, 1_024),
        (16, 1_024),
        (128, 512),
        (256, 256),
        (4096, 128),
    ],
)
@pytest.mark.parametrize(
    "space",
    [voyager.Space.Euclidean, voyager.Space.Cosine],
    ids=lambda x: x.name if hasattr(x, "name") else str(x),
)
@pytest.mark.parametrize(
    "storage_data_type,distance_tolerance,recall_tolerance",
    [
        (voyager.StorageDataType.E4M3, 0.03, 0.4),
        (voyager.StorageDataType.Float8, 0.03, 0.5),
        (voyager.StorageDataType.Float32, 2e-7, 1.0),
    ],
    ids=lambda x: x.name if hasattr(x, "name") else str(x),
)
def test_create_and_query(
    num_dimensions: int,
    num_elements: int,
    space: voyager.Space,
    storage_data_type: voyager.StorageDataType,
    distance_tolerance: float,
    recall_tolerance: float,
    tmp_path,
):
    input_data = np.random.random((num_elements, num_dimensions)).astype(np.float32) * 2 - 1
    if storage_data_type == voyager.StorageDataType.Float8:
        input_data = np.round(input_data * 127) / 127

    ids = list(range(len(input_data)))

    index = voyager.Index(
        space=space,
        num_dimensions=num_dimensions,
        ef_construction=num_elements,
        M=20,
        storage_data_type=storage_data_type,
    )

    assert str(space).split(".")[-1] in repr(index)
    assert str(num_dimensions) in repr(index)
    assert str(storage_data_type).split(".")[-1] in repr(index)

    index.ef = num_elements

    assert index.add_items(input_data, ids) == ids

    assert len(index) == num_elements
    assert len(index) == index.num_elements

    labels, distances = index.query(input_data, k=1)
    matches = np.sum(labels[:, 0] == np.arange(len(input_data)))
    assert matches / len(input_data) >= recall_tolerance
    np.testing.assert_allclose(
        distances[:, 0],
        np.zeros(len(input_data)),
        atol=(distance_tolerance * num_dimensions),
    )

    np.testing.assert_equal(set(ids), set(index.ids))

    # Test the single-query interface too:
    for i, vector in enumerate(input_data):
        labels, distances = index.query(vector, k=1)
        if storage_data_type != voyager.StorageDataType.Float32:
            pass
        else:
            assert labels[0] == i
        assert distances[0] < (distance_tolerance * num_dimensions)

    output_file = tmp_path / "index.voy"
    index.save(str(output_file))
    assert output_file.stat().st_size > 0
    assert len(index.as_bytes()) > 0
    assert len(bytes(index)) > 0
    assert index.as_bytes() == output_file.read_bytes()

    with BytesIO() as f:
        index.save(f)
        assert index.as_bytes() == f.getvalue()


@pytest.mark.parametrize(
    "space,expected_distances",
    [
        (voyager.Space.Euclidean, [[0.0, 1.0, 2.0, 2.0, 2.0]]),
        (voyager.Space.InnerProduct, [[-2.0, -1.0, 0.0, 0.0, 0.0]]),
        (voyager.Space.Cosine, [[0, 1.835e-01, 4.23e-01, 4.23e-01, 4.23e-01]]),
    ],
)
@pytest.mark.parametrize("right_dimension", list(range(1, 128, 3)))
@pytest.mark.parametrize("left_dimension", list(range(1, 32, 5)))
def test_spaces(space, expected_distances, left_dimension, right_dimension):
    input_data = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
        ]
    )
    data2 = np.concatenate(
        [
            np.zeros([input_data.shape[0], left_dimension]),
            input_data,
            np.zeros([input_data.shape[0], right_dimension]),
        ],
        axis=1,
    )

    num_dimensions = data2.shape[1]
    index = voyager.Index(space=space, num_dimensions=num_dimensions, ef_construction=100, M=16)
    index.ef = 10
    index.add_items(data2)

    _labels, distances = index.query(np.asarray(data2[-1:]), k=5)
    np.testing.assert_allclose(distances, expected_distances, atol=1e-3)


@pytest.mark.parametrize(
    "num_dimensions,num_elements",
    [
        (4, 1_024),
        (16, 1_024),
        (128, 512),
        (256, 256),
        (4096, 128),
    ],
)
@pytest.mark.parametrize(
    "space",
    [
        voyager.Space.Euclidean,
        voyager.Space.InnerProduct,
        # Note: Cosine is not included here, as vectors are normalized when stored in the index.
    ],
)
def test_get_vectors(num_dimensions: int, num_elements: int, space):
    input_data = np.random.random((num_elements, num_dimensions)).astype(np.float32) * 2 - 1
    index = voyager.Index(space=space, num_dimensions=num_dimensions)

    labels = list(range(num_elements))

    # Before adding anything, getting any labels should fail
    with pytest.raises(RuntimeError):
        index.get_vectors(labels)

    index.add_items(input_data, labels)

    with pytest.raises(TypeError):
        index.get_vectors(labels[0])

    for expected_vector, label in zip(input_data, labels):
        np.testing.assert_equal(index.get_vector(label), expected_vector)

    np.testing.assert_equal(index.get_vectors(labels), input_data)


@pytest.mark.parametrize(
    "num_dimensions,num_elements",
    [
        (4, 1_024),
        (16, 1_024),
        (128, 512),
        (256, 256),
        (4096, 128),
    ],
)
@pytest.mark.parametrize("space", [voyager.Space.Euclidean, voyager.Space.Cosine])
@pytest.mark.parametrize(
    "storage_data_type",
    [voyager.StorageDataType.Float8, voyager.StorageDataType.Float32],
)
def test_load_from_file_buffer(
    num_dimensions: int,
    num_elements: int,
    space: voyager.Space,
    storage_data_type: voyager.StorageDataType,
):
    input_data = np.random.random((num_elements, num_dimensions)).astype(np.float32) * 2 - 1
    if storage_data_type == voyager.StorageDataType.Float8:
        input_data = np.round(input_data * 127) / 127

    index = voyager.Index(
        space=space,
        num_dimensions=num_dimensions,
        ef_construction=num_elements,
        M=20,
        storage_data_type=storage_data_type,
    )

    index.add_items(input_data)
    with BytesIO(index.as_bytes()) as f:
        reloaded = voyager.Index.load(
            file_like=f,
            space=space,
            num_dimensions=num_dimensions,
            storage_data_type=storage_data_type,
        )

    labels = list(range(num_elements))
    np.testing.assert_equal(index.get_vectors(labels), reloaded.get_vectors(labels))


@pytest.mark.parametrize(
    "num_dimensions,num_elements",
    [
        (4, 1_024),
        (16, 1_024),
        (128, 512),
        (256, 256),
        (4096, 128),
    ],
)
@pytest.mark.parametrize("space", [voyager.Space.Euclidean, voyager.Space.Cosine])
@pytest.mark.parametrize(
    "initial_data_type,reloaded_data_type",
    [
        (voyager.StorageDataType.Float8, voyager.StorageDataType.Float32),
        (voyager.StorageDataType.Float32, voyager.StorageDataType.Float8),
    ],
)
def test_load_incorrect_type(
    num_dimensions: int,
    num_elements: int,
    space: voyager.Space,
    initial_data_type: voyager.StorageDataType,
    reloaded_data_type: voyager.StorageDataType,
):
    """
    Ensure that if loading a Float8 index as a Float32 index (or vice versa)
    Voyager catches the issue.
    """
    input_data = np.random.random((num_elements, num_dimensions)).astype(np.float32) * 2 - 1
    if initial_data_type == voyager.StorageDataType.Float8:
        input_data = np.round(input_data * 127) / 127

    index = voyager.Index(
        space=space,
        num_dimensions=num_dimensions,
        ef_construction=num_elements,
        M=20,
        storage_data_type=initial_data_type,
    )

    index.add_items(input_data)
    with BytesIO(index.as_bytes()) as f:
        with pytest.raises(ValueError):
            voyager.Index.load(
                file_like=f,
                space=space,
                num_dimensions=num_dimensions,
                storage_data_type=reloaded_data_type,
            )


@pytest.mark.parametrize("space", [voyager.Space.Euclidean, voyager.Space.Cosine])
@pytest.mark.parametrize("query_ef,rank_tolerance", [(1, 500), (2, 75), (100, 1)])
def test_query_ef(space: voyager.Space, query_ef: int, rank_tolerance: int):
    num_dimensions = 32
    num_elements = 1_000
    input_data = np.random.random((num_elements, num_dimensions)).astype(np.float32) * 2 - 1

    index = voyager.Index(
        space=space, num_dimensions=num_dimensions, ef_construction=num_elements, M=20
    )

    index.ef = num_elements
    index.add_items(input_data)

    # Query with a high query_ef to get the "correct" results
    closest_labels_per_vector, _ = index.query(input_data, k=num_elements, query_ef=num_elements)

    labels, _ = index.query(input_data, k=1, query_ef=query_ef)
    for vector_index, returned_labels in enumerate(labels):
        returned_label = returned_labels[0]
        actual_rank = list(closest_labels_per_vector[vector_index]).index(returned_label)
        assert actual_rank < rank_tolerance

    # Test the single-query interface too:
    for vector_index, vector in enumerate(input_data):
        returned_labels, _ = index.query(vector, k=1, query_ef=query_ef)
        returned_label = returned_labels[0]
        actual_rank = list(closest_labels_per_vector[vector_index]).index(returned_label)
        assert actual_rank < rank_tolerance


@pytest.mark.parametrize("num_dimensions", [4])
@pytest.mark.parametrize("num_elements", [100, 1_000])
@pytest.mark.parametrize("space", [voyager.Space.Euclidean, voyager.Space.InnerProduct])
@pytest.mark.parametrize(
    "storage_data_type,tolerance",
    [
        (voyager.StorageDataType.Float8, 0.01),
        (voyager.StorageDataType.Float32, 0.01),
        (voyager.StorageDataType.E4M3, 0.075),
    ],
)
def test_add_single_item(
    num_dimensions: int,
    num_elements: int,
    space: voyager.Space,
    storage_data_type: voyager.StorageDataType,
    tolerance: float,
):
    np.random.seed(123)
    input_data = np.random.random((num_elements, num_dimensions)).astype(np.float32) * 2 - 1
    index = voyager.Index(
        space=space,
        num_dimensions=num_dimensions,
        storage_data_type=storage_data_type,
    )

    labels = list(range(num_elements))

    # Before adding anything, getting any labels should fail
    with pytest.raises(RuntimeError):
        index.get_vectors(labels)

    for label, item in zip(labels, input_data):
        index.add_item(item, label)

    with pytest.raises(TypeError):
        index.get_vectors(labels[0])

    for expected_vector, label in zip(input_data, labels):
        np.testing.assert_allclose(index.get_vector(label), expected_vector, atol=tolerance)

    np.testing.assert_allclose(index.get_vectors(labels), input_data, atol=tolerance)

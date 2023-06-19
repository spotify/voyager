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
    "storage_data_type",
    [voyager.StorageDataType.E4M3, voyager.StorageDataType.Float8, voyager.StorageDataType.Float32],
    ids=lambda x: x.name if hasattr(x, "name") else str(x),
)
def test_recreate_index(
    num_dimensions: int,
    num_elements: int,
    space: voyager.Space,
    storage_data_type: voyager.StorageDataType,
):
    input_data = np.random.random((num_elements, num_dimensions)).astype(np.float32) * 2 - 1

    index = voyager.Index(
        space=space,
        num_dimensions=num_dimensions,
        ef_construction=num_elements,
        M=20,
        storage_data_type=storage_data_type,
    )

    ids = index.add_items(input_data)
    assert sorted(ids) == sorted(index.ids)

    recreated = voyager.Index(
        index.space,
        index.num_dimensions,
        index.M,
        index.ef_construction,
        max_elements=len(index),
        storage_data_type=index.storage_data_type,
    )
    ordered_ids = list(index.ids)
    recreated.add_items(index.get_vectors(ordered_ids), ordered_ids)

    for _id in ids:
        assert _id in index
        assert _id in recreated
        np.testing.assert_allclose(index[_id], recreated[_id], 0.08, 1e-2)

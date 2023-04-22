import pytest

import numpy as np

from voyager import E4M3T, Index, Space, StorageDataType


RANGES_AND_EXPECTED_ERRORS = [
    # (Min value, max value, step), expected maximum error
    ((-448, 448, 1), 16),
    ((-10, 10, 1e-1), 0.500000001),
    ((-1, 1, 1e-2), 0.0500000001),
    ((-0.01, 0.01, 1e-4), 0.009),
]

VALID_E4M3_VALUES = set([float(E4M3T.from_char(x)) for x in range(256)])


@pytest.mark.parametrize(
    "_input,expected_error",
    [
        (v, error)
        for (_min, _max, step), error in RANGES_AND_EXPECTED_ERRORS
        for v in np.arange(_min, _max, step)
    ],
)
def test_range(_input: float, expected_error: float):
    v = E4M3T(_input)
    roundtrip_value = float(v)
    assert not np.isnan(roundtrip_value)
    error = abs(roundtrip_value - _input)
    assert error <= expected_error, repr(v)
    if _input < 0 and roundtrip_value != 0.0:
        assert roundtrip_value < 0
    elif _input > 0 and roundtrip_value != 0.0:
        assert roundtrip_value > 0


@pytest.mark.parametrize("_input", list(range(256)))
def test_rounding_exact(_input: int):
    expected = E4M3T.from_char(_input)
    converted = E4M3T(float(expected))
    actual = float(converted)
    if np.isnan(float(expected)):
        assert np.isnan(actual)
    else:
        assert actual == float(expected), f"Expected {expected}, but got {converted}"


@pytest.mark.parametrize("_input", list(range(256)))
@pytest.mark.parametrize("offset", [-1e-6, 1e-6])
def test_rounding_near_actual_values(_input: int, offset: float):
    expected = E4M3T.from_char(_input)
    converted = E4M3T(float(expected) + offset)
    actual = float(converted)
    if np.isnan(float(expected)):
        assert np.isnan(actual)
    else:
        assert actual == float(expected), (
            f"\nProvided {float(expected)} + {offset} ="
            f" {float(expected) + offset}\nExpected {expected},\nbut got  {converted}"
        )


@pytest.mark.parametrize(
    "_input",
    list(np.arange(-1.1, 1.1, 0.001))
    + [(2**-7 * (1 + x / 8)) for x in range(8)]
    + list(np.arange(-448, 448, 1.0))
    + [0.04890749],
)
def test_rounding(_input: float):
    closest_above = min(
        VALID_E4M3_VALUES, key=lambda v: (abs(v - _input) if v >= _input else 10000)
    )
    closest_below = min(
        VALID_E4M3_VALUES, key=lambda v: (abs(v - _input) if v <= _input else 10000)
    )
    expected = min([closest_above, closest_below], key=lambda v: abs(v - _input))
    if closest_above != closest_below and abs(closest_above - _input) == abs(
        closest_below - _input
    ):
        # Round to nearest, ties to even:
        above_is_even = E4M3T(closest_above).mantissa % 2 == 0
        below_is_even = E4M3T(closest_below).mantissa % 2 == 0
        if above_is_even and below_is_even:
            raise NotImplementedError(
                "Both numbers above and below the target are even!"
                f" {E4M3T(closest_above)} vs {E4M3T(closest_below)}"
            )
        elif above_is_even:
            expected = closest_above
        else:
            expected = closest_below
    converted = E4M3T(_input)
    actual = float(converted)
    assert actual == expected, (
        f"Expected {_input} to round to {expected} ({E4M3T(expected)}) when converting"
        f" to E4M3 but found {actual} ({converted}) (closest above option was"
        f" {closest_above}, closest below option was {closest_below})"
    )


@pytest.mark.parametrize("_input", [0.04890749])
def test_rounding_known_edge_cases(_input: float):
    closest_above = min(
        VALID_E4M3_VALUES, key=lambda v: (abs(v - _input) if v >= _input else 10000)
    )
    closest_below = min(
        VALID_E4M3_VALUES, key=lambda v: (abs(v - _input) if v <= _input else 10000)
    )
    expected = min([closest_above, closest_below], key=lambda v: abs(v - _input))
    if closest_above != closest_below and abs(closest_above - _input) == abs(
        closest_below - _input
    ):
        # Round to nearest, ties to even:
        above_is_even = E4M3T(closest_above).mantissa % 2 == 0
        below_is_even = E4M3T(closest_below).mantissa % 2 == 0
        if above_is_even and below_is_even:
            raise NotImplementedError(
                "Both numbers above and below the target are even!"
                f" {E4M3T(closest_above)} vs {E4M3T(closest_below)}"
            )
        elif above_is_even:
            expected = closest_above
        else:
            expected = closest_below
    converted = E4M3T(_input)
    actual = float(converted)
    assert actual == expected, (
        f"Expected {_input} to round to {expected} ({E4M3T(expected)}) when converting"
        f" to E4M3 but found {actual} ({converted}) (closest above option was"
        f" {closest_above}, closest below option was {closest_below})"
    )


def test_size():
    assert E4M3T(1.2345).size == 1


def test_nan():
    assert np.isnan(float(E4M3T(np.nan)))


@pytest.mark.parametrize("_input", [-123456, -449, 449, 123456])
def test_out_of_range(_input: float):
    with pytest.raises(ValueError):
        E4M3T(_input)


@pytest.mark.parametrize("a,b", [(a, a + 1e-2) for a in np.arange(-448, 448, 1e-2)])
def test_monotonically_increasing(a: float, b: float):
    assert float(E4M3T(a)) <= float(E4M3T(b))


def normalized(vec: np.ndarray) -> np.ndarray:
    return np.array(vec).astype(np.float32) / (
        np.sqrt(
            np.sum(
                np.power(np.array(vec).astype(np.float32), 2).astype(np.float32)
            ).astype(np.float32)
        ).astype(np.float32)
        + 1e-30
    ).astype(np.float32)


def test_cosine():
    REAL_WORLD_VECTOR = [
        -0.28728199005126953,
        -0.4670010209083557,
        0.2676819860935211,
        -0.1626259982585907,
        -0.6251270174980164,
        0.2816449999809265,
        0.32270801067352295,
        0.33403000235557556,
        -0.7520139813423157,
        0.5022000074386597,
        0.7720339894294739,
        -0.5909199714660645,
        0.5918650031089783,
        -0.15842899680137634,
        -0.11246500164270401,
        0.24038001894950867,
        -1.157925009727478,
        -0.16482099890708923,
        0.09613300859928131,
        0.5384849905967712,
        0.17511099576950073,
        0.09210799634456635,
        -0.2158990055322647,
        -0.1197270005941391,
        -0.5386099815368652,
        0.196150004863739,
        -0.8914260864257812,
        -0.19836701452732086,
        0.3211739957332611,
        0.33692699670791626,
        0.620635986328125,
        -0.8655009865760803,
        -0.2893890142440796,
        0.2558070123195648,
        -0.0019950000569224358,
        0.25856301188468933,
        -0.831616997718811,
        1.3858330249786377,
        -0.5884850025177002,
        -0.24664302170276642,
        0.00035700001171790063,
        0.8199999928474426,
        -0.1729460060596466,
        0.6167529821395874,
        0.1001340001821518,
        0.2342749983072281,
        0.47478801012039185,
        0.6487500071525574,
        0.3548029959201813,
        0.2365729957818985,
        -0.713392972946167,
        -0.9608209729194641,
        -0.09217199683189392,
        -0.0563880018889904,
        -0.022280000150203705,
        -0.3831019997596741,
        -0.10219399631023407,
        -0.1772879958152771,
        -0.2045920193195343,
        -0.5201849937438965,
        -1.6222929954528809,
        0.7166309952735901,
        -0.3722609877586365,
        -0.4575370252132416,
        0.5124289989471436,
        0.02841399982571602,
        0.06806100159883499,
        -0.2725119888782501,
        -0.5817689895629883,
        -0.2708030045032501,
        1.121297001838684,
        -0.639868974685669,
        0.39189401268959045,
        -0.1527390033006668,
        0.6738319993019104,
        -0.7513130307197571,
        -0.23471000790596008,
        -0.8855159878730774,
        0.7264220118522644,
        0.4370560348033905,
    ]
    index = Index(
        Space.Cosine, num_dimensions=80, storage_data_type=StorageDataType.E4M3
    )
    index.add_item(REAL_WORLD_VECTOR)
    normalized_vector = normalized(REAL_WORLD_VECTOR)
    expected = np.array([float(E4M3T(x)) for x in normalized_vector])
    actual = index.get_vector(0)
    mismatch_indices = [i for i, (a, b) in enumerate(zip(expected, actual)) if a != b]
    np.testing.assert_allclose(
        expected,
        actual,
        err_msg=(
            f"Got mismatches at indices: {mismatch_indices}:\n\tExpected:"
            f" {expected[mismatch_indices]}\n\tGot: "
            f" \t{actual[mismatch_indices]}\n\tOriginal value(s):"
            f" {normalized_vector[mismatch_indices]}"
        ),
    )

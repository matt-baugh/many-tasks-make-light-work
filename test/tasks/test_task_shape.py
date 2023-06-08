import pytest
import numpy as np

from multitask_method.tasks.task_shape import intersect_aligned_hyperellipse_edge, \
    intersect_to_aligned_hyperrectangle_edge


@pytest.fixture
def setup_shape_test_params():
    shape = np.array([100, 50])

    origin00 = np.array([[0], [0]])

    simple_directions = [
        np.array([[1], [0]]),
        np.array([[-1], [0]]),
        np.array([[0], [1]]),
        np.array([[0], [-1]])
    ]

    origin01 = np.array([[20], [-10]])

    return shape, origin00, simple_directions, origin01


def test_simple_rectangle_intersections(setup_shape_test_params):
    shape, origin00, simple_directions, origin01 = setup_shape_test_params

    simple_directions_results0 = [
        np.array([[50], [0]]),
        np.array([[-50], [0]]),
        np.array([[0], [25]]),
        np.array([[0], [-25]])
    ]

    for i, (d, r) in enumerate(zip(simple_directions, simple_directions_results0)):
        assert np.array_equal(intersect_to_aligned_hyperrectangle_edge(shape, origin00, d), r)

    all_simple_origins0 = np.concatenate([origin00] * len(simple_directions), axis=1)
    all_simple_directions0 = np.concatenate(simple_directions, axis=1)
    all_simple_results0 = np.concatenate(simple_directions_results0, axis=1)

    # Stacked test
    assert np.array_equal(intersect_to_aligned_hyperrectangle_edge(shape,
                                                                   all_simple_origins0,
                                                                   all_simple_directions0),
                          all_simple_results0)
    assert np.array_equal(intersect_to_aligned_hyperrectangle_edge(shape,
                                                                   origin00,
                                                                   all_simple_directions0),
                          all_simple_results0)

    simple_directions_results1 = [
        np.array([[50], [-10]]),
        np.array([[-50], [-10]]),
        np.array([[20], [25]]),
        np.array([[20], [-25]])
    ]
    for i, (d, r) in enumerate(zip(simple_directions, simple_directions_results1)):
        assert np.array_equal(intersect_to_aligned_hyperrectangle_edge(shape, origin01, d), r)

    all_simple_origins1 = np.concatenate([origin01] * len(simple_directions), axis=1)
    all_simple_directions1 = np.copy(all_simple_directions0)
    all_simple_results1 = np.concatenate(simple_directions_results1, axis=1)

    assert np.array_equal(intersect_to_aligned_hyperrectangle_edge(shape,
                                                                   all_simple_origins1,
                                                                   all_simple_directions1),
                          all_simple_results1)
    assert np.array_equal(intersect_to_aligned_hyperrectangle_edge(shape,
                                                                   origin01,
                                                                   all_simple_directions1),
                          all_simple_results1)

    # Test unnormalised directions

    assert np.array_equal(intersect_to_aligned_hyperrectangle_edge(shape,
                                                                   all_simple_origins1,
                                                                   all_simple_directions1 / 2),
                          all_simple_results1)


def test_simple_ellipse_intersections(setup_shape_test_params):
    shape, origin00, simple_directions, origin01 = setup_shape_test_params

    simple_directions_results0 = [
        np.array([[50], [0]]),
        np.array([[-50], [0]]),
        np.array([[0], [25]]),
        np.array([[0], [-25]])
    ]

    for i, (d, r) in enumerate(zip(simple_directions, simple_directions_results0)):
        assert np.array_equal(intersect_aligned_hyperellipse_edge(shape, origin00, d), r)

    all_simple_origins0 = np.concatenate([origin00] * len(simple_directions), axis=1)
    all_simple_directions0 = np.concatenate(simple_directions, axis=1)
    all_simple_results0 = np.concatenate(simple_directions_results0, axis=1)

    # Stacked test
    assert np.array_equal(intersect_aligned_hyperellipse_edge(shape,
                                                              all_simple_origins0,
                                                              all_simple_directions0),
                          all_simple_results0)
    assert np.array_equal(intersect_aligned_hyperellipse_edge(shape,
                                                              origin00,
                                                              all_simple_directions0),
                          all_simple_results0)

    # Test at edge
    edge_coords = all_simple_results0
    assert np.array_equal(intersect_aligned_hyperellipse_edge(shape,
                                                              edge_coords,
                                                              all_simple_directions0),
                          all_simple_results0)


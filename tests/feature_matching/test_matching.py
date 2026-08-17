import numpy as np

from stereoscan.feature_matching.matching import circular_match, match_along_epipolar, match_within_window


def test_match_within_window_finds_nearest_valid_candidate():
    query_points = np.array([[50, 50]])
    query_desc = np.array([[10, 10, 10]], dtype=np.uint8)
    candidate_points = np.array([[52, 49], [90, 90]])  # second is far outside the window
    candidate_desc = np.array([[10, 10, 10], [10, 10, 10]], dtype=np.uint8)

    idx, sad = match_within_window(query_points, query_desc, candidate_points, candidate_desc, window_radius=5)

    assert idx[0] == 0
    assert sad[0] == 0


def test_match_within_window_no_candidate_returns_minus_one():
    query_points = np.array([[50, 50]])
    query_desc = np.array([[10, 10, 10]], dtype=np.uint8)
    candidate_points = np.array([[500, 500]])
    candidate_desc = np.array([[10, 10, 10]], dtype=np.uint8)

    idx, sad = match_within_window(query_points, query_desc, candidate_points, candidate_desc, window_radius=5)

    assert idx[0] == -1
    assert sad[0] == -1


def test_match_within_window_picks_lowest_sad_among_multiple_in_range():
    query_points = np.array([[50, 50]])
    query_desc = np.array([[10, 10, 10]], dtype=np.uint8)
    candidate_points = np.array([[51, 50], [52, 50]])  # both within radius
    candidate_desc = np.array([[50, 50, 50], [10, 10, 10]], dtype=np.uint8)  # index 1 is the true match

    idx, sad = match_within_window(query_points, query_desc, candidate_points, candidate_desc, window_radius=5)

    assert idx[0] == 1
    assert sad[0] == 0


def test_match_along_epipolar_respects_tolerance():
    query_points = np.array([[50, 50]])
    query_desc = np.array([[10, 10, 10]], dtype=np.uint8)
    candidate_points = np.array([[10, 51], [10, 52]])  # y=51 within tol=1, y=52 is not
    candidate_desc = np.array([[10, 10, 10], [10, 10, 10]], dtype=np.uint8)

    idx, sad = match_along_epipolar(
        query_points, query_desc, candidate_points, candidate_desc, y_tolerance=1
    )

    assert idx[0] == 0  # only the y=51 candidate qualifies


def test_match_along_epipolar_ignores_horizontal_distance():
    query_points = np.array([[500, 50]])
    query_desc = np.array([[10, 10, 10]], dtype=np.uint8)
    candidate_points = np.array([[5, 50]])  # far away in x, but same row
    candidate_desc = np.array([[10, 10, 10]], dtype=np.uint8)

    idx, sad = match_along_epipolar(query_points, query_desc, candidate_points, candidate_desc, y_tolerance=1)

    assert idx[0] == 0
    assert sad[0] == 0


def test_circular_match_accepts_consistent_circle():
    curr_left_points = np.array([[50, 50]])
    curr_left_desc = np.array([[10, 10, 10]], dtype=np.uint8)
    prev_left_points = np.array([[52, 49]])
    prev_left_desc = np.array([[10, 10, 10]], dtype=np.uint8)
    prev_right_points = np.array([[30, 49]])
    prev_right_desc = np.array([[10, 10, 10]], dtype=np.uint8)
    curr_right_points = np.array([[31, 50]])
    curr_right_desc = np.array([[10, 10, 10]], dtype=np.uint8)

    matches = circular_match(
        curr_left_points, curr_left_desc,
        prev_left_points, prev_left_desc,
        prev_right_points, prev_right_desc,
        curr_right_points, curr_right_desc,
        window_radius=5, epipolar_tolerance=1,
    )

    assert matches.shape == (1, 4)
    assert tuple(matches[0]) == (0, 0, 0, 0)


def test_circular_match_rejects_when_a_leg_has_no_candidate():
    curr_left_points = np.array([[50, 50]])
    curr_left_desc = np.array([[10, 10, 10]], dtype=np.uint8)
    prev_left_points = np.array([[500, 500]])  # outside the window: leg 1 fails
    prev_left_desc = np.array([[10, 10, 10]], dtype=np.uint8)
    prev_right_points = np.array([[30, 49]])
    prev_right_desc = np.array([[10, 10, 10]], dtype=np.uint8)
    curr_right_points = np.array([[31, 50]])
    curr_right_desc = np.array([[10, 10, 10]], dtype=np.uint8)

    matches = circular_match(
        curr_left_points, curr_left_desc,
        prev_left_points, prev_left_desc,
        prev_right_points, prev_right_desc,
        curr_right_points, curr_right_desc,
        window_radius=5, epipolar_tolerance=1,
    )

    assert matches.shape == (0, 4)


def test_circular_match_rejects_inconsistent_loop():
    # curr_left[0] completes all four legs, but a competing curr_left[1] on
    # the same epipolar row has a closer descriptor to curr_right's match,
    # so the loop closes on index 1 instead of 0 -> chain 0 must be rejected.
    curr_left_points = np.array([[50, 50], [10, 50]])
    curr_left_desc = np.array([[77, 77, 77], [76, 76, 76]], dtype=np.uint8)
    prev_left_points = np.array([[52, 49]])
    prev_left_desc = np.array([[77, 77, 77]], dtype=np.uint8)
    prev_right_points = np.array([[30, 49]])
    prev_right_desc = np.array([[77, 77, 77]], dtype=np.uint8)
    curr_right_points = np.array([[31, 50]])
    curr_right_desc = np.array([[76, 76, 76]], dtype=np.uint8)  # closer to curr_left[1]

    matches = circular_match(
        curr_left_points, curr_left_desc,
        prev_left_points, prev_left_desc,
        prev_right_points, prev_right_desc,
        curr_right_points, curr_right_desc,
        window_radius=5, epipolar_tolerance=1,
    )

    assert matches.shape == (0, 4)

"""Hardware-free checks of the checkerboard strip measurement.

Frames are synthesised with the sheet at a known place, then pushed through the
same measurement the checker uses.  The sign convention is the part most worth
guarding: if it flips, the tool tells you to turn the tripod the wrong way and
the error doubles instead of cancelling.

Run with:  python3 -m pytest sensor_utils/test/test_board_strip.py -v
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sensor_utils.board_strip import compare, measure  # noqa: E402

WIDTH, HEIGHT = 640, 360
FLOOR, SHEET, SQUARE = 160, 215, 60
SQUARE_PX = 16.0
SQUARES = 8


def synthetic_frame(centre_x=320.0, top_y=336.0, roll_deg=0.0,
                    square_px=SQUARE_PX, squares=SQUARES, seed=0, gain=1.0):
    """A speckled floor with a checkerboard sheet along the bottom edge.

    The sheet is placed so ``centre_x`` is the middle of its top edge and
    ``top_y`` is where that edge falls, matching what the measurement reports.
    """
    rng = np.random.default_rng(seed)
    frame = np.clip(
        rng.normal(FLOOR, 12.0, (HEIGHT, WIDTH)), 90, 200
    ).astype(np.uint8)

    margin = square_px * 0.3
    sheet_w = int(round(squares * square_px + 2 * margin))
    sheet_h = int(round(3 * square_px + margin))
    sheet = np.full((sheet_h, sheet_w), SHEET, np.uint8)
    for row in range(3):
        for column in range(squares):
            if (row + column) % 2:
                continue
            x = int(round(margin + column * square_px))
            y = int(round(margin + row * square_px))
            sheet[y:y + int(square_px), x:x + int(square_px)] = SQUARE

    angle = np.deg2rad(roll_deg)
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])
    anchor = np.array([sheet_w / 2.0, 0.0])
    offset = np.array([centre_x, top_y]) - rotation @ anchor
    transform = np.hstack([rotation, offset.reshape(2, 1)])

    cv2.warpAffine(sheet, transform, (WIDTH, HEIGHT), dst=frame,
                   flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
    return np.clip(frame.astype(float) * gain, 0, 255).astype(np.uint8)


# ------------------------------------------------------------------ recovery

def test_measurement_recovers_the_placement():
    result = measure(synthetic_frame(centre_x=320.0, top_y=336.0, roll_deg=0.0))
    assert result is not None
    assert abs(result['centre_offset_px']) < 2.0
    assert abs(result['top_edge_y'] - 336.0) < 2.0
    assert abs(result['roll_deg']) < 0.3
    assert abs(result['square_px'] - SQUARE_PX) < 1.5


def test_four_squares_fall_either_side_of_a_centred_sheet():
    """The rule the check was described in: centre splits the strip 4 and 4."""
    result = measure(synthetic_frame(centre_x=320.0))
    assert result['split']['left'] == 4
    assert result['split']['right'] == 4


def test_offset_sheet_no_longer_splits_evenly():
    result = measure(synthetic_frame(centre_x=320.0 + 2 * SQUARE_PX))
    assert result['split']['left'] == 2
    assert result['split']['right'] == 6


@pytest.mark.parametrize('centre_x,expected', [(300.0, -20.0), (340.0, 20.0)])
def test_centre_offset_tracks_the_sheet(centre_x, expected):
    result = measure(synthetic_frame(centre_x=centre_x))
    assert abs(result['centre_offset_px'] - expected) < 2.0


@pytest.mark.parametrize('roll_deg', [-2.0, -0.8, 0.8, 2.0])
def test_roll_tracks_the_sheet_tilt(roll_deg):
    result = measure(synthetic_frame(roll_deg=roll_deg))
    assert abs(result['roll_deg'] - roll_deg) < 0.3


# ------------------------------------------------------------ sign convention

@pytest.mark.parametrize('shift,expected_hint', [(6.0, 'RIGHT'), (-6.0, 'LEFT')])
def test_sideways_drift_points_the_camera_back(shift, expected_hint):
    """A strip that moved right means the camera moved left, so: turn right."""
    reference = measure(synthetic_frame())
    live = measure(synthetic_frame(centre_x=320.0 + shift))

    result = compare(reference, live)
    assert result['verdict'] == 'ADJUST'
    assert result['hints'] == [expected_hint]
    assert abs(result['axes']['yaw']['delta'] - shift) < 2.0


@pytest.mark.parametrize('shift,expected_hint', [(8.0, 'DOWN'), (-8.0, 'UP')])
def test_vertical_drift_points_the_camera_back(shift, expected_hint):
    """A strip that slid down means the camera tilted up, so: aim down."""
    reference = measure(synthetic_frame())
    live = measure(synthetic_frame(top_y=336.0 + shift))

    result = compare(reference, live)
    assert result['verdict'] == 'ADJUST'
    assert result['hints'] == [expected_hint]
    assert abs(result['axes']['pitch']['delta'] - shift) < 2.0
    assert abs(result['axes']['pitch']['rows'] - shift / SQUARE_PX) < 0.2


@pytest.mark.parametrize('tilt,expected_hint', [(1.5, 'CW'), (-1.5, 'CCW')])
def test_tilt_drift_points_the_camera_back(tilt, expected_hint):
    reference = measure(synthetic_frame())
    live = measure(synthetic_frame(roll_deg=tilt))

    result = compare(reference, live)
    assert result['verdict'] == 'ADJUST'
    assert result['hints'] == [expected_hint]
    assert abs(result['axes']['roll']['delta'] - tilt) < 0.3


def test_unmoved_camera_passes():
    reference = measure(synthetic_frame(seed=1))
    live = measure(synthetic_frame(seed=2))

    result = compare(reference, live)
    assert result['verdict'] == 'PASS'
    assert result['hints'] == []


def test_drift_within_tolerance_still_passes():
    reference = measure(synthetic_frame())
    live = measure(synthetic_frame(centre_x=322.0))
    assert compare(reference, live)['verdict'] == 'PASS'


def test_all_three_axes_are_reported_together():
    reference = measure(synthetic_frame())
    live = measure(synthetic_frame(centre_x=330.0, top_y=346.0, roll_deg=1.5))

    result = compare(reference, live)
    assert set(result['hints']) == {'RIGHT', 'DOWN', 'CW'}


# --------------------------------------------------------------- robustness

@pytest.mark.parametrize('gain', [0.55, 0.75, 1.0])
def test_measurement_is_independent_of_exposure(gain):
    """Thresholds come from Otsu, so a darker frame must read the same.

    Lighting on the track will not match whatever it was when the reference was
    taken, and a threshold that drifts with brightness would show up as fake
    camera drift.
    """
    bright = measure(synthetic_frame(gain=1.0))
    dim = measure(synthetic_frame(gain=gain))
    assert dim is not None
    assert abs(dim['centre_offset_px'] - bright['centre_offset_px']) < 1.0
    assert abs(dim['top_edge_y'] - bright['top_edge_y']) < 1.0
    assert abs(dim['roll_deg'] - bright['roll_deg']) < 0.2


def test_no_sheet_in_view_returns_nothing():
    rng = np.random.default_rng(0)
    floor = np.clip(rng.normal(FLOOR, 12.0, (HEIGHT, WIDTH)), 90, 200).astype(np.uint8)
    assert measure(floor) is None


def test_sheet_outside_the_search_band_is_not_found():
    """Only the bottom of the frame is searched, which is where the sheet is."""
    assert measure(synthetic_frame(top_y=60.0)) is None

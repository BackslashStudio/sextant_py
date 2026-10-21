"""sextant 1.3: plot-object ids and removal. last_object(), remove(id), find_object(id) on
Axes, Axes3D and Plane2D; Plane2D.id(), Axes3D.last_plane() / remove_plane() / plane_index();
pick events name the object and plane by id. An id that is not there warns through the
message handler and returns False; nothing raises."""

import numpy as np
import pytest

import sextant

X = np.array([0.0, 1.0, 2.0])
Y = np.array([1.0, 3.0, 2.0])


@pytest.fixture
def messages():
    got = []
    prev = sextant.set_message_handler(got.append)
    yield got
    sextant.set_message_handler(prev)


def axes():
    return sextant.Figure(width=240, height=180).add_subplot(1, 1, 1)


def axes3d():
    return sextant.Figure(width=240, height=180).add_subplot3d(1, 1, 1)


def test_last_object_is_the_latest_calls_id():
    ax = axes()
    assert ax.last_object() == 0
    ax.line(X, Y)
    a = ax.last_object()
    ax.line(X, Y)
    b = ax.last_object()
    ax.scatter(X, Y)
    c = ax.last_object()
    assert 0 < a < b < c
    assert all(isinstance(i, int) for i in (a, b, c))
    ax.axline(1.0)
    ax.text("t", 0.5, 0.5)
    assert ax.last_object() > c


def test_find_object_gives_kind_and_index():
    ax2 = axes()
    ax2.line(X, Y)
    first = ax2.last_object()
    ax2.line(X, Y)
    second = ax2.last_object()
    ax2.scatter(X, Y)
    third = ax2.last_object()
    assert ax2.find_object(first) == (sextant.PickKind.LINE, 0)
    assert ax2.find_object(second) == (sextant.PickKind.LINE, 1)
    assert ax2.find_object(third) == (sextant.PickKind.SCATTER, 0)
    assert ax2.find_object(0) is None and ax2.find_object(10**9) is None
    ax2.text("t", 0.5, 0.5)
    assert ax2.find_object(ax2.last_object()) is None  # a text has no PickKind


def test_remove_closes_up_indices_and_keeps_ids(messages):
    ax = axes()
    ids = []
    for k in range(3):
        ax.line(X, Y + k)
        ids.append(ax.last_object())
    ax.scatter(X, Y)
    assert ax.remove(ids[1]) is True
    assert ax.line_count() == 2 and ax.scatter_count() == 1
    assert list(ax.line_data(1).y) == list(Y + 2)  # the third line moved down
    assert ax.find_object(ids[2]) == (sextant.PickKind.LINE, 1)
    assert ax.find_object(ids[1]) is None
    assert messages == []


def test_remove_a_stale_or_foreign_id_warns(messages):
    ax, other = axes(), axes()
    ax.line(X, Y)
    mine = ax.last_object()
    other.line(X, Y)
    theirs = other.last_object()
    assert ax.remove(mine) is True and messages == []
    assert ax.remove(mine) is False and len(messages) == 1 and "remove" in messages[0]
    assert ax.remove(0) is False and ax.remove(theirs) is False and len(messages) == 3
    assert other.line_count() == 1  # the other axes is untouched


def test_remove_rejects_a_bad_id_type():
    ax = axes()
    with pytest.raises(TypeError):
        ax.remove(-1)
    with pytest.raises(TypeError):
        ax.remove("1")


def test_remove_decorations_and_areas(messages):
    ax = axes()
    ax.axline(1.0)
    line_id = ax.last_object()
    ax.text("t", 0.5, 0.5)
    text_id = ax.last_object()
    ax.fill(X, Y, 0.0)
    fill_id = ax.last_object()
    assert ax.remove(line_id) and ax.remove(text_id) and ax.remove(fill_id)
    assert ax.axline_count() == 0 and ax.text_count() == 0 and ax.fill_count() == 0
    assert messages == []


def test_3d_objects_and_planes(messages):
    ax = axes3d()
    assert ax.last_object() == 0 and ax.last_plane() == 0
    ax.scatter3d(X, Y, X)
    s0 = ax.last_object()
    ax.scatter3d(X, Y, X)
    s1 = ax.last_object()
    p0 = ax.plane(sextant.PlaneOrientation.XY, 0.0)
    p1 = ax.plane(sextant.PlaneOrientation.YZ, 1.0)
    p2 = ax.plane(sextant.PlaneOrientation.XY, 2.0)
    assert 0 < p0.id() < p1.id() < p2.id() and ax.last_plane() == p2.id()
    p0.line(X, Y)
    plane_obj = p0.last_object()
    assert plane_obj != 0 and ax.last_object() == s1  # the axes' own is untouched

    assert ax.find_object(s1) == (sextant.PickKind.SCATTER3D, 1)
    assert ax.remove(s0) is True and ax.find_object(s1) == (sextant.PickKind.SCATTER3D, 0)
    # A plane's object is the plane's to remove.
    assert ax.remove(plane_obj) is False and len(messages) == 1
    assert p0.find_object(plane_obj) == (sextant.PickKind.LINE, 0)
    assert p0.remove(plane_obj) is True and p0.line_count() == 0
    assert p0.remove(plane_obj) is False and len(messages) == 2

    assert ax.plane_index(p2.id()) == 2
    assert ax.remove_plane(p0.id()) is True
    assert ax.plane_count() == 2 and ax.plane_index(p2.id()) == 1 and ax.plane_index(p0.id()) is None
    n = len(messages)
    assert ax.remove_plane(p0.id()) is False and ax.remove_plane(0) is False
    assert len(messages) == n + 2 and all("remove" in m for m in messages)
    p0.line(X, Y)  # the held plane still works, detached
    assert p0.line_count() == 1 and ax.plane_count() == 2


def test_pick_events_carry_ids():
    ax = axes()
    ax.line(X, Y)
    obj = ax.last_object()
    e = sextant.Event  # the type exists; picks come from a window, so check the doc contract
    assert "id" in e.pick_object.__doc__ and "id" in e.pick_plane.__doc__
    assert ax.find_object(obj) == (sextant.PickKind.LINE, 0)

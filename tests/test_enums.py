"""The str-valued Enum classes in sextant/_enums.py against the C++ name tables."""

import enum

import pytest

import sextant
from sextant import _enums

TABLES = sextant._sextant._enum_tables()
CLASSES = {name: cls for name, cls in vars(_enums).items()
           if isinstance(cls, type) and issubclass(cls, _enums._Named) and cls is not _enums._Named}


def test_one_class_per_cpp_enum():
    assert set(CLASSES) == set(TABLES)
    for name in CLASSES:
        assert getattr(sextant, name) is CLASSES[name]


@pytest.mark.parametrize("name", sorted(TABLES))
def test_members_are_the_cpp_names_in_order(name):
    assert [m.value for m in CLASSES[name]] == TABLES[name]["names"]


@pytest.mark.parametrize("name", sorted(TABLES))
def test_aliases_resolve(name):
    cls = CLASSES[name]
    for alias, canonical in TABLES[name]["aliases"].items():
        assert cls(alias) is cls(canonical)


def test_members_are_strings():
    m = sextant.LineStyle.DASHED
    assert isinstance(m, str) and isinstance(m, enum.Enum)
    assert m == "dashed" and str(m) == "dashed" and f"{m}" == "dashed"
    assert sextant.LineStyle("Dash-Dot") is sextant.LineStyle.DASHDOT
    with pytest.raises(ValueError):
        sextant.LineStyle("wavy")


def test_members_are_accepted_wherever_strings_are():
    fig = sextant.Figure(width=64, height=48)
    fig.axes().line([0, 1], [0, 1], linestyle=sextant.LineStyle.DOTTED, color="red")
    fig2 = sextant.Figure(width=64, height=48, theme=sextant.PanelTheme.DARK)
    a3 = fig2.add_subplot3d(1, 1, 1)
    a3.set_projection(sextant.Projection.PERSPECTIVE)
    p = a3.plane(sextant.PlaneOrientation.YZ, 0)
    assert p.orientation() is sextant.PlaneOrientation.YZ
    assert a3.camera()["projection"] is sextant.Projection.PERSPECTIVE

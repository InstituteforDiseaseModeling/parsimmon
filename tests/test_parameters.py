import numpy as np
import pytest
import sciris as sc

from parsimmon.parameters import ParameterSet, ParameterSetManager, _ParamLink


@sc.timer()
def test_param_range_arange():
    r = ParameterSet.arange(0, 6, 2)
    vals = list(r)
    assert vals == [0, 2, 4], f"Expected [0, 2, 4], got {vals}"
    assert len(r) == 3, f"Expected len 3, got {len(r)}"

    r2 = ParameterSet.arange(3)
    assert list(r2) == [0, 1, 2], f"Expected [0, 1, 2], got {list(r2)}"


@sc.timer()
def test_param_range_iter():
    r = ParameterSet.iter([1, 4, 8])
    vals = list(r)
    assert vals == [1, 4, 8], f"Expected [1, 4, 8], got {vals}"
    assert len(r) == 3, f"Expected len 3, got {len(r)}"


@sc.timer()
def test_param_range_link():
    r = ParameterSet.arange(3)
    link = r.link(lambda x: x * 2)

    assert isinstance(link, _ParamLink), f"Expected _ParamLink, got {type(link)}"
    assert len(link.source) == 1, f"Expected 1 source, got {len(link.source)}"
    assert link.source[0] is r, "Link source should be the original range by identity"
    assert link.fn(5) == 10, "Link fn should double its input"


@sc.timer()
def test_pset_single_group():
    ps = ParameterSet({"a": {"x": 1}})
    ps.add("G", {"b": {"y": 2}})

    results = list(ps)
    assert len(results) == 1, f"Expected 1 dict, got {len(results)}"

    d = results[0].pars
    assert d["a"]["x"] == 1, f"Expected a.x=1, got {d['a']['x']}"
    assert d["b"]["y"] == 2, f"Expected b.y=2, got {d['b']['y']}"


@sc.timer()
def test_pset_range_expansion():
    ps = ParameterSet({"a": {"x": ParameterSet.arange(3)}})
    ps.add("G", {})

    results = list(ps)
    assert len(results) == 3, f"Expected 3 dicts, got {len(results)}"

    xs = [d.pars["a"]["x"] for d in results]
    assert list(xs) == [0, 1, 2], f"Expected [0, 1, 2], got {xs}"


@sc.timer()
def test_pset_multi_range_cartesian():
    ps = ParameterSet({
        "a": {"x": ParameterSet.arange(3)},
        "b": {"y": ParameterSet.iter([10, 20])},
    })
    ps.add("G", {})

    results = list(ps)
    assert len(results) == 6, f"Expected 3x2=6 dicts, got {len(results)}"

    pairs = [(d.pars["a"]["x"], d.pars["b"]["y"]) for d in results]
    for x in [0, 1, 2]:
        for y in [10, 20]:
            assert (x, y) in pairs, f"Missing combination x={x}, y={y}"


@sc.timer()
def test_pset_update_group():
    ps = ParameterSet()
    ps.add("G", {"a": 1, "b": 2})
    ps.add("G", {"b": 99, "c": 3})

    results = list(ps)
    assert len(results) == 1, f"Expected 1 dict, got {len(results)}"

    d = results[0].pars
    assert d["a"] == 1, f"a should be preserved, got {d.get('a')}"
    assert d["b"] == 99, f"b should be overwritten to 99, got {d.get('b')}"
    assert d["c"] == 3, f"c should be added, got {d.get('c')}"


@sc.timer()
def test_pset_update_defaults():
    ps = ParameterSet({"a": 1})
    ps.add("G1", {"b": 10})
    ps.add("G2", {"b": 20})
    ps.add({"a": 99})

    results = list(ps)
    assert len(results) == 2, f"Expected 2 dicts, got {len(results)}"

    for d in results:
        assert d.pars["a"] == 99, f"Base update should propagate, got a={d.pars.get('a')}"

    assert results[0].pars["b"] == 10, f"G1 b should be 10, got {results[0].pars.get('b')}"
    assert results[1].pars["b"] == 20, f"G2 b should be 20, got {results[1].pars.get('b')}"


@sc.timer()
def test_pset_clear():
    ps = ParameterSet()
    ps.add("G", {"a": {"x": 1, "y": 2}, "b": 3})
    ps.clear("G", ["a.x"])

    results = list(ps)
    d = results[0].pars
    assert "x" not in d.get("a", {}), f"a.x should be removed, got {d}"
    assert d["a"]["y"] == 2, f"a.y should remain, got {d}"
    assert d["b"] == 3, f"b should remain, got {d}"

    ps2 = ParameterSet({"a": {"x": 1, "y": 2}, "b": 3})
    ps2.add("G", {})
    ps2.clear(["a.x", "b"])

    results2 = list(ps2)
    d2 = results2[0].pars
    assert "x" not in d2.get("a", {}), "a.x should be removed from base"
    assert "b" not in d2, "b should be removed from base"
    assert d2["a"]["y"] == 2, "a.y should remain in base"


@sc.timer()
def test_pset_len():
    ps = ParameterSet({
        "a": ParameterSet.arange(3),
        "b": ParameterSet.iter([10, 20]),
    })
    ps.add("G1", {})
    ps.add("G2", {"c": 99})

    expected = 6 + 6  # 3x2 per group, 2 groups
    assert len(ps) == expected, f"Expected {expected}, got {len(ps)}"
    assert len(list(ps)) == expected, "Actual iteration count mismatch"


@sc.timer()
def test_pset_attr_access():
    ps = ParameterSet({"a": {"x": 1, "y": 2}})
    ps.add("G", {"a": {"x": 10}, "b": {"z": 3}})

    assert ps.a.x == 1, f"Expected ps.a.x=1, got {ps.a.x}"
    assert ps.a.y == 2, f"Expected ps.a.y=2, got {ps.a.y}"

    assert ps.G.a.x == 10, f"Expected ps.G.a.x=10, got {ps.G.a.x}"
    assert ps.G.b.z == 3, f"Expected ps.G.b.z=3, got {ps.G.b.z}"

    r = ParameterSet.arange(3)
    ps2 = ParameterSet()
    ps2.add("G", {"a": {"b": r}})
    assert list(ps2.G.a.b) == [0, 1, 2], "Range values should match after copy"
    assert hasattr(ps2.G.a.b, "link"), "Copied range should still have .link() method"


@sc.timer()
def test_pset_attr_setattr():
    ps = ParameterSet()
    ps.add("G", {"a": {"x": 1}})

    ps.G.a.y = 99
    assert ps.G.a.y == 99, f"Expected ps.G.a.y=99 after assignment, got {ps.G.a.y}"
    assert ps._groups["G"]["a"]["y"] == 99, "Should write through to underlying dict"

    results = list(ps)
    assert results[0].pars["a"]["y"] == 99, "Assigned value should appear in iterated pars"


@sc.timer()
def test_pset_group_copy():
    ps = ParameterSet()
    r = ParameterSet.arange(3)
    ps.add("G1", {"a": {"b": r}})
    ps.add("G2", ps.G1)

    assert ps.G2.a.b is not r, "Copied group should have a new _ParamRange"
    assert list(ps.G2.a.b) == [0, 1, 2], "Copied range should have same values"

    ps.G2.a.c = 99
    assert "c" not in ps._groups["G1"].get("a", {}), "G1 should not be affected by G2 mutation"


@sc.timer()
def test_pset_collision_detection():
    ps = ParameterSet()
    with pytest.raises(ValueError, match="collides with a ParameterSet attribute"):
        ps.add("add", {"x": 1})
    with pytest.raises(ValueError, match="collides with a ParameterSet attribute"):
        ps.add("groups", {"x": 1})
    with pytest.raises(ValueError, match="collides with a ParameterSet attribute"):
        ps.add("_private", {"x": 1})

    ps2 = ParameterSet({"a": {"x": 1}})
    with pytest.raises(ValueError, match="collides with default key"):
        ps2.add("a", {"y": 2})

    ps3 = ParameterSet()
    ps3.add("G", {"x": 1})
    with pytest.raises(ValueError, match="collides with group name"):
        ps3.add({"G": {"y": 2}})

    # bypass validation to test ambiguous getattr
    ps4 = ParameterSet()
    ps4._base["conflict"] = sc.objdict(x=1)
    ps4._groups["conflict"] = sc.objdict(x=2)
    with pytest.raises(AttributeError, match="exists as both"):
        _ = ps4.conflict


@sc.timer()
def test_short_link():
    ps = ParameterSet()
    r = ParameterSet.arange(3)
    ps.add("G", {"a": {"b": r, "c": ps.link("b", lambda x: x * 10)}})

    results = list(ps)
    assert len(results) == 3, f"Expected 3 expansions, got {len(results)}"

    for d in results:
        b = d.pars["a"]["b"]
        c = d.pars["a"]["c"]
        assert c == b * 10, f"Short link should resolve c = b*10; got b={b}, c={c}"


@sc.timer()
def test_range_identity_link():
    ps = ParameterSet()
    r = ParameterSet.arange(3)
    ps.add("G", {"a": {"b": r, "c": r.link(lambda x: x + 100)}})

    results = list(ps)
    assert len(results) == 3, f"Expected 3 expansions, got {len(results)}"

    for d in results:
        b = d.pars["a"]["b"]
        c = d.pars["a"]["c"]
        assert c == b + 100, f"Identity link should resolve c = b+100; got b={b}, c={c}"


@sc.timer()
def test_link_chain():
    ps = ParameterSet()
    r = ParameterSet.arange(3)
    ps.add(
        "G",
        {
            "a": r,
            "b": ps.link("a", lambda x: x * 2),
            "c": ps.link("b", lambda x: x + 100),
        },
    )

    results = list(ps)
    assert len(results) == 3, f"Expected 3 expansions, got {len(results)}"

    for d in results:
        a = d.pars["a"]
        b = d.pars["b"]
        c = d.pars["c"]
        assert b == a * 2, f"b should be a*2; got a={a}, b={b}"
        assert c == b + 100, f"c should be b+100; got b={b}, c={c}"


@sc.timer()
def test_full_path_link():
    ps = ParameterSet()
    r = ParameterSet.iter([0.1, 0.5, 0.9])
    ps.add(
        "G",
        {
            "strains": {"rd": {"init": r}, "ss": {"init": ps.link("strains.rd.init", lambda x: round(1 - x, 2))}},
        },
    )

    results = list(ps)
    assert len(results) == 3, f"Expected 3 expansions, got {len(results)}"

    for d in results:
        rd = d.pars["strains"]["rd"]["init"]
        ss = d.pars["strains"]["ss"]["init"]
        assert np.isclose(rd + ss, 1.0, rtol=0.01), f"rd + ss should be ~1.0; got rd={rd}, ss={ss}"


@sc.timer()
def test_manager_add_bare():
    pm = ParameterSetManager()

    @pm.add
    def simple(ps):
        ps.add("simple", {"a": 1})
        return ps

    ps = pm._build("simple")
    results = list(ps)
    assert len(results) == 1, f"Expected 1 dict, got {len(results)}"
    assert results[0].pars["a"] == 1, f"Expected a=1, got {results[0].pars}"
    assert ps.groups == ["simple"], f"Expected group 'simple', got {ps.groups}"


@sc.timer()
def test_manager_add_named():
    pm = ParameterSetManager()

    @pm.add("Custom")
    def my_fn(ps):
        ps.add("Custom", {"a": 1})
        return ps

    ps = pm._build("Custom")
    results = list(ps)
    assert len(results) == 1, f"Expected 1 dict, got {len(results)}"
    assert ps.groups == ["Custom"], f"Expected group 'Custom', got {ps.groups}"


@sc.timer()
def test_manager_extend():
    pm = ParameterSetManager()

    @pm.add
    def parent(ps):
        ps.add("P", {"a": 1, "b": 2})
        return ps

    @pm.add(extends=parent)
    def child(ps):
        ps.add("C", {"b": 99, "c": 3})
        return ps

    ps = pm._build("child")
    assert "P" in ps.groups, f"Parent group should be inherited, got {ps.groups}"
    assert "C" in ps.groups, f"Child group should be present, got {ps.groups}"

    results = list(ps)
    assert len(results) == 2, f"Expected 2 dicts, got {len(results)}"

    p_result = next(r for r in results if r.group == "P").pars
    c_result = next(r for r in results if r.group == "C").pars
    assert p_result["a"] == 1 and p_result["b"] == 2, f"P should be unchanged: {p_result}"
    assert c_result["b"] == 99 and c_result["c"] == 3, f"C should have overrides: {c_result}"


@sc.timer()
def test_manager_extend_defaults():
    pm = ParameterSetManager()

    @pm.add
    def base():
        return {"a": {"x": 1, "y": 2}}

    @pm.add(extends=base)
    def child(ps):  # ruff: ignore[reportUnusedFunction]
        ps.add({"a": {"x": 99}})
        ps.add("G", {})
        return ps

    ps = pm._build("child")
    results = list(ps)
    assert len(results) == 1, f"Expected 1 dict, got {len(results)}"

    d = results[0].pars
    assert d["a"]["x"] == 99, f"Default x should be updated to 99, got {d['a']['x']}"
    assert d["a"]["y"] == 2, f"Default y should be inherited as 2, got {d['a']['y']}"


@sc.timer()
def test_manager_cli_overrides():
    pm = ParameterSetManager()

    @pm.add
    def grp(ps):
        ps.add({"sim": {"dur": 40, "n_agents": 1000}})
        ps.add("grp", {})
        return ps

    ps = pm._build("grp", cli_overrides=["sim.dur=20"])
    results = list(ps)

    d = results[0].pars
    assert d["sim"]["dur"] == 20, f"Expected dur=20, got {d['sim']['dur']}"
    assert isinstance(d["sim"]["dur"], (int, np.integer)), f"dur should be int, got {type(d['sim']['dur'])}"
    assert d["sim"]["n_agents"] == 1000, "n_agents should be unchanged"


@sc.timer()
def test_manager_zero_arg_builder():
    pm = ParameterSetManager()

    @pm.add
    def base():
        return {"a": {"b": 0}}

    @pm.add(extends=base)
    def child(ps):
        ps.add("G", {"a": {"c": 1}})
        return ps

    ps = pm._build("child")
    results = list(ps)
    assert len(results) == 1, f"Expected 1 dict, got {len(results)}"

    d = results[0].pars
    assert d["a"]["b"] == 0, f"Inherited default a.b should be 0, got {d['a']['b']}"
    assert d["a"]["c"] == 1, f"Child override a.c should be 1, got {d['a']['c']}"


@sc.timer()
def test_manager_extend_chain():
    pm = ParameterSetManager()

    @pm.add
    def base():
        return {"a": {"b": 0}}

    @pm.add(extends=base)
    def mid(ps):
        return {"a": {"b": 1}}

    @pm.add(extends=mid)
    def leaf(ps):
        ps.add("G1", {"a": {"b": 2}})
        ps.add("G2", {"a": {"b": 3}})
        return ps

    ps = pm._build("leaf")
    assert ps.groups == ["G1", "G2"], f"Expected G1, G2; got {ps.groups}"

    results = list(ps)
    assert len(results) == 2, f"Expected 2 dicts, got {len(results)}"

    g1 = next(r for r in results if r.group == "G1").pars
    g2 = next(r for r in results if r.group == "G2").pars
    assert g1["a"]["b"] == 2, f"G1 override should be 2, got {g1['a']['b']}"
    assert g2["a"]["b"] == 3, f"G2 override should be 3, got {g2['a']['b']}"


@sc.timer()
def test_manager_return_none_raises():
    pm = ParameterSetManager()

    @pm.add
    def bad(ps):
        ps.add("G", {"a": 1})

    with pytest.raises(TypeError, match="Mapping or ParameterSet"):
        pm._build("bad")


if __name__ == "__main__":
    T = sc.timer()

    test_param_range_arange()
    test_param_range_iter()
    test_param_range_link()
    test_pset_single_group()
    test_pset_range_expansion()
    test_pset_multi_range_cartesian()
    test_pset_update_group()
    test_pset_update_defaults()
    test_pset_clear()

    test_pset_len()
    test_pset_attr_access()
    test_pset_attr_setattr()
    test_pset_group_copy()
    test_pset_collision_detection()
    test_short_link()
    test_range_identity_link()
    test_link_chain()
    test_full_path_link()
    test_manager_add_bare()
    test_manager_add_named()
    test_manager_extend()
    test_manager_extend_defaults()
    test_manager_cli_overrides()
    test_manager_zero_arg_builder()
    test_manager_extend_chain()
    test_manager_return_none_raises()

    T.toc()

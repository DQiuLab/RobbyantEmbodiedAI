import pytest

from src import stream_merge as sm


def test_unit_of_groups_lora_parts_with_their_base():
    p = "state.model.m.layers.0.self_attn.q_proj"
    for s in (".base_layer.weight", ".base_layer.bias", ".lora_A.default.weight", ".lora_B.default.weight"):
        assert sm.unit_of(p + s) == p
    assert sm.unit_of("state.model.m.norm.weight") == "state.model.m.norm.weight"


def test_plan_groups_never_splits_a_lora_unit_and_keeps_order():
    p = "x.q_proj"
    sizes = {"a": 6, p + ".base_layer.weight": 5, p + ".lora_A.default.weight": 1, p + ".lora_B.default.weight": 1, "b": 3}
    groups = sm.plan_groups(sizes, group_bytes=8)
    assert groups == [["a"], [p + ".base_layer.weight", p + ".lora_A.default.weight", p + ".lora_B.default.weight"], ["b"]]
    assert [k for g in groups for k in g] == list(sizes)


def test_merge_group_matches_amd_formula_and_renames_bias():
    torch = pytest.importorskip("torch")
    g = torch.Generator().manual_seed(0)
    base, a, b = torch.randn(4, 3, generator=g), torch.randn(2, 3, generator=g), torch.randn(4, 2, generator=g)
    bias = torch.randn(4, generator=g)
    p = "m.q_proj"
    out = sm.merge_group({p + ".base_layer.weight": base, p + ".base_layer.bias": bias,
                          p + ".lora_A.default.weight": a, p + ".lora_B.default.weight": b, "m.norm.weight": bias},
                         alpha=64, rank=32)
    assert set(out) == {p + ".weight", p + ".bias", "m.norm.weight"}
    assert torch.allclose(out[p + ".weight"], base + 2.0 * (b @ a))
    assert torch.equal(out[p + ".bias"], bias)


def test_merge_group_rejects_half_lora_unit():
    torch = pytest.importorskip("torch")
    with pytest.raises(KeyError):
        sm.merge_group({"m.q.lora_A.default.weight": torch.zeros(1, 1)}, 64, 32)

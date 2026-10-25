import json

from tools import analyze_trace


def test_summarise_busy_fraction_and_tops(tmp_path):
    ev = [
        {"ph": "X", "cat": "cpu_op", "name": "aten::mm", "ts": 0, "dur": 100},
        {"ph": "X", "cat": "kernel", "name": "gemm", "ts": 10, "dur": 30},
        {"ph": "X", "cat": "kernel", "name": "gemm", "ts": 30, "dur": 20},   # overlaps → merged
        {"ph": "X", "cat": "kernel", "name": "tiny", "ts": 80, "dur": 5},
        {"ph": "X", "cat": "cuda_runtime", "name": "cudaLaunchKernel", "ts": 5, "dur": 3},
    ]
    s = analyze_trace.summarise(ev)
    assert s["wall_s"] == 100 / 1e6
    assert abs(s["gpu_busy_frac"] - 0.45) < 1e-9          # (10..50)=40 + 5 = 45 of 100
    assert s["top_kernels"][0][0] == "gemm" and s["top_kernels"][0][2] == 2
    assert s["kernels_under_10us_frac"] == 1 / 3
    p = tmp_path / "t.json"
    p.write_text(json.dumps({"traceEvents": ev}))
    assert analyze_trace.main([str(p)]) == 0


def test_gemm_precision_classification():
    c = analyze_trace.gemm_precision
    assert c("void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x64_16x4_tn_align4>") == "tf32/fp32"
    assert c("void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_tn_align8>") == "bf16"
    assert c("sm80_xmma_gemm_bf16bf16_bf16f32_f32_tn_n_tilesize128x128x32") == "bf16"
    assert c("ampere_sgemm_128x64_tn") == "tf32/fp32"
    assert c("sm86_xmma_fprop_implicit_gemm_indexed_tf32f32_tf32f32_f32_nhwckrsc") == "tf32/fp32"
    assert c("group_gemm_same_nk_kernel") == "other-gemm"
    assert c("triton_poi_fused_add_fill_mul_silu_sub_2") is None
    ev = [
        {"ph": "X", "cat": "kernel", "name": "cutlass_80_tensorop_s1688gemm_x", "ts": 0, "dur": 30},
        {"ph": "X", "cat": "kernel", "name": "cutlass_80_tensorop_bf16_s16816gemm_x", "ts": 40, "dur": 10},
        {"ph": "X", "cat": "kernel", "name": "elementwise", "ts": 60, "dur": 40},
    ]
    s = analyze_trace.summarise(ev)
    assert s["gemm_precision_s"] == {"tf32/fp32": 30 / 1e6, "bf16": 10 / 1e6}

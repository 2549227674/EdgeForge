"""Regenerate the README figures (light + dark).

Usage: python docs/figures/make_figures.py [output_dir]
Numbers come from files under results/ and docs/ (paths noted beside each figure).
Needs matplotlib and a CJK font (Microsoft YaHei, Noto Sans CJK SC or similar).
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)

for f in ("msyh.ttc", "msyhbd.ttc"):
    for base in ("/mnt/c/Windows/Fonts", "C:/Windows/Fonts"):
        if Path(base, f).exists():
            font_manager.fontManager.addfont(str(Path(base, f)))
plt.rcParams["font.family"] = ["Microsoft YaHei", "Noto Sans CJK SC", "PingFang SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

THEMES = {
    "light": dict(surface="#ffffff", panel="#f6f8fa", ink="#1f2328", ink2="#59636e", muted="#818b98",
                  grid="#e6e9ed", axis="#d1d9e0", blue="#2a78d6", orange="#eb6834", aqua="#1baf7a",
                  gray1="#c9d1d9", gray2="#8c959f", good="#0ca30c", bad="#d03b3b", badfill="#fbe3e3"),
    "dark": dict(surface="#0d1117", panel="#161b22", ink="#f0f6fc", ink2="#9198a1", muted="#6e7681",
                 grid="#21262d", axis="#30363d", blue="#3987e5", orange="#d95926", aqua="#199e70",
                 gray1="#3d444d", gray2="#6e7681", good="#0ca30c", bad="#e66767", badfill="#3b1a1c"),
}


def style_axes(ax, t, grid_axis="y"):
    ax.set_facecolor(t["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["axis"])
        ax.spines[side].set_linewidth(1)
    ax.tick_params(colors=t["ink2"], labelsize=9, length=0, pad=6)
    ax.grid(axis=grid_axis, color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def titles(fig, t, title, subtitle):
    fig.text(0.012, 0.975, title, fontsize=13, fontweight="bold", color=t["ink"], va="top")
    fig.text(0.012, 0.905, subtitle, fontsize=9.5, color=t["ink2"], va="top")


def save(fig, name, t_name, t):
    fig.patch.set_facecolor(t["surface"])
    fig.savefig(OUT / f"{name}_{t_name}.png", dpi=180, facecolor=t["surface"])
    plt.close(fig)


# ---------------------------------------------------------------- 1. architecture
def fig_arch(t_name, t):
    fig = plt.figure(figsize=(13, 7.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 26)
    ax.set_ylim(0, 14.8)
    ax.axis("off")
    fig.patch.set_facecolor(t["surface"])

    def box(x, y, w, h, title, body, accent, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.22",
                                    facecolor=t["panel"], edgecolor=accent, linewidth=1.6 if bold else 1.1))
        ax.text(x + w / 2, y + h - 0.32, title, ha="center", va="top", fontsize=10.2,
                fontweight="bold", color=t["ink"])
        if body:
            ax.text(x + w / 2, y + h - 0.86, body, ha="center", va="top", fontsize=8.6,
                    color=t["ink2"], linespacing=1.3)

    def arrow(p0, p1, rad=0.0):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=11, linewidth=1.2,
                                     color=t["muted"], connectionstyle=f"arc3,rad={rad}",
                                     shrinkA=2, shrinkB=2))

    def lane(x, y, w, h, label, accent):
        ax.add_patch(Rectangle((x, y + h - 0.08), 0.5, 0.08, color=accent, lw=0))
        ax.text(x, y + h + 0.12, label, fontsize=10.5, fontweight="bold", color=t["ink"], va="bottom")

    ax.text(0.5, 14.35, "EdgeForge 全链路：训练 → 部署 → 算子优化", fontsize=15, fontweight="bold",
            color=t["ink"], va="top")
    ax.text(0.5, 13.55, "同一个模型（Gemma4 E4B）贯穿训练、PC 推理与 RK3588 板端；每个优化都与强基线对照并做正确性检查",
            fontsize=9.8, color=t["ink2"], va="top")

    # ① training: vertical stack on the left
    lane(0.5, 1.9, 6.2, 10.2, "① 训练与蒸馏 · 云端 2×96 GB GPU", t["aqua"])
    bw, bh = 6.2, 1.86
    ys = [10.12, 8.1, 6.08, 4.06, 2.04]
    items = [("多源 agent 轨迹", "9 个来源，统一为 messages / tools"),
             ("数据治理", "近重去重 · 测试集去污染 · 脱敏\n15.4 万条规范记录"),
             ("4K 轮级切窗", "历史只作上下文，监督互斥\n监督信号保留 99.99%"),
             ("SFT", "QLoRA · 双卡 DDP · token 加权损失\n4,096 步，验证 loss 1.63 → 0.84"),
             ("在线蒸馏（OPD）", "31B teacher（vLLM 部署）\nrsLoRA · 960 步")]
    for (ttl, body), y in zip(items, ys):
        box(0.5, y, bw, bh, ttl, body, t["aqua"])
    for y0, y1 in zip(ys[:-1], ys[1:]):
        arrow((0.5 + bw / 2, y0), (0.5 + bw / 2, y1 + bh))

    # hub
    box(7.6, 5.55, 3.4, 3.1, "端侧模型", "SFT + OPD 合并\nBF16 · 张量审计\n独立进程重载", t["ink2"], bold=True)
    arrow((0.5 + bw, ys[-1] + bh / 2), (7.6, 6.4), rad=0.18)

    # ② PC GPU
    lane(12.0, 9.05, 13.5, 3.05, "② PC 推理 · RTX 4060 Laptop 8 GB", t["blue"])
    pw, ph, py = 3.12, 2.3, 9.5
    pc = [("NF4 加载", "调用统计（shape census）\n确定优化对象"),
          ("Triton 算子", "RMSNorm · NF4 GEMV\nGQA decode attention"),
          ("接入与校验", "PyTorch 自定义算子\n逐 token 对照"),
          ("lm_head GPU 镜像", "输出一致\n每 token 延迟 1.2×")]
    xs = [12.0 + i * (pw + 0.34) for i in range(4)]
    for (ttl, body), x in zip(pc, xs):
        box(x, py, pw, ph, ttl, body, t["blue"])
    for x0, x1 in zip(xs[:-1], xs[1:]):
        arrow((x0 + pw, py + ph / 2), (x1, py + ph / 2))
    arrow((11.0, 7.9), (12.0, py + 0.9), rad=-0.15)

    # ③ RK3588
    lane(12.0, 1.9, 13.5, 6.05, "③ RK3588 板端 · 4×A76 + 4×A55 + NPU", t["orange"])
    ry1, rw, rh = 5.6, 4.3, 2.3
    rk = [("补齐共享 KV 权重", "转换器缺 54 个键\n从官方权重补全、逐位校验"),
          ("领域校准 + W8A8", "128 窗口分层校准集\nRKLLM 量化转换"),
          ("NPU 单核部署", "16K 上下文生成验证\nNPU 负载约 94%")]
    rxs = [12.0 + i * (rw + 0.3) for i in range(3)]
    for (ttl, body), x in zip(rk, rxs):
        box(x, ry1, rw, rh, ttl, body, t["orange"])
    for x0, x1 in zip(rxs[:-1], rxs[1:]):
        arrow((x0 + rw, ry1 + rh / 2), (x1, ry1 + rh / 2))
    arrow((11.0, 6.2), (12.0, ry1 + 1.0), rad=0.12)
    ry2 = 2.04
    an = [("IOVA 排障", "ioctl 拦截 + ftrace\n定位 4 GiB 窗口耗尽"),
          ("ARM NEON 算子", "FP16 GEMV 较 -O3 2.0×\nRMSNorm 2.35×"),
          ("NPU 原语评测", "RKNN 矩阵乘按形状扫描\nCPU/NPU attention 对照")]
    for (ttl, body), x in zip(an, rxs):
        box(x, ry2, rw, rh, ttl, body, t["orange"])
    ax.text(12.0, ry2 + rh + 0.33, "板端分析（与部署并行）", fontsize=8.8, color=t["muted"], va="bottom")

    # method strip
    ax.add_patch(FancyBboxPatch((0.5, 0.35), 25.0, 1.05, boxstyle="round,pad=0.02,rounding_size=0.2",
                                facecolor=t["panel"], edgecolor=t["axis"], linewidth=1))
    ax.text(0.95, 0.875, "测量方法", fontsize=10, fontweight="bold", color=t["ink"], va="center")
    ax.text(3.1, 0.875, "测前冻结比较对象与通过条件  →  强基线（PyTorch eager / torch.compile / GCC -O3）  →  "
            "正确性检查（数值容差 · 逐 token）  →  机器可读记录  →  结果裁决（失败结果同样保留）",
            fontsize=9.2, color=t["ink2"], va="center")
    save(fig, "architecture", t_name, t)


# ---------------------------------------------------------------- 2. IOVA window
def fig_iova(t_name, t):
    # bytes: docs/m0/06_board_smoke.md §4.1 (3-core, domain 0, page-aligned map sizes)
    maps = [12_709_888, 88_829_952, 3_972_792_320]
    window, req, MB = 4 * 1024 ** 3, 265_568_256, 1e6
    W, U, R = window / MB, sum(maps) / MB, req / MB
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 4.4), gridspec_kw=dict(height_ratios=[1, 1.35]))
    fig.subplots_adjust(left=0.03, right=0.97, top=0.76, bottom=0.1, hspace=0.75)
    titles(fig, t, "多核 NPU 初始化失败：设备地址窗口（IOVA）装不下，而不是内存不够",
           "RK3588 · 3 核配置 · IOMMU domain 0 的 4 GiB 窗口；数据来自 ioctl 拦截与 ftrace 映射记录（页对齐，单位 MB）")
    for a in (a1, a2):
        a.set_facecolor(t["surface"])
        for sp in a.spines.values():
            sp.set_visible(False)
        a.set_yticks([])
        a.tick_params(colors=t["ink2"], labelsize=8.5, length=0)
    a1.barh(0, W, height=0.42, color=t["gray1"], lw=0)
    a1.barh(0, U, height=0.42, color=t["blue"], lw=0)
    a1.text(40, 0, "已映射 4,074.3 MB（3 块缓冲，其中模型权重 3,972.8 MB）", fontsize=9, color="#ffffff", va="center")
    a1.text(W, -0.42, "剩余 220.6 MB", fontsize=9, color=t["ink2"], va="top", ha="right")
    a1.set_ylim(-0.95, 0.5)
    a1.set_xlim(0, 4450)
    a1.set_xticks([0, 1000, 2000, 3000, 4000])
    a1.set_xticklabels(["0", "1,000", "2,000", "3,000", "4,000"])
    a1.set_title("完整窗口", loc="left", fontsize=9.5, color=t["ink2"], pad=6)
    lo, hi = 3950, 4420
    a2.barh(0.55, W - U, left=U, height=0.4, color=t["gray1"], lw=0)
    a2.barh(0.55, U - lo, left=lo, height=0.4, color=t["blue"], lw=0)
    a2.text(U - 6, 0.55, "已映射", fontsize=9, color="#ffffff", va="center", ha="right")
    a2.text(U + 6, 0.55, "剩余 220.6 MB", fontsize=9, color=t["ink"], va="center")
    a2.barh(-0.1, R, left=U, height=0.4, color=t["badfill"], edgecolor=t["bad"], lw=1.4)
    a2.text(U + 6, -0.1, "第 4 笔申请 265.6 MB", fontsize=9, color=t["ink"], va="center")
    a2.axvline(W, color=t["ink"], lw=1.2, ymin=0.08, ymax=0.98)
    a2.text(W - 4, 0.98, "4 GiB 上限", fontsize=9, color=t["ink"], va="bottom", ha="right")
    a2.annotate("", xy=(U + R, -0.62), xytext=(W, -0.62), arrowprops=dict(arrowstyle="<->", color=t["bad"], lw=1.2))
    a2.text(W + 4, -0.8, "超出 44.9 MB → 分配失败（-12）", fontsize=9, color=t["ink"], va="top", ha="left")
    a2.set_xlim(lo, hi)
    a2.set_ylim(-1.25, 1.05)
    a2.set_xticks([4000, 4100, 4200, 4300, 4400])
    a2.set_xticklabels(["4,000", "4,100", "4,200", "4,300", "4,400"])
    a2.set_title("窗口末端放大", loc="left", fontsize=9.5, color=t["ink2"], pad=6)
    save(fig, "iova_window", t_name, t)


# ---------------------------------------------------------------- 3. NEON baselines
def fig_neon(t_name, t):
    # ms, p50. INT8: results/operator/board/ope3/ope3_receipt_20260919.json (two-segment aggregate)
    # FP16 A76: docs/operator/..._OPE5_FP16_GEMV正式段报告; FP16 A55: ..._OPE5_A55_FP16_GEMV (7.254041 ms × 4.65)
    groups = ["INT8 GEMV · 4×A76\n编译器已生成 SDOT，手写仅快 1.04×",
              "FP16 GEMV · 4×A76\n编译器未生成向量 FMA，手写快 2.00×",
              "FP16 GEMV · 4×A55\n同源代码在小核上，手写快 4.65×"]
    vals = {"标量（禁用向量化）": [5.806, 5.836, 33.732],
            "GCC -O3 自动向量化": [0.978, 5.837, 33.731],
            "手写 NEON": [0.945, 2.923, 7.254]}
    colors = [t["gray1"], t["gray2"], t["blue"]]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.74, bottom=0.2)
    titles(fig, t, "ARM NEON 手写 vs 编译器：收益取决于编译器是否已经向量化",
           "M=1, K=10240, N=2560 矩阵向量乘；同一线程拓扑与数据，p50 延迟，以标量版本为 1.0（越低越好）")
    style_axes(ax, t)
    w = 0.17
    for i, (name, v) in enumerate(vals.items()):
        xs = [g + (i - 1) * (w + 0.04) for g in range(3)]
        rel = [v[g] / vals["标量（禁用向量化）"][g] for g in range(3)]
        ax.bar(xs, rel, width=w, color=colors[i], label=name, lw=0)
        for x, r, ms in zip(xs, rel, v):
            ax.text(x, r + 0.025, f"{ms:.2f} ms" if ms < 10 else f"{ms:.1f} ms", ha="center", va="bottom",
                    fontsize=8, color=t["ink2"])
    ax.set_xticks(range(3))
    ax.set_xticklabels(groups, color=t["ink"], fontsize=9.2, linespacing=1.6)
    ax.set_ylim(0, 1.2)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("相对延迟", color=t["ink2"], fontsize=9)
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=3, frameon=False, fontsize=9)
    for txt in leg.get_texts():
        txt.set_color(t["ink2"])
    save(fig, "neon_baselines", t_name, t)


# ---------------------------------------------------------------- 4. NPU shapes
def npu_series():
    rows = list(csv.DictReader(open(ROOT / "results/operator/board/ope2/ope2_cells_20260918.csv")))
    series = {}
    for r in rows:
        if r["terminal_status"] != "PASS" or not r["type"].startswith("int8"):
            continue
        if r["scope"] not in ("formal", "synthetic_m_sweep"):
            continue
        key, m, g = (int(r["K"]), int(r["N"])), int(r["M"]), float(r["measured_gops"])
        series.setdefault(key, {})
        series[key][m] = max(g, series[key].get(m, 0.0))
    return series


def fig_npu(t_name, t):
    s = npu_series()
    lines = [((10240, 2560), "FFN down_proj（K=10240, N=2560）", t["orange"], 2.2),
             ((2560, 10240), "FFN gate/up_proj（K=2560, N=10240）", t["blue"], 2.0),
             ((2048, 2560), "attention o_proj（K=2048, N=2560）", t["gray2"], 1.5)]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.7, bottom=0.14)
    titles(fig, t, "RKNN 矩阵乘按真实形状测试：同样的权重量，K 维不同，吞吐相差约 50 倍",
           "INT8 · NPU core0 · librknnrt 2.3.2 matmul API；两个 FFN 投影权重量相同（2560×10240）；峰值为单核标称的 87%")
    style_axes(ax, t, grid_axis="both")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axhline(2000, color=t["axis"], lw=1.2)
    ax.text(1.05, 2150, "单核标称 2 TOPS", fontsize=8.5, color=t["ink2"], va="bottom")
    for key, label, color, lw in lines:
        pts = sorted(s[key].items())
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        ax.plot(xs, ys, color=color, lw=lw, solid_capstyle="round", label=label, zorder=3)
        ax.scatter(xs, ys, s=22, color=color, edgecolor=t["surface"], linewidth=1.2, zorder=4)
        extra = " · 峰值" if key == (2048, 2560) else ""
        ax.text(xs[-1] * 1.15, ys[-1], f"{ys[-1]:,.0f} GOPS{extra}", fontsize=9, color=t["ink"], va="center")
    ax.set_xlim(0.8, 7082 * 9)
    ax.set_ylim(10, 5000)
    ax.set_xticks([1, 10, 100, 1000, 7082])
    ax.set_xticklabels(["1", "10", "100", "1,000", "7,082"])
    ax.set_yticks([10, 100, 1000])
    ax.set_yticklabels(["10", "100", "1,000"])
    ax.set_xlabel("M（token 数，对数轴）", color=t["ink2"], fontsize=9)
    ax.set_ylabel("吞吐 GOPS（对数轴）", color=t["ink2"], fontsize=9)
    leg = ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3, frameon=False, fontsize=8.8)
    for txt in leg.get_texts():
        txt.set_color(t["ink2"])
    save(fig, "npu_matmul_shapes", t_name, t)


# ---------------------------------------------------------------- 5. NPU load
def fig_load(t_name, t):
    run = ROOT / "results/operator/board/ope4/ope4_segment1_20260921T130538+0800"
    load = list(csv.DictReader(open(run / "npu_load_10hz.csv")))
    prompts = list(csv.DictReader(open(run / "c02_a2/prompt_summary.csv")))
    win = [(int(p["run_start_ns"]), int(p["run_return_ns"]), int(p["prompt_id"])) for p in prompts]
    o, e = win[0][0], win[-1][1]
    sel = [r for r in load if o <= int(r["monotonic_raw_ns"]) <= e]
    xs = [(int(r["monotonic_raw_ns"]) - o) / 1e9 for r in sel]
    ys = [float(r["npu_load_core0_pct"]) for r in sel]
    others = max(max(float(r["npu_load_core1_pct"]), float(r["npu_load_core2_pct"])) for r in sel)
    mean = sum(ys) / len(ys)
    fig, ax = plt.subplots(figsize=(10, 3.6))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.72, bottom=0.17)
    titles(fig, t, f"板端运行：NPU 单核持续高负载（平均 {mean:.1f}%）",
           f"本项目模型 · W8A8 · 16K 上下文 · 5 条固定提示；10 Hz 采样；其余两个 NPU 核最高负载 {others:.0f}%（单核运行）")
    style_axes(ax, t)
    for i, (s0, s1, pid) in enumerate(win):
        if i % 2 == 0:
            ax.axvspan((s0 - o) / 1e9, (s1 - o) / 1e9, color=t["panel"], lw=0)
        ax.text(((s0 + s1) / 2 - o) / 1e9, 103, f"提示 {pid}", fontsize=8.5, color=t["ink2"], ha="center")
    ax.plot(xs, ys, color=t["blue"], lw=1.6)
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlim(0, (e - o) / 1e9)
    ax.set_xlabel("时间（秒）", color=t["ink2"], fontsize=9)
    save(fig, "npu_load", t_name, t)


# ---------------------------------------------------------------- 6. GPU levels
def fig_gpu(t_name, t):
    # K1: results/operator/k1/op2_receipt.json (2.513462×); model TPOT 146.330 -> 127.332 ms (op3 receipt)
    # K3: day-1 1.275002×; reconnect TPOT 138.568327 -> 141.700198 ms; lm_head paired TPOT 1.196628×
    rows = [("RMSNorm（Triton）", 2.513462, 146.330 / 127.332, "输出与原路径分叉：不计为无损提速", False),
            ("GQA decode attention（Triton）", 1.275002, 138.568327 / 141.700198, "输出分叉且整体更慢：未采用", False),
            ("lm_head GPU 镜像", None, 1.196628, "逐 token 输出一致：采用", True)]
    fig, ax = plt.subplots(figsize=(10, 3.9))
    fig.subplots_adjust(left=0.24, right=0.66, top=0.7, bottom=0.18)
    titles(fig, t, "GPU 路径：算子级加速不等于模型级收益",
           "RTX 4060 Laptop · 本项目模型 NF4 路径；算子级为隔离测试（较最强基线），模型级为每 token 延迟之比；竖线为 1.0×")
    style_axes(ax, t, grid_axis="x")
    ax.axvline(1.0, color=t["axis"], lw=1.2)
    for i, (name, kern, model, note, ok) in enumerate(rows):
        y = 2 - i
        if kern is not None:
            ax.plot([model, kern], [y, y], color=t["gray1"], lw=2, solid_capstyle="round", zorder=2)
            ax.scatter([kern], [y], s=70, color=t["blue"], edgecolor=t["surface"], linewidth=2, zorder=3,
                       label="算子级（隔离测试）" if i == 0 else None)
            ax.text(kern, y + 0.2, f"{kern:.2f}×", fontsize=8.8, color=t["ink"], ha="center")
        ax.scatter([model], [y], s=70, color=t["orange"], edgecolor=t["surface"], linewidth=2, zorder=3,
                   label="模型级（每 token 延迟）" if i == 0 else None)
        ax.text(model, y - 0.36, f"{model:.2f}×", fontsize=8.8, color=t["ink"], ha="center")
        ax.text(2.78, y, "✓" if ok else "✗", fontsize=10, color=t["good"] if ok else t["bad"], va="center",
                ha="left", clip_on=False, fontweight="bold")
        ax.text(2.86, y, note, fontsize=9, color=t["ink"], va="center", ha="left", clip_on=False)
    ax.set_yticks([2, 1, 0])
    ax.set_yticklabels([r[0] for r in rows], color=t["ink"], fontsize=9.3)
    ax.set_xlim(0.85, 2.7)
    ax.set_ylim(-0.7, 3.1)
    ax.set_xticks([1.0, 1.5, 2.0, 2.5])
    ax.set_xticklabels(["1.0×", "1.5×", "2.0×", "2.5×"])
    leg = ax.legend(loc="upper left", bbox_to_anchor=(-0.33, 1.3), ncol=2, frameon=False, fontsize=9)
    for txt in leg.get_texts():
        txt.set_color(t["ink2"])
    save(fig, "gpu_kernel_vs_model", t_name, t)


# ---------------------------------------------------------------- 7. SFT loss
def fig_sft(t_name, t):
    # results/m1/c01_s_completion_audit.md §2, token-weighted holdout loss
    steps = [0, 1024, 2048, 3072, 4096]
    loss = [1.6302139, 0.8784115, 0.8543855, 0.8415895, 0.8385551]
    fig, ax = plt.subplots(figsize=(10, 3.6))
    fig.subplots_adjust(left=0.07, right=0.97, top=0.72, bottom=0.17)
    titles(fig, t, "SFT 验证集损失：1.63 → 0.84",
           "1,948 个验证窗口（按源会话隔离，与训练集交集为 0）；token 加权交叉熵；5 次完整评测")
    style_axes(ax, t)
    ax.plot(steps, loss, color=t["blue"], lw=2, solid_capstyle="round")
    ax.scatter(steps, loss, s=40, color=t["blue"], edgecolor=t["surface"], linewidth=2, zorder=3)
    ax.text(steps[0] + 60, loss[0], f"{loss[0]:.3f}", fontsize=9, color=t["ink"], va="center")
    ax.text(steps[-1], loss[-1] + 0.06, f"{loss[-1]:.3f}", fontsize=9, color=t["ink"], ha="center")
    ax.set_xticks(steps)
    ax.set_xticklabels([f"{s:,}" for s in steps])
    ax.set_ylim(0.6, 1.8)
    ax.set_xlabel("optimizer step", color=t["ink2"], fontsize=9)
    ax.set_ylabel("验证集 loss", color=t["ink2"], fontsize=9)
    save(fig, "sft_holdout_loss", t_name, t)


if __name__ == "__main__":
    for name, theme in THEMES.items():
        for fn in (fig_arch, fig_iova, fig_neon, fig_npu, fig_load, fig_gpu, fig_sft):
            fn(name, theme)
    print("wrote", len(list(OUT.glob("*.png"))), "png files to", OUT)

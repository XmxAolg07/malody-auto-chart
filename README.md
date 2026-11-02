# Malody 4K 自动写谱 — 阶段 1：音频分析核心

当前进度：**BPM 识别** 和 **波形图** 两个功能已完成，PC 端可跑通全流程。
安卓端与写谱本体尚未开始（等确定技术路线）。

> 源数据保护：`ex jack/`、`reg speed/` 等 8 个文件夹里的 `.ogg` / `.mc` / `.jpg`
> 都是真实谱面素材，脚本**只读**，不做任何修改。所有生成物写到 `out/`；
> 以后需要改谱面或音频，先复制到 `work/` 再动副本。

---

## 1. 现在能做什么

| 功能 | 说明 |
|---|---|
| 音频解码 | ogg / mp3 / wav / flac / opus；soundfile 优先，找不到时自动用 ffmpeg |
| BPM 识别 | 输出 BPM、首拍 offset、拍点时间表、置信度，以及 ½× / ⅔× / 1.5× / 2× 倍频候选 |
| 波形图 | 整曲 min/max/RMS 包络渲染成 PNG；另有放大视图，带拍线 |
| 数据输出 | JSON（拍点、包络数组、音频信息），供后续写谱和 UI 直接使用 |

一首 4 分钟的曲子完整分析约 **2–3 秒**。

## 2. 快速开始

```bat
run.cmd "reg stream\0\Camellia - Spin Eternally.ogg"
```

打包版：把音频拖到 `dist\MalodyAudioAnalyzer\MalodyAudioAnalyzer.exe` 上即可（见第 8 节）。

`run.cmd` 会自动找本机自带 numpy / Pillow / soundfile 的 Python；
如果你用自己的 Python，先装依赖：

```bat
pip install -r requirements.txt
python cli.py "歌曲.ogg" -o out --width 2000 --zoom 20
```

已生成的演示结果在 `out/` 里（`样例曲目 Silentroom - Nhelv`、`Camellia - Spin Eternally`）。

参数：

- `-o out`：输出目录
- `--width 2000`：波形图宽度（像素）
- `--zoom 20`：额外输出前 20 秒放大图（带拍线，方便核对 offset）
- `--duration 120`：只分析前 120 秒（快速预览）
- `--bpm-min/--bpm-max`：BPM 搜索范围
- `--prior-center/--prior-octaves/--density-weight`：节奏先验（默认已按 4K 谱面调好，一般不动）

## 3. 输出文件

```
out/
├─ 歌曲.analysis.json      # 完整分析结果
├─ 歌曲.waveform.png       # 整曲波形 + 刻度
└─ 歌曲.waveform.zoom.png  # 前 20 秒放大 + 拍线
```

`analysis.json` 结构：

```jsonc
{
  "file": "...",
  "audio": { "duration": 230.28, "samplerate": 44100, "channels": 2, "format": "OGG", ... },
  "rhythm": {
    "bpm": 221.993,
    "offset_ms": 250.5,          // 首拍相位（0 ~ 一拍长度）
    "confidence": 0.715,
    "beat_strength": 3.6,
    "coverage": 0.72,
    "candidates": [              // 第一个是选中值，其余是倍频备选
      { "bpm": 221.993, "score": 0.72, "confidence": 1.0 },
      { "bpm": 111.0,  "score": 0.55, "confidence": 0.76 }
    ],
    "beats_ms": [250.5, 520.8, ...]   // 拍点时间（毫秒）
  },
  "waveform": { "min": [...], "max": [...], "rms": [...], "width": 2000 }
}
```

## 4. 算法（为什么不是简单找峰值）

1. **STFT**：4096 点窗、约 23 ms 步进，40 带 mel 滤波器组。
2. **起音强度**：对数 mel 谱的谱通量；同时单独算 200 Hz 以下低频带（底鼓）。
3. **候选 BPM**：对起音包络做自相关，再用 1×/2×/3×/4× 谐波梳状打分，得到候选表。
4. **倍频纠正**：对每个候选做相位扫描，算
   `覆盖率 × 网格密度 × 节奏先验`。
   - 覆盖率 = 该网格捕获到的起音能量占比（半速网格会漏掉八分音符）；
   - 网格密度 = 有起音的拍点占比（半速网格会有一半空拍）。
   这一步把 1/2、2/3 倍频的误判纠正回来。
5. **细调**：BPM 在 ±1.5% 内细化，相位在 ±1/40 拍内细化，得到最终 offset。
6. **波形**：按列取 min/max/RMS，Pillow 直接绘制（不依赖 matplotlib）。

## 5. 验证结果

**合成信号**（可精确控制真值，90–240 BPM，每拍一个重音）：
12 组全部命中，offset 误差约 20–25 ms。

**真实谱面 40 首**（以 `.mc` 里标注的 BPM 为参考答案，只读音频）：

```
exact rate: 85.0%   exact+octave: 95.0%
```

失败案例集中在**变速曲目**（一首歌里写了多个 BPM 的段位谱），
以及 330 BPM 这类被记成 165 的极高速曲子。前者需要分段 BPM 检测（下一步）。

复现：

```bat
python tests\test_synthetic.py
python tests\bench_charts.py --limit 40 --duration 120
```

## 6. 已知限制

- 全曲只给**一个恒定 BPM**；变速曲需要分段分析。
- BPM 低于 80 且没有八分/十六分音符时，可能报成双倍；`candidates` 里会给出减半值，后续 UI 可一键切换。
- offset 约 ±25 ms（受起音检测延迟影响），写谱前建议在编辑器里微调。
- 目前只有命令行和 exe，没有图形界面，也没有安卓端。

## 7. 目录结构

```
自动写谱/
├─ core/
│  ├─ audio.py       # 解码、重采样、元数据
│  ├─ bpm.py         # 起音检测、BPM/offset/拍点
│  ├─ waveform.py    # 包络提取、PNG 渲染
│  └─ analyze.py     # 一条龙分析入口
├─ cli.py            # 命令行
├─ run.cmd           # Windows 启动器（拖拽文件即可）
├─ build_exe.cmd     # PyInstaller 打包脚本
├─ tests/
│  ├─ test_synthetic.py   # 合成信号回归测试
│  ├─ bench_charts.py     # 用 .mc 标注的 BPM 做真实曲目基准
│  └─ tune_prior.py       # 调参用的探索脚本
├─ vendor/           # soundfile + cffi（wheel 解开，免安装）
└─ out/              # 生成结果
```

## 8. 打包 Windows exe

已生成：`dist\MalodyAudioAnalyzer\`（onedir 免安装版）与 `dist\MalodyAudioAnalyzer-win64.zip`（约 31 MB）。

- 用法：把音频拖到 `MalodyAudioAnalyzer.exe` 上，结果写到同目录 `out\`
- **整个文件夹一起拷贝**，不能只拿 exe
- 重新打包：

```bat
build_exe.cmd "C:\path\to\python.exe"
```

采用 onedir 而非 onefile：本机安全软件会拦截 onefile 自解压时往临时目录写 DLL
（报 `fopen: Permission denied`）。onedir 不需要解压，启动也更快。

## 9. 下一步（待定）

1. **变速曲 BPM 分段**：滑动窗口 + 变更点检测，输出 `time` 数组（Malody 格式）。
2. **4K 谱面生成**：按 vibro / stream / jack / tech 风格出 note，再导出 `.mc`。
3. **PC 界面**：波形 + 拍线 + 试听 + 参数调整。
4. **安卓端**：两条路线二选一
   - A. Kotlin 原生（PC 也用 Compose Desktop，算法只用维护一份 Kotlin 版）；
   - B. Chaquopy 内嵌 Python（复用现在的算法，Android 原生解码后把 PCM 交给 Python）。

> 建议先做 1（变速曲）和 3（PC 界面），把“分析 → 预览 → 生成 → 导出 .mc”闭环跑通，
> 再决定安卓用哪条路线。

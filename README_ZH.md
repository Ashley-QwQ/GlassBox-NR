# GlassBox-NR v2.0.1-preview

[English](README.md)

**对齐 `nvngx_dlssnr.dll` 310.8.0.0。** 这是针对该固定目标库推理链的独立 **CPU 参考实现**（CPU reference implementation，C++），不覆盖其他版本（包括 310.9.16）。许可：Apache-2.0（见 `NOTICE`）。

本版的声明是**实测逐字节等价（bit-exact）**，限于所列输入、配置与硬件：1080p / 1440p / 4K 的帧 0（各 158 次 launch），以及多帧链 **1080p f0–f5**、**1440p f0–f3** 与 **4K f0–f2**（两种工作区填充）上，CPU 自有链都复现了 GPU 采集的终图，逐字节相同。形式等价证明 **NOT_ESTABLISHED**，本仓库任何地方都不写"已证明"。

您需要自备合法获得的 `nvngx_dlssnr.dll`。DLL、权重以及由权重字节派生的任何内容都不在本仓库和发布附件里。

## 仓库内容（git HEAD）

| 目录 | 内容 |
|---|---|
| `cpp/` | 链各模块的 C++ 源码（10 个模块，57 个文件） |
| `runtime/`、`third_party/mlxdlss/tools/` | Python 运行时胶水（加载器、ABI、pin）与权重提取器（MLX-DLSS，Apache-2.0，原样） |
| `devkit/` | 面向用户的 1080p 帧 0 运行驱动，以及 `payload_layout.py` |
| `params/` | 逐 launch 参数文件（329 个） |
| `modules/` | 编译好的 Python 扩展模块（10 个，`cpython-313-x86_64-linux-gnu`） |
| `docs/FILES_MAP_V2.json` | 仓库路径 -> 发布白名单路径，`payload_layout.py` 使用 |

其余内容（按路径计约 32 GB，去重后 16.6 GB）是**载荷**：一个压缩分卷 `payload.tar.zst.001`（462,661,132 B），作为 GitHub Release 附件提供。v1 的 Python 工具链（`v0.1.0-preview`）已不在 HEAD，仍可通过 git tag `v0.1.0-preview`（提交 `8f951866`）取得。

## 快速开始（Linux x86-64，Python 3.13，numpy，safetensors，zstd >= 1.4）

1. 从 GitHub Release [`v2.0.1-preview`](https://github.com/Ashley-QwQ/GlassBox-NR/releases/tag/v2.0.1-preview) 把 `payload.tar.zst.001` 下载到本仓库根目录。清单已在 git 里：`devkit/data/PAYLOAD_MANIFEST.json`。
2. 校验分卷：
   ```
   echo "c86119619d0638e7bfa94b6ab84f6dd465bba132aaf9d9b4130c648082455b8c  payload.tar.zst.001" | sha256sum -c -
   ```
3. 解包（必须带 `--long=27`；约需 16.6 GB 磁盘）：
   ```
   mkdir extract && cd extract && zstd -d --long=27 < ../payload.tar.zst.001 | tar -x && cd ..
   ```
4. 把文件摆到驱动期望的位置（硬链接；git 里的文件取自本检出）：
   ```
   python devkit/payload_layout.py --manifest devkit/data/PAYLOAD_MANIFEST.json --extracted extract --repo . --out PAYLOAD_DIR --tables TABLES_DIR
   ```
5. 用您自己的 DLL 生成全部 5 个权重文件（`weights.bin`、`device_weights.bin`、`weights-device.bin`、`weights_gpu.bin`、`weights-packed.safetensors`；仅标准库，约数秒；脚本会对照 manifest 的 size/sha256 打印 OK/FAIL，且拒绝覆盖已有文件）：
   ```
   python devkit/make_weights.py /path/to/nvngx_dlssnr.dll WEIGHTS
   ```
   `WEIGHTS` 即下一步的 `--weights` 目录。DLL 和生成的权重不属于本发布，请勿上传。
6. 运行 1080p 帧 0 链（158 次 launch）并与参考表比对。两个帧 0 输入文件来自 payload，第 4 步会把它们放到下面 `--inputs` 指定的路径：
   ```
   python devkit/run_1080_f0.py --root WORK --payload PAYLOAD_DIR --weights WEIGHTS --tables TABLES_DIR \
       --inputs PAYLOAD_DIR/kit/data/video-multiframe-r01-20261006-codex/resume2-20261007/chain_port/chain-input-only-v2/frame-000 --threads 4
   ```
   加 `--verify-only` 只做重定位和文件检查，不运行。退出码 0 表示每个 launch 与参考表一致，且终图 sha256 为 `764e7dd0d8f440b5afeb36e72fea3d8bc137bef891e8f09dc3476b6fef8d34ef`；2 表示某个 launch 不一致，3 表示终图不一致，4 表示回执不完整（必须恰好按顺序含 launch 0..stop-after 各一次）。`python devkit/test_run_1080_f0.py` 不需要 payload 即可自测这项比较。

## 声明分层

发布封存：**M2**，统一封存 M（内容基底 `9d0a4593...`）的 release-clean 子封存：不含反汇编派生数据，`repack` 重写为闭式地址映射，派生 JSON 已剥离，路径守卫字面量改为 XOR 编码。
v2.0.1-preview 是 M2 加上 HMMA 零符号修复（见 `CHANGELOG.md`）：改了三个模块和六个 C++ 源文件，载荷随之重钉。v2.0.1 上只重跑了 1080p f0（用户驱动，从新卷端到端：158/158，终图 `764e7dd0...`）；下表其余各行都是在 v2.0.0 的模块上测得的。

| 层 | 范围 | 内容 |
|---|---|---|
| 在 **M2** 上跑过的数值门 | 1080p f0、1440p f0、4K L0-L99 | 1080 f0：158 launch 字节相等，终图 `764e7dd0...`。1440 f0：158 launch MATCH，终图 `20b4f026...`。4K L0-L99：完整工作区加六 POST 纹理，各段 PASS |
| 用户驱动，只用载荷 | 1080p f0 | 从解包载荷加 git 文件端到端：158/158 launch 命中参考表，终图 `764e7dd0...` |
| M2 上**只有配置普查**（无数值门） | 4K L100-L157、1080 f1-f5、1440 f3 | 1,738 条普查记录与 M 逐条相同（映射登记的路径 / sha 除外）；M2 上未做数值重跑 |
| 取自 **M**（或其父封存 854a8724），未在 M2 上重测 | 4K f0 L0-L157 终图 `152d6f07...`（带发布说明列出的限制）、1440p f0-f3、1080p f0-f5、MSVC 构建（1080 f0）、非有限值线（held-out 144/147 终图相同，3 张由守卫拒绝）、scene cut f0-f15、3/3 预先登记预测（登记证据是 git 提交时间，不是第三方时间戳） | |
| v2.0.0-preview 发布后在 **M** 上实测（2026-10-22） | 4K f0–f2，两种填充 | 每种填充一个进程从 reset 连跑 474 launch，只用 CPU 自有状态：每个观察点完整工作区与六 POST 一致，终图 f0 `152d6f07...`、f1 `9aa7d914...`、f2 `cc32bdf4...`；f0/f1 第二次运行完全复现；WR B(i) 仅 f0 L0/L1 因原生快照缺口未评估 |
| 封存考题测试（2026-10-24）：代码先冻结，预测在打开答案前提交 git | 1080p：一段开发中从未见过的封存 32 帧序列，外加一张封存单帧 | 32 帧序列终图 32/32 逐字节相同（帧 0 与已知序列同输入，未见部分为帧 1–31：31/31）。单帧在封存 M 上被非有限值守卫拒绝；在补上 +Inf 语义的 A1 子封存上 MATCH。合计 32 MATCH / 1 REFUSED / 0 DIFF。序列内参数固定；改参数路径与 1440p/4K 封存考题尚未进行 |
| 未覆盖 | 4K f3 及以后（没有逐 launch 原生采集）、1440p f4/f5、1440p/4K held-out 输入、其他 DLL 版本、所列之外的多参数流、正式计时 | NOT_RUN |
| 形式等价证明 | | **NOT_ESTABLISHED** |

发布说明里的速度与内存数字都是小样本（每项 1–2 次），不是统计结论。本版不含任何展示图。

## 实测平台

下表每次都跑 1080p 帧 0 一帧（158 次 launch），终图都是 `764e7dd0...`。每个数字只跑了一次，线程数、机器负载和构建都不一样，不是基准测试，几行之间也不能互相比较。

| 平台 | CPU | 线程 | 构建 | 一帧用时 |
|---|---|---|---|---|
| X79 | Xeon E5-2667 v2（8C/16T） | 4 | g++，封存 M2 | 2,822 s |
| 台式机 | Core i5-12400F | 12 | MSVC，较早的封存 `ea731776` | 691 s（8 次均值；WSL/GCC 899 s） |
| 小主机 | Core i5-10600T（WSL） | 6 | g++ | 2,566 s |
| 笔记本 | Core i7-1280P（只用 P 核） | 12 | MSVC，封存 M | 2,522 s，峰值 0.83 GiB |
| 特别嘉宾 | **树莓派 3B+**（Cortex-A53，**1 GB 内存**） | - | 原生 g++ 14.2，封存 M | **10,283 s（约 2.9 小时）**，峰值 RSS **677 MiB**；只核终图，没有逐 launch 门 |

## 输出尾段的出处

尾段 B70 / S157 的逻辑经分析 NVIDIA 二进制得到，代码为独立重写，经 GPU 实测逐字节验证。见 `docs/B70_S157_STATUS.md` 与 `NOTICE`。

## 优先权与历史

`PRIORITY.md` 说明日期方面能证明什么、不能证明什么，以及如何对照 tag `v0.1.0-preview` 核对 v1 承诺。

## 许可

代码与数据：**Apache-2.0**（`LICENSE`）。文档：**CC-BY-4.0**（`LICENSE-docs`）。`third_party/mlxdlss/`：MLX-DLSS，保留其自身的 Apache-2.0 许可与声明。详见 `NOTICE`。

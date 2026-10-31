# Changelog

## v2.0.2-preview (2026-10-31)

Run-driver fix only. Modules, numeric C++ sources, parameter files and the payload volume are unchanged: use `payload.tar.zst.001` from the [v2.0.1-preview release](https://github.com/Ashley-QwQ/GlassBox-NR/releases/tag/v2.0.1-preview) (462,661,132 B, sha256 `c8611961...`). Nothing was re-measured for this release.

- **Run driver receipt check** (`devkit/run_1080_f0.py`): before comparing hashes, the driver now requires the worker receipt to hold exactly the launches 0..`--stop-after`, each once and in order, with a matching count in `result.json`; otherwise it exits 4 (`INCOMPLETE RECEIPT`). Previously an empty, partial, duplicated or reordered receipt could be reported as `ALL n LAUNCHES MATCH`. `--stop-after` is limited to 0..157. New stdlib-only self-test `devkit/test_run_1080_f0.py` (11 cases). No numeric code, module or payload changed. Reported by an external implementability review (2026-10-31).

---

## v2.0.1-preview (2026-10-28)

**HMMA zero-sign restore.** At the end of the HMMA helpers (`hmma_seed`, `hmma_dot`, `hmma_cached`) a half-precision −0 (`0x8000`) is now published as +0, as the GPU does. This canonicalisation was in the historical seal `49a98a0b` but missing from M / M2 and from v2.0.0-preview (`e4a9b42`); on the saved sm_120 sweep it is 480 of 1,380,992 cases (first case 307200). The general hardware rule (v1) stays OPEN.

- **Changed**: 6 C++ sources (7 paths: `chain_entry_optm/{decode_reuse.inc,mapexec_base.hpp}`, `chain_outtail_optm/{outtail.cpp,primitives.hpp}`, `chain_split16_optm/scalar_map.hpp`, `chain_tin_optm/{decode_reuse.inc,primitives.inc}`); modules `entry`, `outtail`, `tin` rebuilt with the M2 recipe (`split16` module bytes unchanged); `runtime/M/derived_pins.py` (derived-map pin).
- **New payload volume** `payload.tar.zst.001` (462,661,132 B, sha256 `c8611961...`): same 7,522 members and file set; every text member that pinned an old module or source hash was re-pinned in dependency order (4,822 members), binary members untouched. Whitelist revision v2.3. The re-pinned seal and receipt files carry the new hashes but were not re-measured; the v2.0.0-preview volume stays the original evidence.
- **Gates on v2.0.1**: unit gate on the sm_120 sweep (0 mismatches in all three helpers, was 480 in `hmma_seed`); 1080p f0 end-to-end from the new volume, 158/158 launches, final `764e7dd0...` (unchanged). All other claims are from M2 / M on the v2.0.0 modules and were not re-run.

---

## v2.0.0-preview (2026-10-21)

**Aligned to `nvngx_dlssnr.dll` 310.8.0.0.** Claim: measured byte-for-byte equivalence on the listed inputs; formal proof NOT_ESTABLISHED.

- **Independent CPU C++ chain**: all 158 launches of the chain, including the output-tail stage (B70 / S157, independently written C++; provenance statement in `NOTICE` and `docs/B70_S157_STATUS.md`), at 1080p, 1440p and 4K frame 0.
- **Release seal M2** (release-clean sub-seal of the unified seal M): numeric gates re-run on M2 for 1080p f0, 1440p f0 and 4K L0-L99; 4K L100-L157, 1080p f1-f5 and 1440p f3 are census-only on M2. Evidence from M or its parent seal is labelled as such in the README.
- **Two-part distribution**: git HEAD holds the C++ sources, Python runtime glue, run driver (`devkit/`), per-launch parameters and compiled modules; everything else is one compressed payload volume, `payload.tar.zst.001` (465,105,242 B), a GitHub Release asset. `devkit/payload_layout.py` lays it out; `docs/FILES_MAP_V2.json` maps repo paths to release-whitelist paths.
- **Licence change**: code and data Apache-2.0 (was AGPL-3.0-or-later), documentation CC-BY-4.0, `third_party/mlxdlss/` keeps its own Apache-2.0.
- **v1 toolchain removed from HEAD**: the Python reference package (`glassbox_nr/`, `tools/`, `data/`, `commitment/`, `reference/`, v1 docs) is no longer in HEAD. History is not rewritten; all of it stays reachable through the tag `v0.1.0-preview` (commit `8f951866`).
- **Cleanup**: derived JSON stripped, instruction-level comments removed, `repack` rewritten as a closed-form address map; the offline interpreter tools are no longer published.
- **Not in this release**: showcase renders; 4K multi-frame; other DLL versions.

---

## v2.0.2-preview（2026-10-31）

只改运行驱动。模块、数值 C++ 源码、参数文件和载荷卷都没变：载荷用 [v2.0.1-preview release](https://github.com/Ashley-QwQ/GlassBox-NR/releases/tag/v2.0.1-preview) 里的 `payload.tar.zst.001`（462,661,132 B，sha256 `c8611961...`）。本版没有重新测量。

- **运行驱动的回执检查**（`devkit/run_1080_f0.py`）：比较哈希之前，先要求 worker 回执恰好按顺序含 launch 0..`--stop-after` 各一次，且 `result.json` 的计数一致；否则以退出码 4（`INCOMPLETE RECEIPT`）结束。此前空的、缺项的、重复的或乱序的回执都可能被报告为 `ALL n LAUNCHES MATCH`。`--stop-after` 限定为 0..157。新增只依赖标准库的自测 `devkit/test_run_1080_f0.py`（11 个用例）。数值代码、模块和载荷都没改。来自 2026-10-31 的一次外部可实现性审查。

---

## v2.0.1-preview（2026-10-28）

**HMMA 零符号修复。** HMMA 辅助函数（`hmma_seed`、`hmma_dot`、`hmma_cached`）在末端把半精度 −0（`0x8000`）发布为 +0，与 GPU 一致。这一规范化在历史封存 `49a98a0b` 里有，但 M / M2 和 v2.0.0-preview（`e4a9b42`）缺失；在保存的 sm_120 扫描中涉及 1,380,992 例里的 480 例（首例 307200）。一般硬件规则（v1）仍为 OPEN。

- **改动**：6 个 C++ 源文件（7 个路径：`chain_entry_optm/{decode_reuse.inc,mapexec_base.hpp}`、`chain_outtail_optm/{outtail.cpp,primitives.hpp}`、`chain_split16_optm/scalar_map.hpp`、`chain_tin_optm/{decode_reuse.inc,primitives.inc}`）；`entry`、`outtail`、`tin` 三个模块按 M2 配方重编（`split16` 模块字节不变）；`runtime/M/derived_pins.py`（派生映射钉）。
- **新载荷分卷** `payload.tar.zst.001`（462,661,132 B，sha256 `c8611961...`）：成员数与文件集同为 7,522；凡钉了旧模块或旧源码哈希的文本成员都按依赖顺序重钉（4,822 个），二进制成员不动。白名单修订 v2.3。重钉后的封存与回执文件带的是新哈希，但没有重新测量；v2.0.0-preview 的分卷仍是原始证据。
- **v2.0.1 上的门**：sm_120 扫描上的单元门（三个函数均 0 不一致，原 `hmma_seed` 480 例）；从新卷端到端跑 1080p f0，158/158 launch，终图 `764e7dd0...`（不变）。其余声明来自 v2.0.0 模块上的 M2 / M，未重跑。

---

## v2.0.0-preview（未发布）

**对齐 `nvngx_dlssnr.dll` 310.8.0.0。** 声明：限于所列输入的实测逐字节等价；形式证明 NOT_ESTABLISHED。

- **独立 CPU C++ 链**：1080p、1440p、4K 帧 0 的全部 158 次 launch，含输出尾段（B70 / S157，独立编写的 C++；出处声明见 `NOTICE` 与 `docs/B70_S157_STATUS.md`）。
- **发布封存 M2**（统一封存 M 的 release-clean 子封存）：1080p f0、1440p f0、4K L0-L99 的数值门在 M2 上重跑；4K L100-L157、1080p f1-f5、1440p f3 在 M2 上只有配置普查。取自 M 或其父封存的证据在 README 里如实标注。
- **两部分发布**：git HEAD 放 C++ 源码、Python 运行时胶水、运行驱动（`devkit/`）、逐 launch 参数和编译模块；其余是一个压缩载荷分卷 `payload.tar.zst.001`（465,105,242 B），作为 GitHub Release 附件。`devkit/payload_layout.py` 负责摆放，`docs/FILES_MAP_V2.json` 记录仓库路径到发布白名单路径的映射。
- **许可变更**：代码与数据改为 Apache-2.0（原 AGPL-3.0-or-later），文档 CC-BY-4.0，`third_party/mlxdlss/` 保留其自身的 Apache-2.0。
- **v1 工具链移出 HEAD**：Python 参考包（`glassbox_nr/`、`tools/`、`data/`、`commitment/`、`reference/`、v1 文档）不再在 HEAD；历史不改写，全部仍可通过 tag `v0.1.0-preview`（提交 `8f951866`）取得。
- **清理**：派生 JSON 已剥离，指令级注释已删除，`repack` 重写为闭式地址映射；离线解释器工具不再发布。
- **本版不含**：展示图；4K 多帧；其他 DLL 版本。

---

## Unreleased

- **Reference hashes**: `reference/reference_hashes.json` publishes the sha256 of the reference output at all 154 compared exits (77 exits x 2 published samples), with the reference identity (runtime DLL, `WEIGHTS_HT`, capture GPU and architecture). Hashes only; no reference data. `tools/verify_reference_hashes.py` checks a run against it and prints `N/154 MATCH`.
- **Input generator**: `tools/make_inputs.py` generates the two published samples' colour inputs procedurally and checks them against the sha256 the reference results were produced from.
- **Documentation**: added `docs/SCOPE.md`, `docs/NUMERICAL_EXACTNESS.md`, `docs/B70_S157_STATUS.md`, `docs/REPRODUCIBILITY.md` and `PRIORITY.md`; removed README and `docs/SAMPLE_NAMES.md` references to files that are not part of this repository; the README headline now states the covered range explicitly.
- **Citation metadata**: `CITATION.cff` now lists the author, the repository URL, the code license (AGPL-3.0-or-later) and the public release date (2026-09-17).
- **Fix: Windows clones**: added `.gitattributes` (`* -text`). With Git for Windows' default `core.autocrlf=true`, a fresh clone converted line endings, and the package's own sha256 checks then refused several of its data files, stopping the run at B23. Files are now checked out exactly as committed.
- **Changelog correction**: the 2026-09-14 entry below previously read "Initial public results statement". There is no public, third-party-verifiable record of that date; the entry now describes it as an internal milestone. See `PRIORITY.md`.

---

## Unreleased（中文）

- **参照哈希**：`reference/reference_hashes.json` 发布全部 154 个比对出口（2 个已发布样本 × 77 个出口）参照输出的 sha256，并记录参照身份（运行时 DLL、`WEIGHTS_HT`、采集 GPU 与架构）。只含哈希，不含任何参照数据。`tools/verify_reference_hashes.py` 据此核对一次运行，输出 `N/154 MATCH`。
- **输入生成器**：`tools/make_inputs.py` 以程序化方式生成两个已发布样本的彩色输入，并与生成参照结果时所用输入的 sha256 核对。
- **文档**：新增 `docs/SCOPE.md`、`docs/NUMERICAL_EXACTNESS.md`、`docs/B70_S157_STATUS.md`、`docs/REPRODUCIBILITY.md` 与 `PRIORITY.md`；删除 README 与 `docs/SAMPLE_NAMES.md` 中指向本仓库之外文件的引用；README 首段改为明确写出覆盖范围。
- **引用元数据**：`CITATION.cff` 补全作者、仓库地址、代码许可证（AGPL-3.0-or-later）与公开发布日期（2026-09-17）。
- **修复：Windows 上的 clone**：新增 `.gitattributes`（`* -text`）。在 Git for Windows 默认的 `core.autocrlf=true` 下，新 clone 会转换行尾，本包自带的 sha256 校验随即拒绝若干数据文件，运行在 B23 停止。现在所有文件均按提交时的字节原样检出。
- **更正**：下方 2026-09-14 条目原先把它写成了一次公开结果声明。该日期没有任何公开、可由第三方核验的记录，现改为如实描述为内部里程碑。见 `PRIORITY.md`。

---

## 2026-09-17

- **Runnable preview package**: Initial runnable preview covering nodes B00 through B69 of the 73-node pipeline.
- **Verification scope**: Byte-identical output against reference results across all 77 compared exits for the two published samples (`c1-C-img-full`, `d23-n2-nat320x240`).
- **Tail nodes excluded**: Nodes B70 and S157 are not included; the package does not currently produce the pipeline's final image.
- **Licensing**: Source code licensed under AGPL-3.0-or-later; documentation and data files licensed under CC-BY-4.0; `third_party/mlxdlss/` retains Apache-2.0.

---

## 2026-09-17（中文）

- **可运行预览包**：发布可运行预览包，覆盖 73 节点推理链中的节点 B00 至 B69。
- **验证范围**：两个发布样本（`c1-C-img-full`、`d23-n2-nat320x240`）在全部 77 个比对出口与参考结果逐字节相同。
- **未包含尾部节点**：未包含节点 B70 与 S157；本软件包当前不产出最终图像。
- **许可证**：源代码采用 AGPL-3.0-or-later 许可；文档与数据文件采用 CC-BY-4.0 许可；`third_party/mlxdlss/` 保留 Apache-2.0 许可。

---

## 2026-09-14

- Internal results milestone (not publicly released). This repository was made public on 2026-09-17.

---

## 2026-09-14（中文）

- 内部结果里程碑（未公开发布）。本仓库于 2026-09-17 公开。

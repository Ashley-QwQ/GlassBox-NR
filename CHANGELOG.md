# Changelog

## Unreleased

- **Reference hashes**: `reference/reference_hashes.json` publishes the sha256 of the reference output at all 154 compared exits (77 exits x 2 published samples), with the reference identity (runtime DLL, `WEIGHTS_HT`, capture GPU and architecture). Hashes only; no reference data. `tools/verify_reference_hashes.py` checks a run against it and prints `N/154 MATCH`.
- **Input generator**: `tools/make_inputs.py` generates the two published samples' colour inputs procedurally and checks them against the sha256 the reference results were produced from.
- **Documentation**: added `docs/SCOPE.md`, `docs/NUMERICAL_EXACTNESS.md`, `docs/B70_S157_STATUS.md`, `docs/REPRODUCIBILITY.md` and `PRIORITY.md`; removed README and `docs/SAMPLE_NAMES.md` references to files that are not part of this repository; the README headline now states the covered range explicitly.
- **Citation metadata**: `CITATION.cff` now lists the author, the repository URL, the code license (AGPL-3.0-or-later) and the first public release date (2026-09-17).
- **Changelog correction**: the 2026-09-14 entry below previously read "Initial public results statement". There is no public, third-party-verifiable record of that date; the entry now describes it as an internal milestone. See `PRIORITY.md`.

---

## Unreleased（中文）

- **参照哈希**：`reference/reference_hashes.json` 发布全部 154 个比对出口（2 个已发布样本 × 77 个出口）参照输出的 sha256，并记录参照身份（运行时 DLL、`WEIGHTS_HT`、采集 GPU 与架构）。只含哈希，不含任何参照数据。`tools/verify_reference_hashes.py` 据此核对一次运行，输出 `N/154 MATCH`。
- **输入生成器**：`tools/make_inputs.py` 以程序化方式生成两个已发布样本的彩色输入，并与生成参照结果时所用输入的 sha256 核对。
- **文档**：新增 `docs/SCOPE.md`、`docs/NUMERICAL_EXACTNESS.md`、`docs/B70_S157_STATUS.md`、`docs/REPRODUCIBILITY.md` 与 `PRIORITY.md`；删除 README 与 `docs/SAMPLE_NAMES.md` 中指向本仓库之外文件的引用；README 首段改为明确写出覆盖范围。
- **引用元数据**：`CITATION.cff` 补全作者、仓库地址、代码许可证（AGPL-3.0-or-later）与首次公开发布日期（2026-09-17）。
- **更正**：下方 2026-09-14 条目原写作"首次公开结果声明"。该日期没有任何公开、可由第三方核验的记录，现改为如实描述为内部里程碑。见 `PRIORITY.md`。

---

## 2026-09-17

- **Runnable preview package**: Initial runnable preview covering nodes B00 through B69 of the 73-node pipeline.
- **Verification scope**: Byte-identical output against reference results across all 77 compared exits for the two published samples (`c1-C-img-full`, `d23-n2-nat320x240`).
- **Tail nodes excluded**: Nodes B70 and S157 are not included; the package does not currently produce the pipeline's final image.
- **Licensing**: Source code licensed under AGPL-3.0-or-later; documentation and data files licensed under CC-BY-4.0; `third_party/mlxdlss/` retains Apache-2.0.

---

## 2026-09-17（中文）

- **可运行预览包**：首次发布可运行预览包，覆盖 73 节点推理链中的节点 B00 至 B69。
- **验证范围**：两个发布样本（`c1-C-img-full`、`d23-n2-nat320x240`）在全部 77 个比对出口与参考结果逐字节相同。
- **未包含尾部节点**：未包含节点 B70 与 S157；本软件包当前不产出最终图像。
- **许可证**：源代码采用 AGPL-3.0-or-later 许可；文档与数据文件采用 CC-BY-4.0 许可；`third_party/mlxdlss/` 保留 Apache-2.0 许可。

---

## 2026-09-14

- Internal results milestone (not publicly released). First public release: 2026-09-17.

---

## 2026-09-14（中文）

- 内部结果里程碑（未公开发布）。首次公开发布：2026-09-17。

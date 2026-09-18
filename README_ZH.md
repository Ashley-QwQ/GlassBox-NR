# GlassBox-NR（可运行预览包）

[English](README.md)

**神经渲染降噪推理链的逐字节精确 CPU 重建预览：B00–B69（73 个节点中的 71 个），2 个已发布样本 × 77 个比对出口，与经审计的参考捕获相比零字节差异。B70 / S157 以及最终图像输出不在本公开预览范围内。**

这是对推理执行路径的独立重建 CPU 实现。推理内核与执行语义均为独立重建；权重提取与解包使用随包附带的 MLX-DLSS 工具，保留其原始 Apache-2.0 许可证。未使用厂商源码、头文件或 SDK。

## 范围与要求

本软件包需要：

- 您自行合法获取的专有运行时 DLL 副本。本软件包绝不附带、下载或捆绑该文件。
- Python 3.13 及 `requirements.txt` 中所列的依赖项（numpy、torch、safetensors；仅 CPU 版的 torch 即可）。

全流程在 CPU 上运行。无 GPU 依赖，无网络访问，无遥测。

**可运行范围**：73 节点链中的节点 B00 至 B69，涵盖两个已发布样本（`c1-C-img-full`、`d23-n2-nat320x240`；公开名称列于 [`docs/SAMPLE_NAMES.md`](docs/SAMPLE_NAMES.md)）。在该范围内，两个样本的全部 77 个比对出口均与参考输出逐字节相同。全部 154 个出口的参考 sha256 已发布于 [`reference/reference_hashes.json`](reference/reference_hashes.json)，您可以自行核对（详见下文"核对方法"）。

**尚不可运行**：节点 B70 与 S157（最终输出尾段）。**本软件包当前不产出该推理管道的最终图像。** 所有运行均在 B69 停止。原因与后续计划见 [`docs/B70_S157_STATUS.md`](docs/B70_S157_STATUS.md)。

这是一份预览版，并非成品或"官方"发布。此处的说明并不代表所有代码分支都经过了相同深度的核验，亦不作任何"100%"的断言。上述范围即为已核验的全部内容；覆盖与不覆盖的内容列于 [`docs/SCOPE.md`](docs/SCOPE.md)，"逐字节精确"在本项目中的确切含义见 [`docs/NUMERICAL_EXACTNESS.md`](docs/NUMERICAL_EXACTNESS.md)。

## 运行方法

```
python -B tools/extract_weights.py /path/to/your/nvngx_dlssnr.dll
python -B tools/make_inputs.py
python -B tools/run_pipeline.py --sample all --run-tag my-run
python -B tools/verify_reference_hashes.py runs/my-run
```

若您的 DLL 权重资源与本软件包内核验证所用的 sha256 不符，`extract_weights.py` 将拒绝继续执行（参见其模块文档字符串）。

`make_inputs.py` 用闭式的程序化公式生成每个已发布样本的彩色输入（不依赖外部素材，不取自任何原生捕获），写入 `anchors/<sample>/inputs/colour.rgba16`；只有当其 sha256 与生成参考结果时所用的输入一致时才会写出。

`run_pipeline.py` 运行执行链并写出 `runs/my-run/<sample>/run_manifest.json`：记录每个节点的输出字节数与 sha256，以及本次运行是否为部分运行。`--run-tag` 为必填参数，且在 `--output-dir` 下不得已存在（若需在同一标签下继续未完成的运行，请传入 `--resume`）。本软件包绝不会静默覆盖既往运行的证据记录。

## 核对方法

1. `make_inputs.py`（见上）会核对生成的彩色输入与生成参考结果时所用的输入逐字节相同。加 `--check` 可只核对、不写文件。
2. `verify_reference_hashes.py` 将运行清单中的每个出口与 `reference/reference_hashes.json` 比对，输出 `N/154 MATCH`。除非全部 154 个出口都存在且一致，否则以非零状态退出。
3. 可选：运行 `run_pipeline.py` 时加入 `--audit-opens`，检查 `runs/<tag>/<sample>/open_audit.jsonl`，以确认运行过程仅访问了本软件包自身的目录树。

`reference/reference_hashes.json` 只含哈希，不含任何参考张量数据。其头部记录了参考身份：运行时 DLL 版本与 sha256、`WEIGHTS_HT` sha256、采集所用 GPU 与架构。注意：154 个比对输出不等于 154 个图节点，部分节点有不止一个比对出口（每个样本 71 个节点 → 77 个出口，× 2 个样本 = 154）。

标准环境与完整复现步骤见 [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md)。

## 代码承诺与派生文件

包含 997 个研究代码库文件的完整密码学承诺（`freeze_code_sha256 = 83d1a5c2...`）已发布于 `commitment/freeze-code-manifest.json`。
关于详细说明以及对照封存源文件对发布文件进行的逐文件审计，请参见 [`docs/COMMITMENT_AND_DERIVED_FILES.md`](docs/COMMITMENT_AND_DERIVED_FILES.md) 与 [`docs/DERIVED_FILES_MAP_PUBLIC.json`](docs/DERIVED_FILES_MAP_PUBLIC.json)。

## 优先权与时间戳

本项目关于日期能证明什么、不能证明什么，以及依赖哪些第三方时间戳，见 [`PRIORITY.md`](PRIORITY.md)。

## 许可证

- 本项目编写的源代码：**AGPL-3.0-or-later**（参见 `LICENSE`）。
- 本项目编写或派生的文档与数据文件：**CC-BY-4.0**（参见 `LICENSE-docs`）。
- `third_party/mlxdlss/`：MLX-DLSS，保留其自身的 **Apache-2.0** 许可证（参见 `third_party/mlxdlss/LICENSE`、`NOTICE`、`MODIFICATIONS.md`）。

完整分类说明参见 `NOTICE`。

## 未发布内容及原因

有关具体包含与未包含哪些内容及其原因，请参见 `NOTICE` 及各目录下的 `PROVENANCE.md`（权重、原生捕获数据与原始反汇编文本绝不发布；若干最初将计算所需数据与反汇编派生诊断内容混合的数据文件，在此作为已移除此类内容的派生副本发布——每个此类文件的 `PROVENANCE.md` 条目均列出了所应用的规则及生成的 sha256，因此无需本分发包之外的任何内容即可核验确切的转换过程）。

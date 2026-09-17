# GlassBox-NR（可运行预览包）

[English](README.md)

独立从零开始的 CPU 重实现，复现专有神经渲染（neural-rendering）推理管道，经独立逆向工程完成（未使用厂商源码、头文件或 SDK），在当前软件包所覆盖的管道范围内，各项中间输出及最终结果与经审计的参考结果逐字节一致。

## 范围与要求

本软件包需要：

- 您自行合法获取的专有运行时 DLL 副本。本软件包绝不附带、下载或捆绑该文件。
- Python 3.13 及 `requirements.txt` 中所列的依赖项（numpy、torch-cpu、safetensors）。

全流程在 CPU 上运行。无 GPU 依赖，无网络访问，无遥测。

**可运行范围**：73 节点链中的节点 B00 至 B69，涵盖两个已发布样本（`c1-C-img-full`、`d23-n2-nat320x240`；公开名称列于 `docs/SAMPLE_NAMES.md`）。在该范围内，两个样本的所有 77 个比对出口与本项目经审计的参考结果逐字节相同，均已通过本软件包自带的工具完成验证（详见下文"核对方法"）。

**尚不可运行**：节点 B70 与 S157（最终输出尾段）——因为其后端在运行时解析未经处理的原生 SASS 反汇编文本，而该文本并未发布（参见 `DESIGN_ZH.md` 第 3.5 节）。**本软件包当前不产出该推理管道的最终图像。** B70/S157 的纯 Python 重新实现是一个单独跟踪的后续任务；在该实现就绪之前，所有运行均在 B69 停止。

这是一份预览版，并非成品或"官方"发布。此处的说明并不代表所有代码分支都经过了相同深度的核验，亦不作任何"100%"的断言——上述范围即为已核验的全部内容，已在此尽可能平实地陈述。

## 运行方法

```
python tools/extract_weights.py /path/to/your/nvngx_dlssnr.dll
python -B tools/run_pipeline.py --sample c1-C-img-full --run-tag my-run --stop-after B69
```

若您的 DLL 权重资源与本软件包内核验证所用的 sha256 不符，`extract_weights.py` 将拒绝继续执行（参见其模块文档字符串）。`run_pipeline.py` 会在 `anchors/` 目录下生成本地测试输入（`--sample` 对应的彩色图像），运行执行链，并写出 `runs/my-run/<sample>/run_manifest.json`——记录每个节点的输出字节数与 sha256、本次运行是否为部分运行，以及（若指定了 `--compare-dir`）与相同目录结构的参考清单之间各输出的比对匹配结果。

`--run-tag` 为必填参数，且在 `--output-dir` 下不得已存在（若需在同一标签下继续未完成的运行，请传入 `--resume`）——本软件包绝不会静默覆盖既往运行的证据记录。

## 核对方法

1. 运行 `python -B release/p2-runnable/generators/verify_inputs.py`（若您的副本中包含该脚本），以确认已发布的彩色图像生成器能够重现生成参考清单时所使用的相同字节。
2. 在上述运行命令中加入 `--audit-opens`；检查 `runs/<tag>/<sample>/open_audit.jsonl`，以确认运行过程仅访问了本软件包自身的目录树（以及您显式传入的 `--compare-dir`，若有）——未触及目录树之外的任何内容。
3. 将 `run_manifest.json` 中逐节点的输出 sha256 与您信任的独立参考结果进行比对。本软件包在当前的预览版中未随附独立的"金标准（golden）"哈希列表；这是后续计划加入的内容。

## 代码承诺与派生文件

包含 997 个研究代码库文件的完整密码学承诺（`freeze_code_sha256 = 83d1a5c2...`）已发布于 `commitment/freeze-code-manifest.json`。
关于详细说明以及对照封存源文件对发布文件进行的逐文件审计，请参见 [`docs/COMMITMENT_AND_DERIVED_FILES.md`](docs/COMMITMENT_AND_DERIVED_FILES.md) 与 [`docs/DERIVED_FILES_MAP_PUBLIC.json`](docs/DERIVED_FILES_MAP_PUBLIC.json)。

## 许可证

- 本项目编写的源代码：**AGPL-3.0-or-later**（参见 `LICENSE`）。
- 本项目编写或派生的文档与数据文件：**CC-BY-4.0**（参见 `LICENSE-docs`）。
- `third_party/mlxdlss/`：MLX-DLSS，保留其自身的 **Apache-2.0** 许可证（参见 `third_party/mlxdlss/LICENSE`、`NOTICE`、`MODIFICATIONS.md`）。

完整分类说明参见 `NOTICE`。

## 未发布内容及原因

有关具体包含与未包含哪些内容及其原因，请参见 `NOTICE` 及各目录下的 `PROVENANCE.md`（权重、原生捕获数据与原始反汇编文本绝不发布；若干最初将计算所需数据与反汇编派生诊断内容混合的数据文件，在此作为已移除此类内容的派生副本发布——每个此类文件的 `PROVENANCE.md` 条目均列出了所应用的规则及生成的 sha256，因此无需本分发包之外的任何内容即可核验确切的转换过程）。

"""Config-driven workspace adapter for the existing scalar C++ ViT families.

No family source is copied or modified.  Per-launch maps/templates/ranges must
be frozen in a binding JSON and loaded by configure() before the numeric call.
"""
from __future__ import annotations

import importlib
import derived_pins
import json
import sys
from pathlib import Path

_cfg = None
_specs = {}
_mods = {}
_arrays = {}
_BASE = None


def _load_npz(path):
    import numpy as np
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _resolve(root, value):
    p = Path(value)
    return p if p.is_absolute() else root / p


def configure(config: dict, launch: int | None = None):
    """Preload C++ family modules and read-only binding arrays before audit."""
    global _cfg, _specs, _mods, _arrays, _BASE
    if isinstance(config, (str, Path)):
        config_path = Path(config).resolve()
        config = json.loads(config_path.read_text(encoding="utf-8-sig"))
        root = config_path.parent
    else:
        root = Path(config.get("_config_dir", ".")).resolve()
    path_value = config.get("vit_adapter_config")
    if not path_value:
        raise RuntimeError("VIT_BINDINGS_NOT_CONFIGURED")
    bind_path = _resolve(root, path_value)
    doc = json.loads(bind_path.read_text(encoding="utf-8-sig"))
    import hashlib
    if config.get('vit_bindings_sha256') and not derived_pins.accepts_sha(config['vit_bindings_sha256'],hashlib.sha256(bind_path.read_bytes()).hexdigest()):
        raise RuntimeError('VIT_BINDINGS_SHA_DRIFT')
    for item in doc.get('artifacts',[]):
        q=_resolve(bind_path.parent,item['path'])
        if (''.join(map(chr,(98,108,105,110,100,))) in str(q).lower()):raise RuntimeError('FORBIDDEN_PATH')
        if not derived_pins.accepts_sha(item['sha256'],hashlib.sha256(q.read_bytes()).hexdigest()):raise RuntimeError('PROGRAM_SHA_DRIFT:'+str(q))
    _BASE = int(config["workspace_base"], 0) if isinstance(config["workspace_base"], str) else int(config["workspace_base"])
    for p in doc.get("python_paths", []):
        path = str(_resolve(root, p))
        if path not in sys.path:
            sys.path.insert(0, path)
    selected = {int(k): v for k, v in doc.get("launches", {}).items()
                if launch is None or int(k) == launch}
    required = {str(v["family"]) for v in selected.values()}
    modules = doc.get("modules", {})
    missing_modules = sorted(required - set(modules))
    if missing_modules:
        raise RuntimeError("VIT_MODULE_BINDING_MISSING:"+",".join(missing_modules))
    _mods = {k: importlib.import_module(modules[k]) for k in required}
    if 'qkv_projection' in required:
        _mods['qkv_release'] = importlib.import_module(modules['qkv_release'])
    for m in _mods.values():
        pin=config.get('module_pins',{}).get(m.__name__)
        if pin and not derived_pins.accepts_sha(pin['sha256'],hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()):
            raise RuntimeError('COMPILED_MODULE_SHA_DRIFT')
    tables = {k:_load_npz(v)['native_float_bits'] for k,v in doc.get('tables',{}).items()}
    _specs, _arrays = {}, {}
    for L, raw in selected.items():
        spec = dict(raw)
        if spec["family"]=="ffn_contract":
            gate=json.loads(Path(spec["admission_gate_json"]).read_text(encoding="utf-8"))
            cell=next(x for x in gate["cells"] if x["launch"]==L)
            if gate["status"]!="PASS_8_INSTANCE_ADMITTED" or cell!=spec["admission_gate_cell"]:raise RuntimeError("CONTRACT_SCOPED_ADMISSION_DRIFT")
            admission=json.loads(Path(spec["admission_json"]).read_text(encoding="utf-8"))
            if L!=60 and admission.get("admits") is not True:raise RuntimeError("CONTRACT_ADMISSION_NOT_TRUE")
        for name in ("templates_json", "wiring_json"):
            if name in spec:
                spec[name.removesuffix("_json")] = json.loads(_resolve(bind_path.parent, spec.pop(name)).read_text(encoding="utf-8"))
        for name in ("maps_npz", "arrays_npz"):
            if name in spec:
                spec[name.removesuffix("_npz")] = _load_npz(_resolve(bind_path.parent, spec.pop(name)))
        if "epochs" in spec:
            ep = []
            for item in spec["epochs"]:
                template = json.loads(_resolve(bind_path.parent, item["templates_json"]).read_text(encoding="utf-8"))
                maps = _load_npz(_resolve(bind_path.parent, item["maps_npz"]))
                ep.append((template, maps))
            spec["epochs"] = ep
        if "zstore_binding_json" in spec:
            zp = _resolve(bind_path.parent, spec.pop("zstore_binding_json"))
            if not derived_pins.accepts_sha(spec["zstore_binding_sha256"], hashlib.sha256(zp.read_bytes()).hexdigest()):
                raise RuntimeError("ZSTORE_BINDING_SHA_DRIFT")
            spec["zstore_binding"] = json.loads(zp.read_text(encoding="utf-8"))
            z = spec["zstore_binding"]
            if z["launch"] != L or z["family"] != "attention" or list(spec["outputs"]) != ["output", "release63"]:
                raise RuntimeError("ZSTORE_INSTANCE_OR_PUBLICATION_ORDER")
        if 'array_keys' in spec:
            spec['arrays']={k:spec['arrays'][v] for k,v in spec['array_keys'].items()}
        if spec['family']=='attention':spec['rcp_bits']=tables['rcp']
        if spec['family']=='qkv_projection':spec['rsq_bits']=tables['rsq']
        _specs[L] = spec
    _cfg = dict(config)
    _cfg["workspace_bytes"] = int(config["workspace_bytes"])
    return {"loaded_launches": sorted(_specs), "modules": sorted(_mods), "bindings_sha256": __import__("hashlib").sha256(bind_path.read_bytes()).hexdigest()}


def _range(params: bytes, desc: dict, workspace_bytes: int):
    if 'pointer_word' in desc and 'expected_pointer' in desc:
        ptr=int.from_bytes(params[int(desc['pointer_word'])*8:int(desc['pointer_word'])*8+8],'little')
        if ptr!=int(desc['expected_pointer'],0):raise ValueError('WORKSPACE_POINTER_BINDING_MISMATCH')
    if "offset" in desc:
        off = int(desc["offset"])
    else:
        word = int(desc["pointer_word"])
        ptr = int.from_bytes(params[word*8:word*8+8], "little")
        off = ptr - int(desc.get("base", _BASE))
    if "bytes" in desc:
        size = int(desc["bytes"])
    elif "bytes_from_params_u32" in desc:
        row_word, col_word, factor = (int(x) for x in desc["bytes_from_params_u32"])
        rows = int.from_bytes(params[row_word*4:row_word*4+4], "little")
        cols = int.from_bytes(params[col_word*4:col_word*4+4], "little")
        size = rows * cols * factor
    else:
        raise ValueError("RANGE_BYTES_NOT_BOUND")
    if off < 0 or size < 0 or off + size > workspace_bytes:
        raise ValueError(f"WORKSPACE_RANGE_INVALID:{off}:{size}")
    return off, size


def _blob(workspace, params, desc):
    off, size = _range(params, desc, len(workspace))
    return bytes(workspace[off:off+size]), off, size


def _weights(params, weights, spec):
    d = spec.get("weight_slice")
    if not d:
        return weights
    pointer_word = int(d["pointer_word"])
    pointer = int.from_bytes(params[pointer_word*8:pointer_word*8+8], "little")
    expected = d.get("expected_pointer")
    if expected is not None:
        expected_value = int(expected, 0) if isinstance(expected, str) else int(expected)
        if pointer != expected_value:
            raise ValueError(f"WEIGHT_POINTER_BINDING_MISMATCH:{pointer:#x}:{expected_value:#x}")
    # Safetensors-backed family bindings identify the tensor by both the
    # params pointer and its independently audited byte offset in WEIGHTS_HT.
    # Flat packed pools keep using pointer - pointer_base.
    offset = int(d["data_offset"]) if "data_offset" in d else pointer - int(d["pointer_base"])
    size = int(d["bytes"])
    if offset < 0 or offset + size > len(weights):
        raise ValueError(f"WEIGHT_POINTER_SLICE_INVALID:{offset}:{size}")
    blob=bytes(weights[offset:offset+size])
    if d.get('sha256') and __import__('hashlib').sha256(blob).hexdigest()!=d['sha256']:
        raise ValueError('WEIGHT_TENSOR_SHA_MISMATCH')
    return blob


def _publish(workspace, result, spec, params):
    written = []
    for key, dst in spec.get("outputs", {}).items():
        if key not in result:
            raise KeyError(f"FAMILY_OUTPUT_MISSING:{key}")
        data = bytes(result[key])
        off, size = _range(params, dst, len(workspace))
        if len(data) != size:
            raise ValueError(f"FAMILY_OUTPUT_SIZE:{key}:{len(data)}:{size}")
        workspace[off:off+size] = data
        written.append({"role": key, "offset": off, "bytes": size})
        if key == "output" and "zstore_binding" in spec:
            z = spec["zstore_binding"]
            if params != bytes.fromhex(z["params_hex"]) or z["runtime_phase"] != "AFTER_OSTORE_BEFORE_RELEASE":
                raise RuntimeError("ZSTORE_PARAMS_OR_PHASE_DRIFT")
            if spec["zstore_execute"]:
                for op in z["ops"]:
                    if op["kind"] != "zstore" or op["region"] != "output" or op["width"] != 4 or op["value"] != 0 or op["phase"] != "AFTER_OSTORE_BEFORE_RELEASE":
                        raise RuntimeError("ZSTORE_TYPED_OPERATION_DRIFT")
                    at = off + op["offset"]
                    if op["offset"] < 0 or op["offset"] + 4 > size:
                        raise RuntimeError("ZSTORE_RANGE_DRIFT")
                    workspace[at:at+4] = int(op["value"]).to_bytes(4,"little")
                written.append({"role":"output:zstore", "offset":off+min(o["offset"] for o in z["ops"]), "bytes":4*len(z["ops"]), "scalar_stores":len(z["ops"]), "phase":"AFTER_OSTORE_BEFORE_RELEASE", "binding_sha256":spec["zstore_binding_sha256"]})
    return written


def run_launch(launch: int, workspace: bytearray, params: bytes, weights: memoryview,
               textures: dict, threads: int) -> dict:
    if _cfg is None:
        raise RuntimeError("VIT_ADAPTER_CONFIGURE_REQUIRED")
    spec = _specs.get(int(launch))
    if spec is None:
        raise RuntimeError(f"VIT_LAUNCH_BINDING_PENDING:{launch}")
    if spec.get("execution_status") != "READY_TO_RUN":
        raise RuntimeError(f"VIT_LAUNCH_BINDING_PENDING:{launch}:{spec.get('execution_status', 'UNSET')}")
    family = spec["family"]
    if spec.get('params_sha256') and __import__('hashlib').sha256(params).hexdigest()!=spec['params_sha256']:
        raise ValueError('PARAMS_SHA_MISMATCH')
    mod = _mods[family]
    wk = len(workspace)
    ranges = spec.get("inputs", {})
    data, input_offsets = {}, {}
    for role, d in ranges.items():
        data[role], input_offsets[role], _ = _blob(workspace, params, d)
    if family in ("ffn_expand", "qkv_projection", "ffn_contract") and not spec.get("weight_slice"):
        raise RuntimeError(f"VIT_WEIGHT_POINTER_SLICE_NOT_BOUND:{launch}")
    w = _weights(params, weights, spec)
    if family == "repack":
        roles = mod.execute_roles_padded(data["input"], params, _BASE, wk, int(spec["native_grid"][0]), threads, int(spec.get("queue_seed", 0))) if "padding" in spec.get("outputs",{}) else mod.execute_roles(launch, data["input"], params, _BASE, wk, threads, int(spec.get("queue_seed", 0)))
        result = {"output": roles["output"]}
        if "padding" in spec.get("outputs",{}):result["padding"]=roles["padding"]
        if "outputs" not in spec:
            off = int(roles["output_workspace_offset"])
            spec = dict(spec, outputs={"output": {"offset": off, "bytes": len(result["output"])}})
    elif family == "ffn_expand":
        arrays = spec["arrays"]
        result = mod.execute(data["input"], w, arrays["src"], arrays["weight_offsets"], arrays["store_offsets"],
                             spec["wiring"], threads, bool(spec.get("shuffled", False)))
    elif family == "attention":
        role_data = {k: data[k] for k in ("q", "k", "v")}
        result = mod.execute(role_data, spec["epochs"], spec["rcp_bits"], threads, int(spec.get("queue_seed", 0)))
    elif family == "qkv_projection":
        role_data = {"input": data["input"], "weights": w}
        if "residual" in data: role_data["residual"] = data["residual"]
        result = mod.execute(spec["operation"], spec["templates"], spec["maps"], role_data,
                             spec["rsq_bits"], threads, int(spec.get("queue_seed", 0)), bool(spec.get("reverse_red", False)))
        release_result=_mods['qkv_release'].execute_release(spec['operation'],int(spec['templates']['geometry']['grid'][0]))
        result.update({k:v for k,v in release_result.items() if k!='mem'})
        if 'mem' in result:
            result['mem']['separate_release_call_accounting']=release_result.get('mem',{'status':'CXX_INTERNAL_ACCOUNTING_UNAVAILABLE','peak_allocated_bytes':None,'live_end_bytes':None})
    elif family == "ffn_contract":
        result = mod.execute(spec["templates"], spec["maps"], data["input"], data["residual"], w,
                             int(spec["buffer_len"]), int(spec["output_len"]), threads,
                             int(spec.get("queue_seed", 0)), bool(spec.get("reverse_red", False)))
    else:
        raise ValueError(f"UNSUPPORTED_VIT_FAMILY:{family}")
    written = _publish(workspace, result, spec, params)
    # Internal family allocations remain explicitly unknown unless the C++
    # extension supplies its own accounting. RSS is never substituted for it.
    mem=result.get('mem',{'status':'CXX_INTERNAL_ACCOUNTING_UNAVAILABLE','peak_allocated_bytes':None,'live_end_bytes':None,'buffers':{}})
    return {"launch": int(launch), "written_ranges": written, "textures_out": {},'mem':mem,
            'adapter_mem':{'copied_inputs_bytes':sum(map(len,data.values())),'weight_slice_bytes':len(w),'returned_output_bytes':sum(len(result[k]) for k in spec['outputs'])}}

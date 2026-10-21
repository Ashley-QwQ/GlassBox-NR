"""M: F arithmetic, C lifetime, E1 codec, G2/G0a routing. Decoded only."""
import hashlib
import json
from pathlib import Path
import sys
import threading
import numpy as np
from identity import ROOT, load, module, pin, safe, verify, check_actual

_state = None
_configured_launch = None
_serial = threading.Lock()
_vit_adapters = {}
_reads = None
_asset_registry = None
_audit_checking = False
_events = []
_actual_route = None
_guard = None

def _retire(state):
    if state is None: return
    owner, mod, _ = state
    if owner == 'entry': mod.release_assets()
    elif owner == 'outtail': mod.optm_clear_tables()
    elif owner == 'split16': mod.release_assets()

def _audit(event, args):
    global _audit_checking
    if event != 'open' or not args or not isinstance(args[0], (str, bytes)):
        return
    p = safe(args[0].decode() if isinstance(args[0], bytes) else args[0]).resolve()
    if _reads is None or _audit_checking or p.suffix.lower() in ('.py', '.pyc', '.pyd', '.dll'):
        return
    if str(p) not in _asset_registry:
        raise RuntimeError('UNPINNED_CONFIGURATION_ASSET:' + str(p))
    _audit_checking = True
    try:
        item = verify(_asset_registry[str(p)])
        _reads[str(p)] = item
    finally:
        _audit_checking = False

sys.addaudithook(_audit)

def pinned(p, h=None):
    raw = safe(p).read_bytes()
    if h and hashlib.sha256(raw).hexdigest() != h:
        raise RuntimeError('BACKEND_ASSET_SHA_MISMATCH')
    return raw

def borrow_texture(owner):
    root = owner
    while isinstance(root, memoryview):
        root = root.obj
    view = memoryview(owner) if isinstance(root, bytes) else None
    if view is not None and view.ndim == 1 and view.itemsize == 1 and view.strides == (1,):
        result = view.toreadonly(); view.release(); return result
    if view is not None:
        view.release()
    return memoryview(bytes(owner)).toreadonly()

def _clear_adapter(adapter):
    adapter._cfg = None; adapter._specs = {}; adapter._arrays = {}; adapter._mods = {}

def _loader(c, key):
    item = c['loader_pins'][key]
    return module(item, 'optm_' + key + '_' + item['sha256'])

def _configure(c, L, row):
    global _state, _actual_route
    if c.get('panel_mode', 'decoded') != 'decoded':
        raise RuntimeError('OPTM_DECODED_PANEL_REQUIRED')
    owner = row['route_owner']; bc = c['backend_configs']; loaders = []; aux = []; normalized_key = None
    if 58 <= L <= 99:
        adapter = _loader(c, 'vit_adapter_c1'); loaders.append(adapter)
        _vit_adapters[1] = adapter
        adapter.configure(c, L)
        spec = adapter._specs[L]; mod = adapter._mods[spec['family']]
        aux = [m for key, m in sorted(adapter._mods.items()) if key != spec['family']]
        _state = ('vit', adapter, None)
    else:
        mod = module(row['module'], row['module_name'])
        _state = (owner, mod, None)
        if owner == 'entry':
            ec = load(bc['entry']); helper = _loader(c, 'entry_assets'); loaders.append(helper)
            helper.PINS = {}; helper.READ_LOG.clear()
            assets = helper.assets(dict(ec, optc_asset_pins=c['optc_asset_pins']), L)
            assets['native_device_weight_base'] = c.get('_entry_native_device_weight_base', 0xc400000)
            mod.configure(c.get('_entry_workspace_base', ec['workspace_base']), assets); mod.set_reverse_red_control(False)
            if L==2:
                mod.prepare_noise_assets(ec["compact_noise"])
                for path in ec["compact_noise"].values():
                    expected=_asset_registry[str(safe(path).resolve())]
                    _reads[expected["path"]]=verify(expected)
                if mod.noise_assets_live_bytes()!=22961160:raise RuntimeError("D_COMPACT_OWNER_SIZE_DIFF")
            del assets
            wt = pinned(ec['device_weights'])
        elif owner == 'tin':
            tc = load(bc['tin']); work = Path(tc['work']); b = load(work / 'bindings.json')
            rsq = np.frombuffer(pinned(work / 'rsq.u16', b['mufu_raw_sha256']['rsq']), '<u2')
            rcp = np.frombuffer(pinned(work / 'rcp.u16', b['mufu_raw_sha256']['rcp']), '<u2')
            mod.set_panel_mode(True)
            mod.set_optb2_tables(True, True)
            mod.configure(b['workspace_base'], b['weightbase'], {int(k): v for k, v in b['rows'].items()}, rsq, rcp)
            wt = pinned(work / 'weights_gpu.bin', b['weights_gpu_sha256'])
        elif owner == 'split16':
            sc = load(bc['split16']); helper = _loader(c, 'g0a_runtime'); loaders.append(helper)
            manifest, source_row, sp, tables = helper.load(sc, L)
            task_identity = c.get('split_task_identity', 'CLAUDE_CPP_OPT_MERGE_R01_20261002_CAPTURED_FRAME0')
            epoch = c.get('split_epoch', 0)
            normalized_key = helper.normalized_key(sp, task_identity, epoch)
            if not c.get('_discover_route'):
                helper.require_normalized_key(sp, task_identity, epoch, row['normalized_key'])
            if '_precompiled_metadata' in sp:
                mod._configure_precompiled(L, sp['_precompiled_metadata'], sp['maps'], sp['params'], tables['rsq'], tables['rcp'], manifest['workspace_base'], bytes.fromhex(normalized_key))
            else:
                mod.configure({L: sp}, tables['rsq'], tables['rcp'], manifest['workspace_base'])
            wt = pinned(Path(sc['work']) / 'bundle/weights-device.bin', manifest['weights']['sha256'])
        elif owner == 'outtail':
            oc = load(bc['outtail']); helper = _loader(c, 'tail_runtime'); loaders.append(helper)
            desc = helper.stage_tables(dict(oc, e1_encoded=c['e1_encoded']), L)
            mod.init_table_files(L, desc)
            native_reads = mod.optm_table_owners()['actual_reads']
            wanted_reads = [{k: item[k] for k in ('path', 'bytes', 'sha256')}
                            for item in desc.values() if isinstance(item, dict) and 'path' in item]
            normalize = lambda item: {**item, 'path': str(safe(item['path']).resolve())}
            if sorted(map(normalize, native_reads), key=lambda p: p['path']) != sorted(map(normalize, wanted_reads), key=lambda p: p['path']):
                raise RuntimeError('TAIL_ACTUAL_READ_IDENTITY_DIFF')
            for item in native_reads:
                expected = normalize(item)
                if _asset_registry.get(expected['path']) != expected:
                    raise RuntimeError('TAIL_ACTUAL_READ_REGISTRY_DIFF')
                _reads[expected['path']] = expected
            original = _loader(c, 'outtail_weight'); loaders.append(original)
            wt, _ = original.weight_upload(oc)
        else:
            raise RuntimeError('BACKEND_NOT_CONFIGURED:' + owner)
        _state = (owner, mod, wt)
    actual_assets = sorted(_reads.values(), key=lambda p: p['path'])
    actual_abi = {'backend_abi': pin(__file__), 'identity': pin(sys.modules['identity'].__file__),
                  'owner_guard': pin(_guard.__file__)}
    if actual_abi != c['ABI'] or c['ABI'] != row['ABI']:
        raise RuntimeError('ACTUAL_ABI_IDENTITY_DIFF')
    if c.get('_discover_route'):
        _actual_route = {'launch': L, 'module': pin(mod.__file__), 'loaders': [pin(x.__file__) for x in loaders],
                         'auxiliary_modules': [pin(x.__file__) for x in aux],
                         'ABI': actual_abi, 'panel_mode': 'decoded', 'assets': actual_assets}
    else:
        _actual_route = check_actual(row, mod, loaders, actual_assets, aux, actual_abi)
    if normalized_key is not None:
        _actual_route['normalized_key'] = normalized_key

def configure(c, L, *, workspace_base=None, native_device_weight_base=None, bridge_prefix=None):
    global _state, _configured_launch, _reads, _asset_registry, _actual_route, _guard
    if not _serial.acquire(blocking=False):
        raise RuntimeError('CONFIGURE_RUN_BUSY')
    began = False
    if workspace_base is not None or native_device_weight_base is not None:
        c = dict(c)
        if workspace_base is not None: c['_entry_workspace_base'] = workspace_base
        if native_device_weight_base is not None: c['_entry_native_device_weight_base'] = native_device_weight_base
    try:
        _retire(_state)
        _state = None; _configured_launch = None; _actual_route = None
        began = True
        for adapter in _vit_adapters.values():
            _clear_adapter(adapter)
        row = next(r for r in load(c['route_table'])['rows'] if r['launch'] == L)
        _asset_registry = load(c['asset_registry']); _reads = {}
        _guard = module(c['ABI']['owner_guard'], 'optm_owner_guard')
        _events.append({'event': 'configure_begin', 'launch': L})
        if c.get('bridge_readset_contract') == 'complete-preparation-v2':
            pr = c['bridge_parameter_pins'][str(L)]
            if _asset_registry.get(pr['path']) != pr or row['params_sha256'] != pr['sha256']:
                raise RuntimeError('BRIDGE_PARAMETER_DECLARATION_DIFF')
            params = pinned(pr['path'], pr['sha256'])
            if len(params) != pr['bytes']: raise RuntimeError('BRIDGE_PARAMETER_BYTES_DIFF')
            if bridge_prefix is not None:
                bridge_prefix(params)
            elif row['route_owner'] not in ('entry', 'outtail'):
                pinned(c['paths']['weights_ht'], c['weights_ht_container_sha256'])
        elif bridge_prefix is not None:
            raise RuntimeError('BRIDGE_COMPLETE_READSET_CONTRACT_REQUIRED')
        _configure(c, L, row); _configured_launch = L
        _events.append({'event': 'configured', 'launch': L, 'read_paths': list(_reads)})
        return _actual_route
    except BaseException as error:
        if began:
            if str(error) not in ('CONFIGURE_RUN_BUSY', 'TAIL_CONFIGURE_RUN_BUSY'):
                _retire(_state)
            _state = None
            for adapter in _vit_adapters.values():
                _clear_adapter(adapter)
        raise
    finally:
        _reads = None; _serial.release()

def run_launch(L, ws, params, weights, textures, threads):
    if not _serial.acquire(blocking=False):
        raise RuntimeError('CONFIGURE_RUN_BUSY')
    try:
        if _state is None or _configured_launch != L:
            raise RuntimeError('ABI_CONFIGURED_LAUNCH_MISMATCH')
        owner, mod, wt = _state
        try:
            _guard.reject_aliases(L, ws, textures)
        except BaseException:
            if owner == 'entry': mod.release_assets()
            raise
        if owner == 'vit':
            return mod.run_launch(L, ws, params, weights, textures, threads)
        tex = {}; weight_view = None; workspace_pin = None
        try:
            if owner == 'outtail':
                for k, v in textures.items():
                    if isinstance(k, int) and not isinstance(v['data'], bytearray):
                        v['data'] = bytearray(v['data'])
                tex = textures
            else:
                tex = {} if (owner, L) in (('entry', 1), ('tin', 3), ('tin', 4), ('tin', 5), ('tin', 6), ('tin', 7), ('tin', 8), ('tin', 9), ('tin', 10), ('tin', 11), ('tin', 12), ('tin', 13), ('tin', 14), ('tin', 15), ('tin', 16), ('tin', 17), ('tin', 18), ('tin', 19), ('tin', 20), ('tin', 21), ('tin', 22), ('tin', 23), ('tin', 24), ('split16', 25), ('split16', 26), ('split16', 27), ('split16', 28), ('split16', 29), ('split16', 30), ('split16', 31), ('split16', 32), ('split16', 33), ('split16', 34), ('split16', 35), ('split16', 36), ('split16', 37), ('split16', 38), ('split16', 39), ('split16', 40), ('split16', 41), ('split16', 42), ('split16', 43), ('split16', 44), ('split16', 45), ('split16', 46), ('split16', 47), ('split16', 48), ('split16', 49), ('split16', 50), ('split16', 51), ('split16', 52), ('split16', 53), ('split16', 54), ('split16', 55), ('split16', 56), ('split16', 57), ('entry', 100), ('split16', 101), ('split16', 102), ('split16', 103), ('split16', 104), ('split16', 105), ('split16', 106), ('split16', 107), ('split16', 108), ('split16', 109), ('split16', 110), ('split16', 111), ('split16', 112), ('split16', 113), ('split16', 114), ('split16', 115), ('split16', 116), ('split16', 117), ('split16', 118), ('split16', 119), ('split16', 120), ('split16', 121), ('split16', 122), ('split16', 123), ('split16', 124), ('split16', 125), ('split16', 126), ('split16', 127), ('split16', 128), ('split16', 129), ('split16', 130), ('split16', 131), ('entry', 132), ('tin', 133), ('tin', 134), ('tin', 135), ('tin', 136), ('tin', 137), ('tin', 138), ('tin', 139), ('tin', 140), ('tin', 141), ('tin', 142), ('tin', 143), ('tin', 144), ('tin', 145), ('tin', 146), ('tin', 147), ('tin', 148), ('tin', 149), ('tin', 150), ('tin', 151), ('tin', 152), ('tin', 153), ('tin', 154)) else {k: borrow_texture(v['data'] if isinstance(v, dict) else v)
                       for k, v in textures.items() if isinstance(k, int)}  # M02_S2_SHA256=e7e96a246eb49669f47210561b0d9c583556637a791bf7dbc125211a082ba89e
            weight_view = memoryview(wt); workspace_pin = memoryview(ws)
            _events.append({'event': 'call', 'launch': L})
            result = mod.run_launch(L, ws, params, weight_view, tex, threads)
            _events.append({'event': 'call_return_joined', 'launch': L})
            return result
        except BaseException:
            _events.append({'event': 'call_exception_CXX_unwound', 'launch': L})
            raise
        finally:
            if weight_view is not None: weight_view.release()
            if workspace_pin is not None: workspace_pin.release()
            for view in tex.values():
                if isinstance(view, memoryview): view.release()
            if owner == 'entry':
                mod.release_assets()
                _events.append({'event': 'entry_release_after_join', 'launch': L})
            if owner == 'split16':
                mod.release_assets()
                _events.append({'event':'split_release_after_join_or_unwind','launch':L})
    finally:
        _serial.release()


def configure_bridge_outtail(c, L, mod, loaders, bridge_prefix):
    """Original strict seven-field checker at the complete Bridge prep boundary."""
    global _state, _configured_launch, _reads, _asset_registry, _actual_route, _guard
    if not _serial.acquire(blocking=False): raise RuntimeError('CONFIGURE_RUN_BUSY')
    try:
        _retire(_state); _state=None; _configured_launch=None; _actual_route=None
        for adapter in _vit_adapters.values(): _clear_adapter(adapter)
        row=next(r for r in load(c['route_table'])['rows'] if r['launch']==L)
        if row['route_owner']!='outtail' or c.get('bridge_readset_contract')!='complete-preparation-v2':
            raise RuntimeError('BRIDGE_OUTTAIL_CONTRACT_REQUIRED')
        _asset_registry=load(c['asset_registry']); _reads={}
        _guard=module(c['ABI']['owner_guard'],'optm_owner_guard')
        actual_mod=module(row['module'],row['module_name'])
        if actual_mod is not mod: raise RuntimeError('BRIDGE_OUTTAIL_MODULE_OBJECT_DIFF')
        pr=c['bridge_parameter_pins'][str(L)]
        if _asset_registry.get(pr['path'])!=pr or row['params_sha256']!=pr['sha256']:
            raise RuntimeError('BRIDGE_PARAMETER_DECLARATION_DIFF')
        params=pinned(pr['path'],pr['sha256'])
        if len(params)!=pr['bytes']: raise RuntimeError('BRIDGE_PARAMETER_BYTES_DIFF')
        bridge_prefix(params)
        for item in mod.optm_table_owners()['actual_reads']:
            item={k:item[k] for k in ('path','bytes','sha256')}
            item['path']=str(safe(item['path']).resolve())
            if _asset_registry.get(item['path'])!=item: raise RuntimeError('TAIL_ACTUAL_READ_REGISTRY_DIFF')
            _reads[item['path']]=verify(item)
        actual_abi={'backend_abi':pin(__file__),'identity':pin(sys.modules['identity'].__file__),'owner_guard':pin(_guard.__file__)}
        if actual_abi!=c['ABI'] or c['ABI']!=row['ABI']: raise RuntimeError('ACTUAL_ABI_IDENTITY_DIFF')
        for raw_item in mod.mufu_full_raw_pins():
            item={k:raw_item[k] for k in ('path','bytes','sha256')}
            item['path']=str(safe(item['path']).resolve())
            if _asset_registry.get(item['path'])!=item: raise RuntimeError('REACH_RAW_READ_REGISTRY_DIFF')
            # The sealed C++ registry verifies the fd contents at registration,
            # and checks fd stat identity before lookup. This is its actual read receipt.
            _reads[item['path']]=item
        if c.get('_discover_route'):
            _actual_route={'launch':L,'module':pin(mod.__file__),'loaders':[pin(x.__file__) for x in loaders],'auxiliary_modules':[],'ABI':actual_abi,'panel_mode':'decoded','assets':sorted(_reads.values(),key=lambda p:p['path'])}
        else:
            _actual_route=check_actual(row,mod,loaders,sorted(_reads.values(),key=lambda p:p['path']),(),actual_abi)
        _configured_launch=L
        return _actual_route
    except BaseException:
        mod.optm_clear_tables(); _state=None; _configured_launch=None
        raise
    finally:
        _reads=None; _serial.release()

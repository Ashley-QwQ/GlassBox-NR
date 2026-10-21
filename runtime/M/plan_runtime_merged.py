"""IG-a structural metadata; original G0b maps and normalized identity gates."""
from pathlib import Path
import hashlib,importlib.util,sys
import derived_pins
import numpy as np
from plan_format import decode,unpack_spec
from g0b_maps import read_narrow
cache={}
def verify(v):
    p=Path(v['path'])
    if (''.join(map(chr,(98,108,105,110,100,))) in str(p).casefold()):raise RuntimeError('FORBIDDEN_PATH')
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    if not derived_pins.accepts(v['sha256'],v['bytes'],h.hexdigest(),p.stat().st_size):raise RuntimeError('PLAN_ASSET_SHA_DIFF')
    return p
def index(c):
    v=c['_merged_plan_index'];p=verify(v)
    if v['sha256'] not in cache:
        b=p.read_bytes()
        if not b.startswith(b'MRGPLAN1'):raise RuntimeError('MERGED_PLAN_VERSION')
        cache[v['sha256']]=decode(b[8:])
    return cache[v['sha256']]
def base_helper(c):
    v=c['_original_g0b_helper'];p=verify(v);name='merged_original_g0b_helper_'+v['sha256']
    if name not in sys.modules:
        s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m)
    return sys.modules[name]
def load(config,launch):
    obj=index(config);row=obj['rows'][str(launch)];sp=dict(row['spec'])
    for v in row['source_pins']:verify(v)
    v=row['metadata'];metadata=verify(v).read_bytes();unpack_spec(metadata,row['normalized_key'],launch)
    arrays,cert=read_narrow(config['g0b_certificates'][str(launch)],row['original_map_sha256'],base_helper(config).REQUIRED_FIELDS+base_helper(config).OPTIONAL_FIELDS)
    sp['wiring']=obj['templates'][row['template']]['wiring'];sp['contract']=obj['templates'][row['template']]['contract']
    sp['maps']=base_helper(config).selected_maps(arrays,sp['wiring']);sp['_g0b_fields']=cert['fields']
    if 'thread_class' in sp['maps']:sp['maps']['thread_class']=sp['maps']['thread_class'][...,None]
    for array in sp['maps'].values():array.flags.writeable=False
    sp['params']=verify(row['params']).read_bytes();sp['_precompiled_metadata']=metadata
    tables={}
    for name,v in obj['tables'].items():
        raw=verify(v).read_bytes();tables[name]=np.frombuffer(raw,dtype='<f4')
    if base_helper(config).normalized_key(sp,obj['task_identity'],obj['epoch'])!=row['normalized_key']:raise RuntimeError('ORIGINAL_G0B_NORMALIZED_PLAN_KEY_DIFF')
    return obj['manifest'],row['original_row'],sp,tables
def normalized_key(spec,task_identity,epoch):
    # Exact original G0b function; no relaxed precompiled identity shortcut.
    mod=next(m for n,m in sys.modules.items() if n.startswith('merged_original_g0b_helper_'))
    return mod.normalized_key(spec,task_identity,epoch)
def require_normalized_key(spec,task_identity,epoch,expected):
    mod=next(m for n,m in sys.modules.items() if n.startswith('merged_original_g0b_helper_'))
    return mod.require_normalized_key(spec,task_identity,epoch,expected)

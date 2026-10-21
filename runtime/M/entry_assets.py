"""C3: original asset values, current launch only, full source identity retained."""
import contextlib, hashlib, json, os
from pathlib import Path
import numpy as np
from g0b_maps import read_narrow
import derived_pins
PINS={}
READ_LOG=[]
def identity(c):
    global PINS
    pinpath=str(Path(c.get('optm_asset_pins',c.get('optc_asset_pins'))).resolve())
    if pinpath not in PINS:PINS[pinpath]=json.loads(Path(pinpath).read_text(encoding='utf-8'))
    return PINS[pinpath]
@contextlib.contextmanager
def pinned_file(c,p):
    p=Path(p).resolve()
    if any(''.join(map(chr,(98,108,105,110,100,))) in x.lower() for x in p.parts):raise RuntimeError('FORBIDDEN_PATH')
    expected=identity(c)[str(p)];h=hashlib.sha256()
    with p.open('rb') as f:
        before=os.fstat(f.fileno())
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
        if not derived_pins.accepts(expected['sha256'],expected['bytes'],h.hexdigest(),before.st_size):raise RuntimeError('ENTRY_ASSET_SHA_MISMATCH')
        READ_LOG.append({'path':str(p),'sha256':h.hexdigest(),'bytes':before.st_size})
        f.seek(0);yield f
        after=os.fstat(f.fileno())
        if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise RuntimeError('ENTRY_ASSET_CHANGED_DURING_LOAD')
def arrays(c,p):
    with pinned_file(c,p) as f:
        with np.load(f,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def doc(c,p):
    with pinned_file(c,p) as f:return json.load(f)
def assets(c,L):
    if L==132 and "g0b_l132_certificate" in c and c.get("geometry_assets",{}).get("L132"):
        raise RuntimeError("GEO2_ENTRY_L132_CERTIFICATE_MAP_MIX")
    root=Path(c['root']);base=Path(c['assets']);d={}
    if L in (0,1):return d
    for kind,fn in [('rsq','rsq-domain.npz'),('rcp','rcp-domain.npz')]:
        z=arrays(c,root/'mlxdlss/resources'/fn)
        d[kind]=np.ascontiguousarray(z['native_float_bits'][:0x7c00-0x410].view(np.float32))
        d[kind+'_half']=np.ascontiguousarray(z['native_half_bits'])
    if L==2:pass # Authenticated final C++ compact owner.
    elif L==100:
        p=base/'tail-l100-b39-r02-codex/maps';g=c.get('geometry_assets',{}).get('L100',{});d['l100_maps']=arrays(c,g.get('maps',p/'b39-map.npz'));d['l100_json']=doc(c,g.get('metadata',p/'b39-map.json'))
    elif L==132:
        p=base/'tail-l132-b47-r01-codex/maps'
        if 'g0b_l132_certificate' in c:
            original=p/'L132-1080-lane.npz'
            with pinned_file(c,original):pass
            certpath=c['g0b_l132_certificate']
            with pinned_file(c,certpath):pass
            maps,cert=read_narrow(certpath,identity(c)[str(original.resolve())]['sha256'])
            with pinned_file(c,Path(certpath).parent/cert['narrow']['relative']):pass
        else:
            maps=arrays(c,c.get('geometry_assets',{}).get('L132',{}).get('maps',p/'L132-1080-lane.npz'))
            maps={k:np.ascontiguousarray(v,dtype=np.int64) for k,v in maps.items()}
        for key in ['shfl_idx','psel']:
            if key not in maps:maps[key]=np.zeros(maps['wload_off'].shape[:3]+(0,),np.int8)
        d['l132']={'maps':maps,'wiring':doc(c,c.get('geometry_assets',{}).get('L132',{}).get('wiring',p/'L132-1080-lane.wiring.json'))}
    else:raise ValueError('ENTRY_LAUNCH_OUTSIDE_DOMAIN')
    return d

"""Own round IO; never enumerate an unfiltered recursive directory."""
import hashlib, json, os, sys, time
from pathlib import Path
sys.dont_write_bytecode = True

def safe(p):
    p=Path(p)
    if any(''.join(map(chr,(98,108,105,110,100,))) in s.lower() for s in p.parts): raise RuntimeError('forbidden name')
    return p
def sha(data): return hashlib.sha256(data).hexdigest()
def shafile(p):
    h=hashlib.sha256()
    with safe(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()
def write_json(p,obj):
    p=safe(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=str),encoding='utf8')
def config(p):
    c=json.loads(safe(p).read_text(encoding='utf8'))
    for k in ('repo','work','python','dev_zip','texture_min','texture_full','dll','lock','round','build'):
        safe(c[k])
    return c
def input_dir(c,L): return safe(c['work'])/'inputs'/str(L)
def tables(c):
    d=json.loads((safe(c['work'])/'tables/manifest.json').read_text())
    data={}
    for k in ('ex2','rcp','lg2','rsq'):
        b=(safe(c['work'])/'tables'/f'{k}.bin').read_bytes()
        if sha(b)!=d['sources'][k]['extracted_sha256']:raise RuntimeError('actual MUFU read SHA mismatch')
        data[k]=b
    return {**d['runtime'],**data}
def texture_desc(c,L):
    d=json.loads((input_dir(c,L)/'manifest.json').read_text())
    out={'_workspace_base':d['workspace_base']}
    for s,v in d['textures'].items():
        p=input_dir(c,L)/v['file'];b=p.read_bytes()
        if sha(b)!=v['sha256']: raise RuntimeError('input SHA mismatch')
        out[int(s)]={**v,'data':bytearray(b)}
    return out,d
def weight_upload(c):
    """Serialized canonical WEIGHTS_HT -> actual 512-aligned upload layout, in RAM."""
    import importlib.util
    root=safe(c['repo']);p=root/'research-upstream/python/mlxdlss/tools/extract_dlssnr_weights.py'
    spec=importlib.util.spec_from_file_location('outtail_weight_parser',p);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    raw=(safe(c['work'])/'weights.bin').read_bytes()
    if sha(raw)!='836f445d06ecd2e59bb9f17b84b91c143396fd76ccda1c9dc7fe81d5edd548f4':raise RuntimeError('WEIGHTS_HT canonical mismatch')
    resource_sha256=sha(raw);tensors=m.parse_weight_map(raw)
    if any(not t.payload.flags.owndata for t in tensors.values()):raise RuntimeError("CPLUS_TENSOR_OWNER_REQUIRED")
    del raw
    upload=bytearray();rows=[]
    for name in sorted(tensors):
        start=(len(upload)+511)//512*512;upload.extend(bytes(start-len(upload)));tensor=tensors.pop(name);b=tensor.payload.tobytes();upload.extend(b);rows.append({'name':name,'upload_offset':start,'bytes':len(b),'sha256':sha(b)})
        del b,tensor
    return upload,{'source_resource_sha256':resource_sha256,'upload_sha256':sha(upload),'upload_bytes':len(upload),'parser':str(p),'parser_sha256':shafile(p),'tensor_records':rows}
def save_outputs(out,ws,tex,rec):
    out=safe(out);out.mkdir(parents=True,exist_ok=False)
    inp=Path(rec['input_directory']) if 'input_directory' in rec else None
    def publish(n,data):
        source=inp/n if inp else None
        if source and source.exists() and shafile(source)==sha(data):os.link(source,out/n)
        else:(out/n).write_bytes(data)
    publish('workspace.bin',ws)
    rec['workspace_sha256']=sha(ws);rec['texture_sha256']={}
    for k,v in tex.items():
        if isinstance(k,int):
            data=v['data'];publish(f'tex{k}.bin',data);rec['texture_sha256'][str(k)]=sha(data)
    rec['peak_rss_bytes']=peak_rss()
    if rec['peak_rss_bytes']>12*1024**3: raise RuntimeError('RSS above 12 GiB')
    write_json(out/'record.json',rec)
def peak_rss():
    import psutil
    m=psutil.Process().memory_info();return int(getattr(m,'peak_wset',m.rss))
def arm(c,out):
    """Deny capture/target reads in the numerical process, audit every open path."""
    denied=[safe(c[k]).resolve() for k in ('dev_zip','texture_min','texture_full')]
    forbidden=safe(c['work'])/'targets';audit=[]
    def hook(event,args):
        if event not in ('open','os.listdir','os.scandir'): return
        if not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=safe(os.fsdecode(args[0])).resolve()
        if p in denied or forbidden.resolve() in p.parents: raise RuntimeError('predictor capture/target read denied')
        if event=='open':audit.append({'path':str(p),'mode':str(args[1]) if len(args)>1 else ''})
    sys.addaudithook(hook);return audit

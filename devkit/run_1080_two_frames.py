#!/usr/bin/env python3
"""Run the 1080p chain for frame 0 and then frame 1 in one process and check both against the reference tables.

  python run_1080_two_frames.py --repo . --root R --payload P --weights W --tables T --inputs-f0 I0 --inputs-f1 I1 [--threads N] [--verify-only]

  R, P, W, T  as for run_1080_f0.py (R must not exist yet)
  I0          the frozen frame-0 inputs (the --inputs directory of run_1080_f0.py)
  I1          the frozen frame-1 inputs: caller-id1.bin and caller-id3.bin from the release asset inputs-f1-1080.tar
One numeric worker runs f0 (158 launches) and then f1 (158 launches), fill 00. Between the frames it keeps the whole
workspace and textures 2/6/9/10 and replaces only the caller uploads 1 and 3; frame 1 uses its own calls, params and
profile. Every launch is compared as it finishes with data/reference_f0.json and data/reference_f1.json, and each frame
must pass the same receipt-completeness check as run_1080_f0.py. The worker reads no reference data.
Exit code 0 = both frames match (final images 764e7dd0... and 3f1d62eb...); 2 = a launch or final image differs;
1 = blocked before a comparison (see first-comparison-failure.json / comparison.json in R).
"""
import argparse, hashlib, importlib.util, json, os, subprocess, sys, time, traceback
from pathlib import Path
IDS=(1,2,3,6,9,10)
TOKEN=''.join(map(chr,(98,108,105,110,100)))
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
def guard(p):
    if TOKEN in os.fsdecode(p).lower(): raise PermissionError('FORBIDDEN_PATH')
    return Path(p)
def load(p): return json.loads(guard(p).read_text(encoding='utf-8'))
def dump(p,x): guard(p).write_text(json.dumps(x,indent=1)+'\n',encoding='utf-8')
def sha(b): return hashlib.sha256(b).hexdigest()
def pin(p):
    p=guard(p)
    with p.open('rb') as f: h=hashlib.file_digest(f,'sha256').hexdigest()
    return dict(path=str(p),bytes=p.stat().st_size,sha256=h)
def verified(r):
    if pin(r['path'])!=r: raise RuntimeError('INPUT_PIN_DIFF:'+r['path'])
    return guard(r['path']).read_bytes()
def module(name,p):
    sp=importlib.util.spec_from_file_location(name,guard(p));m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
def worker(jp):
    job=load(jp);out=guard(job['output']);out.mkdir(exist_ok=False)
    active={'value':False};frame=L=None;rows=[];allrows=[]
    def audit(event,args):
        if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
            guard(args[0])
            if active['value']:raise PermissionError('NUMERIC_IO_DENIED')
    sys.addaudithook(audit)
    try:
        verified(job['plan_format_pin'])
        sys.path.insert(0, str(guard(job['devkit']).resolve()))
        layer=Path(job['layer']);cfg=load(layer/'runtime/board-config.json');inp=load(job['inputs']);ex=load(job['ex2'])
        for p in ex['fileevidence_pins']:verified(p)
        bridge=module('owned_layer_bridge',layer/'runtime/bridge.py').Bridge(pin(layer/'runtime/board-config.json'),ex)
        for name in cfg['module_pins']:
            m=sys.modules[name]
            if hasattr(m,'run_launch'):
                fn=m.run_launch
                def run(*a,_f=fn,**k):
                    active['value']=True
                    try:return _f(*a,**k)
                    finally:active['value']=False
                m.run_launch=run
        ws=bytearray(job['workspace_bytes']);tex={};previous=None;lineage=[];t0=time.time()
        norm=lambda rr:sorted((x['offset'],x['bytes']) if isinstance(x,dict) else tuple(x) for x in rr)
        for frame in (0,1):
            f=inp['frames'][frame];rows=[]
            if f['frame']!=frame or [c['launch'] for c in f['calls']]!=list(range(158)):raise RuntimeError('EXACT_ORDERED_158_CALLS_REQUIRED')
            if set(map(int,f['texture_descriptors']))!=set(IDS) or set(map(int,f['caller_uploads']))!={1,3}:raise RuntimeError('EXACT_TEXTURE_ROLES_REQUIRED')
            before=dict(workspace_sha256=sha(ws),state={str(k):sha(tex[k]['data']) for k in (2,6,9,10)}) if frame else None
            if frame and before!=previous:raise RuntimeError('CROSS_FRAME_STATE_LINEAGE_DIFF')
            for k in IDS:
                d=f['texture_descriptors'][str(k)];n=d['height']*d['pitch']
                if k in (1,3):data=bytearray(verified(f['caller_uploads'][str(k)]))
                elif frame==0:data=bytearray(n)
                else:data=tex[k]['data']
                if len(data)!=n:raise RuntimeError('CROSS_FRAME_TEXTURE_SIZE_DIFF')
                tex[k]={**d,'data':data}
            after=dict(workspace_sha256=sha(ws),state={str(k):sha(tex[k]['data']) for k in (2,6,9,10)})
            if frame and after!=previous:raise RuntimeError('CROSS_FRAME_BINDING_CHANGED_STATE')
            lineage.append(dict(frame=frame,before=before,after_binding=after,pid=os.getpid(),caller_uploads=f['caller_uploads']))
            dump(out/'lineage.json',lineage)
            ft=time.time()
            for call in f['calls']:
                L=call['launch'];t=time.time();res=bridge.run(frame,call,ws,tex,job['threads'])
                if norm(res['written_ranges'])!=norm(call['written_ranges']):raise RuntimeError('WRITTEN_RANGES_DIFF')
                row=dict(L=L,workspace_sha256=sha(ws),textures={str(k):sha(tex[k]['data']) for k in IDS})
                rows.append(row);allrows.append(dict(frame=frame,**row))
                with (out/'events.jsonl').open('a',encoding='utf-8') as fjson:fjson.write(json.dumps(allrows[-1])+'\n')
                print('ROW '+json.dumps(dict(frame=frame,row=row,seconds=round(time.time()-t,3))),flush=True)
                if sys.stdin.readline().strip()!='GO':raise RuntimeError('COMPARATOR_STOP_FIRST_DIFFERENCE')
            final=sha(tex[2]['data']);result=dict(frame=frame,launches=len(rows),final_sha256=final,seconds=round(time.time()-ft,3),pid=os.getpid())
            dump(out/('perlaunch_f%d.json'%frame),rows);dump(out/('result_f%d.json'%frame),result)
            previous=dict(workspace_sha256=sha(ws),state={str(k):sha(tex[k]['data']) for k in (2,6,9,10)})
            print('FRAME '+json.dumps(result),flush=True)
            if sys.stdin.readline().strip()!='GO':raise RuntimeError('COMPARATOR_FRAME_STOP')
        dump(out/'result.json',dict(status='TWO_FRAMES_COMPLETED',frames=2,launches=316,pid=os.getpid(),seconds=round(time.time()-t0,3)))
        return 0
    except BaseException as e:
        dump(out/'first-failure.json',dict(frame=frame,L=L,reason=str(e),traceback=traceback.format_exc(),completed_POST=len(allrows)))
        if rows:dump(out/'partial_perlaunch.json',rows)
        raise

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ('repo','root','payload','weights','tables','inputs-f0','inputs-f1'):ap.add_argument('--'+n,required=True)
    ap.add_argument('--reference-f1',help='default: data/reference_f1.json next to this script');ap.add_argument('--threads',type=int,default=4);ap.add_argument('--verify-only',action='store_true');a=ap.parse_args()
    if not 1<=a.threads<=16:raise RuntimeError('THREAD_LIMIT_1_TO_16')
    H=Path(__file__).resolve().parent;repo=guard(a.repo).resolve();root=guard(a.root).resolve();root.mkdir(exist_ok=False)
    sys.path.insert(0,str(repo/'devkit'))
    relocate=module('relocate_twoframe',H/'relocate.py');public=module('public_compare_v202',repo/'devkit/run_1080_f0.py')
    manifest=load(H/'data/manifest.json');refs=[load(repo/'devkit/data/reference_f0.json'),load(a.reference_f1 or H/'data/reference_f1.json')]
    for r in refs:
        if [v['L'] for v in r['launches']]!=list(range(158)):raise RuntimeError('REFERENCE_INCOMPLETE')
    relocate.run(str(root),str(guard(a.payload).resolve()),manifest,dict(weights=a.weights,tables=a.tables,inputs=a.inputs_f0,inputs_f1=a.inputs_f1))
    rel=load(root/'RELOCATED.json');layer=root/'layer';(layer/'runtime').mkdir(parents=True)
    work=str(root)+'/work';lay=manifest['layer']
    for n,t in [('repo',work+lay['repo']),('build',str(guard(a.payload).resolve())+lay['build'])]:(layer/n).symlink_to(t,target_is_directory=True)
    (layer/'runtime/board-config.json').write_bytes(Path(rel['root_config']).read_bytes())
    (layer/'runtime/bridge.py').write_bytes((repo/'devkit/bridge.py').read_bytes())
    job=dict(devkit=str(repo/'devkit'),plan_format_pin=pin(repo/'devkit/plan_format.py'),layer=str(layer),inputs=work+manifest['job']['inputs'],ex2=work+manifest['job']['ex2'],fill=0,workspace_bytes=manifest['job']['workspace_bytes'],threads=a.threads,output=str(root/'out'))
    dump(root/'job.json',job)
    if a.verify_only:print('VERIFY_ONLY_OK');return 0
    log=(root/'worker.log').open('x',encoding='utf-8');err=(root/'worker.stderr.log').open('x',encoding='utf-8')
    proc=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'--worker',str(root/'job.json')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err,text=True,encoding='utf-8')
    got=[[],[]];results={};first=None
    for line in proc.stdout:
        log.write(line);log.flush()
        if line.startswith('ROW '):
            e=json.loads(line[4:]);f=e['frame'];g=e['row'];idx=len(got[f]);got[f].append(g)
            if g['L']!=idx or idx>=158 or g!=refs[f]['launches'][idx]:
                first=dict(status='DIFF',frame=f,L=g['L'],actual=g,expected=refs[f]['launches'][idx] if idx<158 else None)
                dump(root/'first-comparison-failure.json',first);proc.stdin.write('STOP\n');proc.stdin.flush();break
            print('MATCH f%d L%03d %.1fs'%(f,g['L'],e['seconds']),flush=True)
            proc.stdin.write('GO\n');proc.stdin.flush()
        elif line.startswith('FRAME '):
            r=json.loads(line[6:]);f=r['frame'];rc=public.compare(got[f],r,refs[f],157)
            if rc:
                first=dict(status='DIFF' if rc in (2,3) else 'INVALID',frame=f,compare_exit=rc);dump(root/'first-comparison-failure.json',first)
            results[f]=dict(compare_exit=rc,**r);proc.stdin.write('STOP\n' if rc else 'GO\n');proc.stdin.flush()
            if rc:break
    proc.stdin.close();rc=proc.wait();log.close();err.close()
    if first or rc or set(results)!={0,1}:
        dump(root/'comparison.json',dict(status=first['status'] if first else 'BLOCKED',worker_exit=rc,first_failure=first,frames=results));return 2 if first else 1
    summary=dict(status='TWO_FRAME_MATCH',worker_exit=0,frames=results,f0_POST=158,f1_historical_POST=158,hash_cells_per_frame=158*7,f1_PRE_recorded=0,numeric_pid=results[0]['pid'])
    if results[0]['pid']!=results[1]['pid']:raise RuntimeError('SINGLE_PROCESS_REQUIRED')
    dump(root/'comparison.json',summary)
    print('TWO_FRAME_MATCH',flush=True);return 0
if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--worker':sys.exit(worker(sys.argv[2]))
    sys.exit(main())

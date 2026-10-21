"""Frozen fixed-profile bridge: native-pointer entry/tail and legacy middle ABI.

No target APIs. Frame state and texture storage are owned by the caller.
This bridges a bounded numerical diagnostic, not a complete native read/write proof.
"""
import hashlib,importlib.util,json,sys,struct,mmap,threading,os
from pathlib import Path
_GUARD_TOKEN = ''.join(map(chr, (98, 108, 105, 110, 100)))      # path-guard token, built at run time
H=Path(__file__).resolve().parents[1]
IDS={1,2,3,6,9,10}
def raw(p):
 p=Path(p)
 if _GUARD_TOKEN in str(p).lower():raise RuntimeError('FORBIDDEN_PATH')
 return p.read_bytes()
def pin(p):b=raw(p);return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def verified(r):
 b=raw(r['path'])
 if len(b)!=r['bytes'] or hashlib.sha256(b).hexdigest()!=r['sha256']:raise RuntimeError('SOURCE_INPUT_PIN_DIFF')
 return b
def load(p):return json.loads(raw(p))
def module(r,name):
 verified(r)
 if name in sys.modules:
  m=sys.modules[name]
  if pin(m.__file__)!=r:raise RuntimeError('MODULE_ALREADY_LOADED_OTHER_IDENTITY')
  return m
 spec=importlib.util.spec_from_file_location(name,r['path']);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
 if pin(m.__file__)!=r:raise RuntimeError('MODULE_IDENTITY_CHANGED')
 return m
class Bridge:
 def __init__(self,config_pin,ex2_profile=None):
  verified(config_pin);self.cfg=load(config_pin['path']);self.cfgpin=config_pin;self.ex=ex2_profile;self.events=[];self.assets_cache={};self.tail_tables={}
  root=H/'repo/cpp/opt/M';sys.path.insert(0,str(root));sys.path.insert(0,str(H/'build-native-v2'))
  # Verify all declared executable sources and modules before any kit import.
  for item in list(self.cfg['ABI'].values())+list(self.cfg['loader_pins'].values())+list(self.cfg['module_pins'].values()):verified(item)
  for name,item in sorted(self.cfg['module_pins'].items()):module(item,name)
  abi=module(self.cfg['ABI']['backend_abi'],'backend_abi')
  self.abi=abi;self.routes={r['launch']:r for r in load(self.cfg['route_table'])['rows']}
  self.entry=sys.modules['dlssnr_chain_entry_optm'];self.tail=sys.modules['dlssnr_chain_outtail_optm']
  self.reach_descriptors=json.loads(verified(self.cfg['mufu_raw_descriptors']))
  self.entry_cfg=load(self.cfg['backend_configs']['entry']);self.tail_cfg=load(self.cfg['backend_configs']['outtail'])
  self.full_weights=None;self.caller_weights=None;self.tail_weights=None;self.tail_weight_identity=None
  self._tail_lock=threading.Lock();self._tail_frame=None;self._tail_next=None;self._tail_started_frame=None;self._tail_joined=False;self._tail_placeholder=None
  self.profile_pins=self.cfg['profile_configs']
  self.geo_per_launch=self.cfg.get('geometry_per_launch_profiles',{})
  self.geo_input=load(self.cfg['geometry_input_descriptor']['path'])
  gd={k:v for k,v in self.geo_input.items() if k not in ('native_param_pins','module_overrides')}
  gd['native_params']={i:verified(v) for i,v in enumerate(self.geo_input['native_param_pins'])}
  gd['textures']={int(k):v for k,v in gd['textures'].items()}
  self._gd=gd;self._geo_modules=('dlssnr_chain_entry_optm','dlssnr_chain_tin_optm','dlssnr_chain_split16_optm','dlssnr_chain_outtail_optm');self._geo_key=tuple(hashlib.sha256(gd['native_params'][i]).hexdigest() for i in range(len(gd['native_params'])))
  for name in ('dlssnr_chain_entry_optm','dlssnr_chain_tin_optm','dlssnr_chain_split16_optm','dlssnr_chain_outtail_optm'):
   descriptor=dict(gd,**self.geo_input.get('module_overrides',{}).get(name,{}));sys.modules[name].configure_geometry(descriptor)
  self.entry_helper=module(self.cfg['loader_pins']['entry_assets'],'entry_assets_bridge')
  self.tail_helper=module(self.cfg['loader_pins']['tail_runtime'],'tail_runtime_bridge')
  self.weight_helper=module(self.cfg['loader_pins']['outtail_weight'],'tail_weight_bridge')
 def load_entry_weights(self):
  if self.full_weights is None:
   self.full_weights=raw(self.entry_cfg['device_weights'])
   if len(self.full_weights)!=147719680 or hashlib.sha256(self.full_weights).hexdigest()!='a5513b1845c98a486985ed04f38e66a1854cce33c2aba3a505866028bd4ee3e5':raise RuntimeError('FULL_CANONICAL_UPLOAD_DIFF')
 def load_caller_weights(self):
  if self.caller_weights is None:
   self.caller_weights=raw(self.cfg['paths']['weights_ht'])
   if hashlib.sha256(self.caller_weights).hexdigest()!=self.cfg['weights_ht_container_sha256']:raise RuntimeError('LOGICAL_VIT_CONTAINER_DIFF')
 def load_caller_weights_mmap(self):
  if os.name=="nt":return self.load_caller_weights_mmap_windows()
  if self.caller_weights is not None:raise RuntimeError('M01_CALLER_OWNER_ALREADY_LIVE')
  p=Path(self.cfg['paths']['weights_ht'])
  if _GUARD_TOKEN in str(p).lower():raise RuntimeError('FORBIDDEN_PATH')
  with p.open('rb') as f:
   mapped=mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ)
  try:
   if not hasattr(mapped,'madvise') or not hasattr(mmap,'MADV_DONTNEED'):
    raise RuntimeError('OPTION_B_PAGE_RELEASE_UNAVAILABLE_STOP')
   h=hashlib.sha256();n=len(mapped)
   for off in range(0,n,8<<20):
    size=min(8<<20,n-off);view=memoryview(mapped)[off:off+size]
    try:h.update(view)
    finally:view.release()
    mapped.madvise(mmap.MADV_DONTNEED,off,size)
   if h.hexdigest()!=self.cfg['weights_ht_container_sha256']:raise RuntimeError('LOGICAL_VIT_CONTAINER_DIFF')
  except BaseException:
   mapped.close();raise
  self.caller_weights=mapped
 def load_caller_weights_mmap_windows(self):
  if self.caller_weights is not None:raise RuntimeError('M01_CALLER_OWNER_ALREADY_LIVE')
  p=Path(self.cfg['paths']['weights_ht'])
  if _GUARD_TOKEN in str(p).lower():raise RuntimeError('FORBIDDEN_PATH')
  mapped=None
  with p.open('rb') as f:
   before=os.fstat(f.fileno());mapped=mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ)
   try:
    h=hashlib.sha256();n=len(mapped)
    for off in range(0,n,8<<20):
     view=memoryview(mapped)[off:min(n,off+(8<<20))]
     try:h.update(view)
     finally:view.release()
    if h.hexdigest()!=self.cfg['weights_ht_container_sha256']:raise RuntimeError('LOGICAL_VIT_CONTAINER_DIFF')
    after=os.fstat(f.fileno())
    if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns):raise RuntimeError('M01_CALLER_FD_IDENTITY_CHANGED')
    # Closing the verified view drops this process's mapped page residency.
    # Reopen the same still-open fd as a fresh read-only view; unused routes do
    # not touch payload pages. No caller byte or forwarded identity is changed.
    mapped.close();mapped=mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ)
   except BaseException:
    if mapped is not None:mapped.close()
    raise
  self.caller_weights=mapped
 def release_caller_weights_m01(self):
  owner=self.caller_weights;self.caller_weights=None
  if isinstance(owner,mmap.mmap):owner.close()
 def release_packed_after_join_m01(self):
  state=self.abi._state
  if state is not None and state[0] in ('tin','split16'):
   # run_launch has returned or unwound: borrower views and native threads joined.
   # The original ABI already reloads packed on every configure. Keep owner/mod.
   self.abi._state=(state[0],state[1],None)
 def load_tail_weights(self):
  if self.tail_weights is None:
   if self.cfg.get('geometry_tail_upload'):
    v=self.cfg['geometry_tail_upload'];self.tail_weights=verified(v);self.tail_weight_identity=dict(upload_bytes=v['bytes'],upload_sha256=v['sha256'],geometry_canonical_original=v)
   else:self.tail_weights,self.tail_weight_identity=self.weight_helper.weight_upload(self.tail_cfg)
   if len(self.tail_weights)!=self.geo_input['tail_upload_bytes']:raise RuntimeError('TAIL_CANONICAL_UPLOAD_LENGTH_DIFF')
 def entry_assets(self,L,wb):
  self.entry_helper.PINS={};self.entry_helper.READ_LOG.clear()
  return dict(self.entry_helper.assets(dict(self.entry_cfg,optc_asset_pins=self.cfg['optc_asset_pins']),L),native_device_weight_base=wb)
 def tail_desc(self,L,params):
  if L==155 and struct.unpack_from('<Q',params,88)[0]:
   if not self.ex:raise RuntimeError('QUALIFIED_SM120_TEMPORAL_EX2_REQUIRED')
   manifest=load(Path(self.tail_cfg['work'])/'tables/manifest.json');d=dict(manifest['runtime'])
   for k in ('ex2','rcp','rsq'):
    v=manifest['sources'][k];d[k]=dict(path=str(Path(self.tail_cfg['work'])/'tables'/(k+'.bin')),bytes=v['bytes'],sha256=v['extracted_sha256'])
   d.update(temporal_param_profiles=[params],ex2_sparse_input=self.ex['inputs'],ex2_sparse_output=self.ex['outputs'])
   return d
  if L not in self.tail_tables:self.tail_tables[L]=self.tail_helper.stage_tables(dict(self.tail_cfg,e1_encoded=self.cfg['e1_encoded']),L)
  return self.tail_tables[L]

 def run(self,frame,call,ws,textures,threads):
  L=call['launch'];owner=call['route_owner'];base=int(call['workspace_base'],0) if isinstance(call['workspace_base'],str) else call['workspace_base'];wb=int(call['native_device_weight_base'],0) if isinstance(call['native_device_weight_base'],str) else call['native_device_weight_base']
  if self.geo_per_launch:
   key='%d:%d'%(frame,L)
   if key in self.geo_per_launch:profile=self.geo_per_launch[key]
   elif frame==0 and str(L) in self.geo_per_launch:profile=self.geo_per_launch[str(L)]
   else:raise RuntimeError('MISSING_FRAME_LAUNCH_PROFILE:%d:%d'%(frame,L))
  else:profile=self.profile_pins[str(frame)]
  self.cfg=json.loads(verified(profile))
  self.tail_cfg=load(self.cfg['backend_configs']['outtail'])
  if self.cfg['bridge_parameter_pins'][str(L)]!=call['call_params']:raise RuntimeError('SEALED_PROFILE_CALL_PARAMETERS_DIFF')
  if frame>0:
   pins=self.cfg['bridge_parameter_pins'];key=tuple(pins[str(i)]['sha256'] for i in range(len(self._geo_key)))
   if key!=self._geo_key:
    gd2=dict(self._gd);gd2['native_params']={i:verified(pins[str(i)]) for i in range(len(self._geo_key))}
    for name in self._geo_modules:sys.modules[name].configure_geometry(dict(gd2,**self.geo_input.get('module_overrides',{}).get(name,{})))
    self._geo_key=key
  if {k for k in textures if isinstance(k,int)}!=IDS:raise RuntimeError('EXACT_SIX_OWNED_TEXTURES_REQUIRED')
  packet={}
  def prefix(params):
   packet['params']=params;self.events.append(dict(event='call_begin',frame=frame,launch=L,owner=owner))
   if owner not in ('entry','outtail'):
    unused=(owner in ('tin','split16') or L in (58,62,67,72,77,82,87,92,97,99))
    if unused:self.load_caller_weights_mmap()
    else:self.load_caller_weights()
  if owner=='entry':
   self.abi.configure(self.cfg,L,workspace_base=base,native_device_weight_base=wb,bridge_prefix=prefix)
   try:r=self.abi.run_launch(L,ws,packet['params'],None,textures,threads)
   finally:self.full_weights=None
  elif owner=='outtail':
   if not self._tail_lock.acquire(blocking=False):raise RuntimeError('M03_TAIL_REENTRY_REJECTED')
   accepted=False
   try:
    def tail_prefix(params):
     nonlocal accepted
     prefix(params)
     if L==155:
      if self._tail_next is not None or (self._tail_started_frame is not None and frame<=self._tail_started_frame):raise RuntimeError('M03_TAIL_ORDER_REJECTED')
      self._tail_started_frame=frame;self._tail_frame=frame;self._tail_joined=False;self.tail_weight_identity=None
     elif L in (156,157):
      receipt=self.tail_weight_identity;previous=self.events[-2] if len(self.events)>=2 else {}
      if self._tail_frame!=frame or self._tail_next!=L or not self._tail_joined or not isinstance(receipt,dict) or receipt.get('upload_bytes')!=self.geo_input['tail_upload_bytes'] or not isinstance(receipt.get('upload_sha256'),str) or len(receipt['upload_sha256'])!=64 or previous!={'event':'call_return_joined','frame':frame,'launch':L-1,'owner':'outtail'}:raise RuntimeError('M03_TAIL_ORDER_OR_RECEIPT_REJECTED')
     else:raise RuntimeError('M03_TAIL_LAUNCH_REJECTED')
     accepted=True
     if not self.tail.reach_stats()['registered']:
      self.tail.init_mufu_full_raw(self.reach_descriptors)
      expected=[{**{k:p[k] for k in ('op','block','path','sha256','bytes')},'storage':'verified read-only pinned fd; global two-block mmap cache; explicit frame/failure retirement'} for p in self.reach_descriptors]
      if list(self.tail.mufu_full_raw_pins())!=expected:raise RuntimeError('REACH_REGISTRY_ACTUAL_PINS_DIFF')
     if L==155:self.load_tail_weights()
     else:
      # Validate the original source identity without re-materializing packed.
      self.abi.verify(self.cfg['tail_weight_source_pin'])
      if self._tail_placeholder is None:
       if self.cfg.get('geometry_tail_upload'):
        v=self.cfg['geometry_tail_upload'];self.abi.verify(v)
        with Path(v['path']).open('rb') as f:self._tail_placeholder=mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ)
       elif hasattr(mmap,'PROT_READ') and hasattr(mmap,'MAP_ANONYMOUS'):
        self._tail_placeholder=mmap.mmap(-1,self.geo_input['tail_upload_bytes'],flags=mmap.MAP_PRIVATE|mmap.MAP_ANONYMOUS,prot=mmap.PROT_READ)
       else:self._tail_placeholder=mmap.mmap(-1,self.geo_input['tail_upload_bytes'],access=mmap.ACCESS_READ)
     desc=self.tail_desc(L,params);self.tail.init_table_files(L,desc)
     expected=sorted([dict(path=str(Path(v['path']).resolve()),bytes=v['bytes'],sha256=v['sha256']) for v in desc.values() if isinstance(v,dict) and 'path' in v],key=lambda v:v['path'])
     actual=sorted([dict(v,path=str(Path(v['path']).resolve())) for v in self.tail.optm_table_owners()['actual_reads']],key=lambda v:v['path'])
     if actual!=expected:raise RuntimeError('TAIL_ACTUAL_TABLE_READS_DIFF')
    self.abi.configure_bridge_outtail(self.cfg,L,self.tail,[self.tail_helper,self.weight_helper],tail_prefix)
    self.abi._guard.reject_aliases(L,ws,textures)
    textures['_workspace_base']=base;textures['_native_device_weight_base']=wb
    view=memoryview(self.tail_weights if L==155 else self._tail_placeholder)
    try:
     if view.itemsize!=1 or view.ndim!=1 or view.strides!=(1,) or view.nbytes!=self.geo_input['tail_upload_bytes'] or (L!=155 and not view.readonly):raise RuntimeError('M03_WEIGHT_BUFFER_METADATA_DIFF')
     r=self.tail.run_launch(L,ws,packet['params'],view,textures,threads)
    finally:
     view.release()
     if L==155:self.tail_weights=None
    if L==155:self._tail_joined=True
    self._tail_next=L+1 if L<157 else None
   except BaseException:
    if accepted:
     self.tail_weights=None;self._tail_joined=False;self._tail_next=None;self._tail_frame=None
     if self._tail_placeholder is not None:self._tail_placeholder.close();self._tail_placeholder=None
     self.tail.reach_clear()
    raise
   finally:
    try:
     if accepted:
      self.tail_weights=None
      try:self.tail.optm_clear_tables()
      except BaseException:
       self._tail_joined=False;self._tail_next=None
       raise
      finally:
       if L==157 or not self._tail_joined:
        if self._tail_placeholder is not None:self._tail_placeholder.close();self._tail_placeholder=None
        self._tail_frame=None;self._tail_next=None;self._tail_joined=False
        self.tail.reach_clear()
    finally:self._tail_lock.release()
  else:
   try:
    self.abi.configure(self.cfg,L,bridge_prefix=prefix)
    abi_owner=self.abi._state[0]
    if abi_owner not in ('vit','vit_adapter','tin','split16'):raise RuntimeError('ORIGINAL_ABI_CALLER_WEIGHT_ROUTE_UNKNOWN:'+str(abi_owner))
    forwarded=memoryview(self.caller_weights)
    try:r=self.abi.run_launch(L,ws,packet['params'],forwarded,textures,threads)
    finally:forwarded.release()
   finally:
    self.release_caller_weights_m01();self.release_packed_after_join_m01()
  if isinstance(r,dict) and r.get('textures_out'):
   for k,v in r['textures_out'].items():
    if k not in IDS:raise RuntimeError('UNKNOWN_OUTPUT_TEXTURE_ID')
    if isinstance(v,dict):textures[k]={**textures[k],**v,'data':bytearray(v['data'])}
    else:textures[k]['data']=bytearray(v)
  self.events.append(dict(event='call_return_joined',frame=frame,launch=L,owner=owner))
  return r

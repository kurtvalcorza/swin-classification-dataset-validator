from __future__ import annotations
import argparse, hashlib, json, os, re, tempfile, unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from PIL import Image, ImageOps, UnidentifiedImageError

TASK="core.task.vision.image-classification"
REP="core.dataset.vision.image-folder"
ALG="org.valcorza.swin-classification-validator.v1"
DIGEST_RE=re.compile(r"^sha256:[0-9a-f]{64}$")
EXT={".jpg":"image/jpeg",".jpeg":"image/jpeg",".png":"image/png",".bmp":"image/bmp",".tif":"image/tiff",".tiff":"image/tiff",".webp":"image/webp"}
SPLITS={"train":"train","val":"validation","valid":"validation","test":"test"}

class ValidationFailure(RuntimeError):
    def __init__(self,finding:dict[str,Any]): super().__init__(finding["message"]); self.finding=finding

@dataclass(frozen=True)
class RuntimeBinding:
    job_id:str; attempt_id:str; worker_release_digest:str; effective_job_spec_digest:str; admission_record_digest:str; security_grant_digest:str; resource_binding_digests:tuple[str,...]=()
    def validate(self):
        if not self.job_id or not self.attempt_id: raise ValueError("job_id and attempt_id are required")
        for n,v in [("worker_release_digest",self.worker_release_digest),("effective_job_spec_digest",self.effective_job_spec_digest),("admission_record_digest",self.admission_record_digest),("security_grant_digest",self.security_grant_digest)]:
            if not DIGEST_RE.fullmatch(v): raise ValueError(f"{n} must be sha256:<64 lowercase hex>")
        if any(not DIGEST_RE.fullmatch(v) for v in self.resource_binding_digests): raise ValueError("resource_binding_digests must be sha256:<64 lowercase hex>")

def cbytes(v): return json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()
def dbytes(v:bytes): return "sha256:"+hashlib.sha256(v).hexdigest()
def djson(v): return dbytes(cbytes(v))
def fdigest(p:Path):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return "sha256:"+h.hexdigest()
def atomic(path:Path,v):
    path.parent.mkdir(parents=True,exist_ok=True); data=cbytes(v)+b"\n"; fd,tmp=tempfile.mkstemp(prefix="."+path.name+".",dir=path.parent)
    try:
        with os.fdopen(fd,"wb") as f: f.write(data); f.flush(); os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)
    return dbytes(data[:-1])
def finding(code,severity,layer,scope,message,observed,expected,evidence=None):
    return {"code":code,"severity":severity,"layer":layer,"scope":scope,"location":None,"message":message,"observed":observed,"expected":expected,"evidence":evidence or {}}
def fail(code,layer,scope,message,observed,expected,**evidence): raise ValidationFailure(finding(code,"FATAL",layer,scope,message,observed,expected,evidence))

def safe_rel(root:Path,p:Path):
    if p.is_symlink(): fail("VISION_SYMLINK_REJECTED","L0","dataset.path","Symbolic links are forbidden.",str(p),"regular path")
    try: return p.resolve(strict=True).relative_to(root).as_posix()
    except ValueError: fail("VISION_PATH_ESCAPE","L0","dataset.path","Path resolves outside dataset root.",str(p),str(root))
def root_checked(root:Path):
    if not root.exists(): fail("VISION_DATASET_ROOT_MISSING","L0","dataset.root","Dataset root does not exist.",str(root),"existing directory")
    if root.is_symlink(): fail("VISION_SYMLINK_REJECTED","L0","dataset.root","Dataset root must not be a symlink.",str(root),"non-symlink directory")
    if not root.is_dir(): fail("VISION_DATASET_ROOT_NOT_DIRECTORY","L0","dataset.root","Dataset root is not a directory.",str(root),"directory")
    return root.resolve(strict=True)
def source_digest(root:Path):
    rows=[]
    for cur,dirs,files in os.walk(root,topdown=True,followlinks=False):
        dirs.sort(); files.sort(); cp=Path(cur)
        for n in dirs: safe_rel(root,cp/n)
        for n in files:
            p=cp/n; rows.append({"path":safe_rel(root,p),"digest":fdigest(p)})
    return djson({"algorithmId":ALG+".source-snapshot","files":rows})
def decode(p:Path,media:str):
    try:
        with Image.open(p) as im: fmt=im.format; im.verify()
        with Image.open(p) as im: im=ImageOps.exif_transpose(im); im.load()
    except (UnidentifiedImageError,OSError,ValueError) as e:
        fail("VISION_IMAGE_DECODE_FAILED","L1","dataset.image","Image decode failed; synthetic replacement pixels are forbidden.",str(p),"decodable image",error=type(e).__name__)
    actual={"JPEG":"image/jpeg","PNG":"image/png","BMP":"image/bmp","TIFF":"image/tiff","WEBP":"image/webp"}.get(fmt or "")
    if actual!=media: fail("VISION_IMAGE_EXTENSION_FORMAT_MISMATCH","L1","dataset.image","Decoded image format contradicts extension.",actual,media,path=str(p))

def inspect_dataset(root:Path):
    root=root_checked(root); src=source_digest(root); entries=sorted(root.iterdir(),key=lambda p:p.name)
    unknown=[p.name for p in entries if p.name not in SPLITS]
    if unknown: fail("VISION_UNKNOWN_ROOT_ENTRY","L1","dataset.structure","Dataset root contains unsupported entries.",unknown,sorted(SPLITS))
    present={p.name for p in entries if p.is_dir()}
    if "val" in present and "valid" in present: fail("VISION_AMBIGUOUS_VALIDATION_SPLIT","L1","dataset.splits","Both val/ and valid/ exist; no precedence is defined.",["val","valid"],"exactly one validation directory")
    if "train" not in present: fail("VISION_MISSING_TRAIN_SPLIT","L2","dataset.splits","train/ is required.",sorted(present),"train present")
    if not ({"val","valid"}&present): fail("VISION_MISSING_VALIDATION_SPLIT","L2","dataset.splits","A validation split is required; test/ is never reinterpreted as validation.",sorted(present),"val/ or valid/ present")
    samples=[]; assets=[]; assignments=[]; classes=set(); class_presence={}; warnings=[]
    for sd in ("train","val","valid","test"):
        sp=root/sd
        if not sp.exists(): continue
        safe_rel(root,sp); logical=SPLITS[sd]; cdirs=sorted(sp.iterdir(),key=lambda p:p.name)
        if not cdirs: fail("VISION_EMPTY_SPLIT","L1","dataset.splits","Split directory is empty.",sd,"one or more class directories")
        seen={}
        for cd in cdirs:
            safe_rel(root,cd)
            if not cd.is_dir(): fail("VISION_UNASSIGNED_SPLIT_FILE","L1","dataset.structure","Files directly inside split directories are not assignable to a class.",cd.name,"class directory")
            label=cd.name; key=unicodedata.normalize("NFC",label).casefold()
            if key in seen and seen[key]!=label: fail("VISION_CLASS_NAME_COLLISION","L2","dataset.classes","Class names collide under normalization/case folding.",[seen[key],label],"distinct normalized labels")
            seen[key]=label; classes.add(label); class_presence.setdefault(label,set()).add(logical); ims=sorted(cd.iterdir(),key=lambda p:p.name)
            if not ims: fail("VISION_EMPTY_CLASS_DIRECTORY","L1","dataset.classes","Class directory is empty.",cd.as_posix(),"one or more image files")
            for p in ims:
                rel=safe_rel(root,p)
                if p.is_dir(): fail("VISION_NESTED_CLASS_DIRECTORY","L1","dataset.structure","Nested directories below a class directory are forbidden.",rel,"image file")
                media=EXT.get(p.suffix.lower())
                if not media: fail("VISION_UNRECOGNIZED_IMAGE_EXTENSION","L1","dataset.image","Unrecognized image extension; files are never silently skipped.",p.suffix.lower(),sorted(EXT),path=rel)
                decode(p,media); dg=fdigest(p); samples.append({"sampleId":rel,"assetIds":[rel],"sourceLocator":rel}); assets.append({"assetId":rel,"digest":dg,"mediaType":media}); assignments.append({"sampleId":rel,"split":logical,"reason":"directory-mapping"})
    if not samples: fail("VISION_NO_SAMPLES","L1","dataset.samples","No image samples were discovered.",0,">= 1")
    seen={}
    for sid in [s["sampleId"] for s in samples]:
        k=unicodedata.normalize("NFC",sid).casefold()
        if k in seen and seen[k]!=sid: fail("VISION_SAMPLE_ID_COLLISION","L1","dataset.samples","Logical sample IDs collide after canonicalization.",[seen[k],sid],"unique canonical sample IDs")
        seen[k]=sid
    amap={a["sampleId"]:a["split"] for a in assignments}; dig={a["assetId"]:a["digest"] for a in assets}
    logical_rows=[{"sampleId":s["sampleId"],"split":amap[s["sampleId"]],"className":Path(s["sampleId"]).parts[1],"contentDigest":dig[s["sampleId"]]} for s in sorted(samples,key=lambda x:x["sampleId"])]
    logical=djson({"algorithmId":ALG+".logical-dataset","representationProfile":REP,"samples":logical_rows})
    cls=sorted(classes); semantic={"schemaVersion":"1.0","taskProfile":TASK,"fields":[{"id":"image","sourceField":None,"logicalType":"core.type.image","semanticRole":"core.role.image","nullable":False},{"id":"target.class","sourceField":None,"logicalType":"core.type.class-label","semanticRole":"core.role.target.class","nullable":False}],"labelMap":{str(i):n for i,n in enumerate(cls)}}
    plan={"schemaVersion":"1.0","logicalDatasetDigest":logical,"assignments":sorted(assignments,key=lambda x:x["sampleId"]),"seedPolicy":None}
    manifest={"schemaVersion":"1.0","sourceArtifactDigest":src,"representationProfile":REP,"logicalDatasetDigest":logical,"samples":sorted(samples,key=lambda x:x["sampleId"]),"assets":sorted(assets,key=lambda x:x["assetId"])}
    for c in cls:
        missing=sorted({"train","validation"}-class_presence.get(c,set()))
        if missing: warnings.append(finding("VISION_CLASS_ABSENT_FROM_REQUIRED_SPLIT","WARNING","L3","dataset.classes","A class is absent from one or more required splits.",{"className":c,"missingSplits":missing},"class represented in train and validation",{"className":c}))
    # SPL9: byte-identical content in more than one split is leakage evidence. Detected by content digest,
    # reported as a WARNING with every affected sample so the operator can act; never dropped or re-split.
    by_digest={}
    for a in assets: by_digest.setdefault(a["digest"],[]).append(a["assetId"])
    leaked=[]
    for dg,ids in sorted(by_digest.items()):
        splits=sorted({amap[i] for i in ids})
        if len(splits)>1: leaked.append({"contentDigest":dg,"sampleIds":sorted(ids),"splits":splits})
    if leaked: warnings.append(finding("VISION_DUPLICATE_CONTENT_ACROSS_SPLITS","WARNING","L3","dataset.splits","Byte-identical images appear in more than one split; held-out metrics on those samples are not independent evidence.",{"duplicateGroups":len(leaked),"groups":leaked},"each image content present in at most one split"))
    return {"sourceArtifactDigest":src,"logicalDatasetDigest":logical,"logicalManifest":manifest,"semanticSchema":semantic,"dataPlan":plan,"warnings":warnings,"classNames":cls}

def evidence(logical,layer,findings,deps=()):
    return {"schemaVersion":"1.0","logicalDatasetDigest":logical,"layer":layer,"algorithmId":ALG+"."+layer.lower(),"outcome":"FAIL" if any(f["severity"] in {"ERROR","FATAL"} for f in findings) else "PASS","findings":findings,"dependencies":list(deps),"cacheability":"REUSABLE"}

def validate_dataset(root:Path,out:Path,b:RuntimeBinding):
    b.validate(); out.mkdir(parents=True,exist_ok=True)
    cfg={"algorithmId":ALG,"taskProfile":TASK,"representationProfile":REP,"acceptedImageExtensions":sorted(EXT),"exifOrientation":"transpose-to-visual-orientation","splitMapping":SPLITS,"requireTrain":True,"requireValidation":True,"testAsValidation":False}; cfgd=djson(cfg)
    ep={"schemaVersion":"1.0","jobId":b.job_id,"attemptId":b.attempt_id,"role":"validator","workerReleaseDigest":b.worker_release_digest,"effectiveJobSpecDigest":b.effective_job_spec_digest,"resourceBindingDigests":list(b.resource_binding_digests),"admissionRecordDigest":b.admission_record_digest,"securityGrantDigest":b.security_grant_digest}; epd=atomic(out/"execution-plan.json",ep)
    try: x=inspect_dataset(root)
    except ValidationFailure as e:
        atomic(out/"validation-findings.json",{"schemaVersion":"1.0","algorithmId":ALG,"findings":[e.finding]})
        rm={"schemaVersion":"1.0","jobId":b.job_id,"attemptId":b.attempt_id,"executionPlanDigest":epd,"workerReleaseDigest":b.worker_release_digest,"outcome":"FAILED","observed":{"findingCodes":[e.finding["code"]]},"artifactManifestDigest":None,"evaluationReportDigest":None,"reproducibility":"IDENTIFIED"}; rmd=atomic(out/"run-manifest.json",rm)
        atomic(out/"result.json",{"schemaVersion":"1.0","jobId":b.job_id,"attemptId":b.attempt_id,"state":"FAILED","failure":{"code":e.finding["code"],"category":"INPUT","retryable":False,"origin":ALG,"details":{"finding":e.finding}},"runManifestDigest":rmd,"artifactManifestDigest":None}); raise
    lmd=atomic(out/"logical-dataset-manifest.json",x["logicalManifest"]); ssd=atomic(out/"semantic-dataset-schema.json",x["semanticSchema"]); dpd=atomic(out/"data-plan.json",x["dataPlan"])
    ed=[]; prev=[]
    for layer,finds in [("L0",[]),("L1",[]),("L2",[]),("L3",x["warnings"]),("L4",[])]:
        dg=atomic(out/"evidence"/(layer.lower()+".json"),evidence(x["logicalDatasetDigest"],layer,finds,prev)); ed.append(dg); prev=[dg]
    vm={"schemaVersion":"1.0","logicalDatasetDigest":x["logicalDatasetDigest"],"datasetProfileDigest":None,"semanticSchemaDigest":ssd,"dataPlanDigest":dpd,"validationEvidenceDigests":ed,"validatorWorkerReleaseDigest":b.worker_release_digest,"effectiveValidationConfigDigest":cfgd,"policySetDigest":None}; vmd=atomic(out/"validated-dataset-manifest.json",vm)
    vi={"schemaVersion":"1.0","algorithmId":ALG+".validated-dataset-identity","validatedDatasetManifestDigest":vmd,"logicalDatasetDigest":x["logicalDatasetDigest"],"effectiveValidationConfigDigest":cfgd}; vi["digest"]=djson({"manifest":vmd,"logical":x["logicalDatasetDigest"],"config":cfgd,"worker":b.worker_release_digest,"semantic":ssd,"dataPlan":dpd,"evidence":ed}); vid=atomic(out/"validated-dataset-identity.json",vi)
    rm={"schemaVersion":"1.0","jobId":b.job_id,"attemptId":b.attempt_id,"executionPlanDigest":epd,"workerReleaseDigest":b.worker_release_digest,"outcome":"SUCCEEDED","observed":{"sourceArtifactDigest":x["sourceArtifactDigest"],"logicalDatasetManifestDigest":lmd,"validatedDatasetManifestDigest":vmd,"validatedDatasetIdentityDigest":vid,"sampleCount":len(x["logicalManifest"]["samples"]),"classNames":x["classNames"]},"artifactManifestDigest":None,"evaluationReportDigest":None,"reproducibility":"REEXECUTABLE"}; rmd=atomic(out/"run-manifest.json",rm)
    atomic(out/"result.json",{"schemaVersion":"1.0","jobId":b.job_id,"attemptId":b.attempt_id,"state":"SUCCEEDED","failure":None,"runManifestDigest":rmd,"artifactManifestDigest":None})
    return {"validatedDatasetManifestDigest":vmd,"validatedDatasetIdentityDigest":vid,"logicalDatasetDigest":x["logicalDatasetDigest"],"dataPlanDigest":dpd,"semanticDatasetSchemaDigest":ssd,"runManifestDigest":rmd}

def parser():
    p=argparse.ArgumentParser(); p.add_argument("dataset_root",type=Path); p.add_argument("output_dir",type=Path)
    for a in ("job-id","attempt-id","worker-release-digest","effective-job-spec-digest","admission-record-digest","security-grant-digest"): p.add_argument("--"+a,required=True)
    p.add_argument("--resource-binding-digest",action="append",default=[]); return p
def main(argv=None):
    a=parser().parse_args(argv); b=RuntimeBinding(a.job_id,a.attempt_id,a.worker_release_digest,a.effective_job_spec_digest,a.admission_record_digest,a.security_grant_digest,tuple(a.resource_binding_digest))
    try: summary=validate_dataset(a.dataset_root,a.output_dir,b)
    except ValidationFailure as e: print(json.dumps({"state":"FAILED","finding":e.finding},sort_keys=True)); return 2
    print(json.dumps({"state":"SUCCEEDED",**summary},sort_keys=True)); return 0
if __name__=="__main__": raise SystemExit(main())

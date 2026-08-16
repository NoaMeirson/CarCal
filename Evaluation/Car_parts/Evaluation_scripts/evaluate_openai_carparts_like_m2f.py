#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate precomputed OpenAI car-part segmentation JSONL with M2F-compatible outputs.

Does NOT call OpenAI. It reads the saved JSONL predictions and evaluates them against
exactly the same image folder + YOLO-segmentation TXT ground truth used for M2F.

Required: pip install pycocotools opencv-python numpy pandas matplotlib pillow
"""
from __future__ import annotations

import argparse, html, json, os, time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# ============================================================
# USER CONFIGURATION — EDIT THESE FOUR PATHS
# ============================================================
TEST_IMAGES_DIR = r"C:\Users\cs513\Desktop\Evaluation\Car_parts\Test_set\images"
GT_REFERENCES_DIR = r"C:\Users\cs513\Desktop\Evaluation\Car_parts\Test_set\labels"
OPENAI_RESULTS_JSONL = r"C:\Users\cs513\Desktop\Evaluation\Car_parts\chatGPT_results_on_test_set_M2F.jsonl"
OUTPUT_DIR = r"C:\Users\cs513\Desktop\Evaluation\Car_parts\Evaluation_results\ChatGPT"

# Keep aligned with the M2F evaluator.
PREDICTION_CONF = 0.001
VISUAL_CONF = 0.25
MATCH_MASK_IOU = 0.50
MAX_DETECTIONS = 100
MAX_VISUALS = 60  # 0 = all

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

CLASS_NAMES = [
    "back_bumper", "back_door", "back_glass", "back_left_door", "back_left_light",
    "back_light", "back_right_door", "back_right_light", "front_bumper", "front_door",
    "front_glass", "front_left_door", "front_left_light", "front_light",
    "front_right_door", "front_right_light", "hood", "left_mirror", "object",
    "right_mirror", "tailgate", "trunk", "wheel",
]
CLASS_TO_INDEX = {n: i for i, n in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)


@dataclass
class Instance:
    cls: int
    score: float
    mask: np.ndarray
    bbox_xyxy: Tuple[float, float, float, float]


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True); return p


def safe_div(a, b):
    return float(a / b) if b else 0.0


def mask_iou(a, b):
    inter = np.logical_and(a, b).sum(dtype=np.float64)
    union = np.logical_or(a, b).sum(dtype=np.float64)
    return float(inter / union) if union else 0.0


def mask_dice(a, b):
    inter = np.logical_and(a, b).sum(dtype=np.float64)
    den = a.sum(dtype=np.float64) + b.sum(dtype=np.float64)
    return float(2 * inter / den) if den else 0.0


def bbox_from_mask(mask):
    ys, xs = np.where(mask)
    if not len(xs): return (0., 0., 0., 0.)
    return (float(xs.min()), float(ys.min()), float(xs.max()+1), float(ys.max()+1))


def encode_mask(mask):
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    if isinstance(rle["counts"], bytes): rle["counts"] = rle["counts"].decode("ascii")
    return rle


def find_label_file(image_path: Path, labels_dir: Path):
    p = labels_dir / f"{image_path.stem}.txt"
    return p if p.exists() else None


def load_gt(label_path: Optional[Path], w: int, h: int) -> List[Instance]:
    out = []
    if label_path is None: return out
    for line_no, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        parts = line.strip().split()
        if not parts: continue
        if len(parts) < 7:
            print(f"WARNING invalid GT polygon: {label_path.name}:{line_no}"); continue
        try:
            cls = int(float(parts[0])); vals = list(map(float, parts[1:]))
        except ValueError:
            continue
        if not (0 <= cls < NUM_CLASSES) or len(vals) % 2: continue
        pts = []
        for i in range(0, len(vals), 2):
            x = max(0, min(w-1, int(round(vals[i] * w))))
            y = max(0, min(h-1, int(round(vals[i+1] * h))))
            pts.append([x, y])
        if len(pts) < 3: continue
        m = np.zeros((h, w), np.uint8)
        cv2.fillPoly(m, [np.asarray(pts, np.int32)], 1)
        m = m.astype(bool)
        if m.any(): out.append(Instance(cls, 1.0, m, bbox_from_mask(m)))
    return out


def load_jsonl(path: Path):
    by_name, issues = {}, []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line: continue
            try: obj = json.loads(line)
            except Exception as e:
                issues.append({"image":"", "detection_index":"", "reason":f"JSON error line {line_no}: {e}", "label":"", "confidence":""}); continue
            response = obj.get("response") or {}
            name = obj.get("fileName") or response.get("FileName")
            if not name:
                issues.append({"image":"", "detection_index":"", "reason":f"Missing fileName line {line_no}", "label":"", "confidence":""}); continue
            name = Path(str(name)).name
            if name in by_name:
                issues.append({"image":name, "detection_index":"", "reason":f"Duplicate record; last wins (line {line_no})", "label":"", "confidence":""})
            by_name[name] = obj
    return by_name, issues


def get_points(det):
    poly = det.get("polygon")
    if isinstance(poly, dict): return poly.get("points")
    if isinstance(poly, list): return poly
    return None


def record_to_predictions(record, w, h, image_name, issues):
    if record is None:
        issues.append({"image":image_name, "detection_index":"", "reason":"Missing JSONL record; zero predictions used", "label":"", "confidence":""})
        return []
    response = record.get("response") or {}
    if response.get("status") not in (None, "ok"):
        issues.append({"image":image_name, "detection_index":"", "reason":f"status={response.get('status')}; zero predictions used", "label":"", "confidence":""})
        return []
    dets = response.get("detections") or []
    if not isinstance(dets, list): return []
    ri = response.get("image") or {}
    sw = float(ri.get("width") or w); sh = float(ri.get("height") or h)
    sx = w / sw if sw > 0 else 1.0; sy = h / sh if sh > 0 else 1.0
    out = []
    for j, d in enumerate(dets):
        label = str(d.get("label", "")).strip()
        try: score = float(d.get("confidence", 0.0))
        except Exception:
            issues.append({"image":image_name,"detection_index":j,"reason":"Invalid confidence","label":label,"confidence":d.get("confidence")}); continue
        if label not in CLASS_TO_INDEX:
            issues.append({"image":image_name,"detection_index":j,"reason":"Unknown class label","label":label,"confidence":score}); continue
        points = get_points(d)
        if not points or len(points) < 3:
            issues.append({"image":image_name,"detection_index":j,"reason":"Polygon missing/<3 points","label":label,"confidence":score}); continue
        pts = []
        try:
            for p in points:
                x = max(0., min(w-1., float(p["x"]) * sx)); y = max(0., min(h-1., float(p["y"]) * sy))
                pts.append([int(round(x)), int(round(y))])
        except Exception:
            issues.append({"image":image_name,"detection_index":j,"reason":"Invalid polygon points","label":label,"confidence":score}); continue
        m = np.zeros((h, w), np.uint8); cv2.fillPoly(m, [np.asarray(pts, np.int32)], 1); m = m.astype(bool)
        if not m.any(): continue
        out.append(Instance(CLASS_TO_INDEX[label], score, m, bbox_from_mask(m)))
    out.sort(key=lambda z: z.score, reverse=True)
    return out[:MAX_DETECTIONS]


def greedy_all(gt, pred, thr):
    cand = [(mask_iou(g.mask,p.mask),gi,pi) for gi,g in enumerate(gt) for pi,p in enumerate(pred) if mask_iou(g.mask,p.mask) >= thr]
    cand.sort(reverse=True); ug, up, matches = set(), set(), []
    for iou,gi,pi in cand:
        if gi in ug or pi in up: continue
        ug.add(gi); up.add(pi); matches.append((gi,pi,iou))
    return matches, sorted(set(range(len(gt)))-ug), sorted(set(range(len(pred)))-up)


def greedy_class(gt, pred, cls, thr):
    gis=[i for i,x in enumerate(gt) if x.cls==cls]; pis=[i for i,x in enumerate(pred) if x.cls==cls]
    cand=[]
    for gi in gis:
        for pi in pis:
            iou=mask_iou(gt[gi].mask,pred[pi].mask)
            if iou>=thr: cand.append((iou,gi,pi))
    cand.sort(reverse=True); ug,up,matches=set(),set(),[]
    for iou,gi,pi in cand:
        if gi in ug or pi in up: continue
        ug.add(gi);up.add(pi);matches.append((gi,pi,iou))
    return matches,[i for i in gis if i not in ug],[i for i in pis if i not in up]


def build_coco_gt(image_paths, gt_by_name):
    ds={"info":{"description":"Car parts GT converted from YOLO TXT"},"licenses":[],"images":[],"annotations":[],"categories":[{"id":i+1,"name":n,"supercategory":"car_part"} for i,n in enumerate(CLASS_NAMES)]}
    name2id={}; ann_id=1
    for image_id,p in enumerate(image_paths,1):
        with Image.open(p) as im: w,h=im.size
        name2id[p.name]=image_id; ds["images"].append({"id":image_id,"file_name":p.name,"width":w,"height":h})
        for inst in gt_by_name[p.name]:
            x1,y1,x2,y2=inst.bbox_xyxy
            ds["annotations"].append({"id":ann_id,"image_id":image_id,"category_id":inst.cls+1,"segmentation":encode_mask(inst.mask),"area":int(inst.mask.sum()),"bbox":[x1,y1,x2-x1,y2-y1],"iscrowd":0}); ann_id+=1
    return ds,name2id


def build_coco_pred(pred_by_name,name2id):
    out=[]
    for name,preds in pred_by_name.items():
        if name not in name2id: continue
        for inst in preds:
            if inst.score < PREDICTION_CONF: continue
            x1,y1,x2,y2=inst.bbox_xyxy
            out.append({"image_id":name2id[name],"category_id":inst.cls+1,"segmentation":encode_mask(inst.mask),"score":float(inst.score),"bbox":[x1,y1,x2-x1,y2-y1]})
    return out


def coco_eval(gt_path,preds):
    gt=COCO(str(gt_path))
    dt=gt.loadRes(preds) if preds else COCO()
    if not preds:
        dt.dataset={"images":gt.dataset["images"],"categories":gt.dataset["categories"],"annotations":[]}; dt.createIndex()
    ev=COCOeval(gt,dt,"segm"); ev.params.maxDets=[1,10,MAX_DETECTIONS]; ev.evaluate();ev.accumulate();ev.summarize()
    s=ev.stats.tolist()
    overall=dict(zip(["segm_AP50_95","segm_AP50","segm_AP75","segm_AP_small","segm_AP_medium","segm_AP_large","segm_AR_1","segm_AR_10","segm_AR_100","segm_AR_small","segm_AR_medium","segm_AR_large"],map(float,s)))
    pr=ev.eval["precision"]; rc=ev.eval["recall"]; th=ev.params.iouThrs; i50=int(np.argmin(abs(th-.5)));i75=int(np.argmin(abs(th-.75)))
    rows=[]
    for k,n in enumerate(CLASS_NAMES):
        a=pr[:,:,k,0,-1];b=pr[i50,:,k,0,-1];c=pr[i75,:,k,0,-1];d=rc[:,k,0,-1]
        va=a[a>-1];vb=b[b>-1];vc=c[c>-1];vd=d[d>-1]
        rows.append({"class_id":k,"class_name":n,"AP50_95":float(va.mean()) if va.size else np.nan,"AP50":float(vb.mean()) if vb.size else np.nan,"AP75":float(vc.mean()) if vc.size else np.nan,"AR50_95":float(vd.mean()) if vd.size else np.nan})
    return overall,pd.DataFrame(rows)


def color_for_class(i):
    pal=[(230,126,34),(52,152,219),(155,89,182),(46,204,113),(241,196,15),(231,76,60),(26,188,156),(149,165,166),(52,73,94)]
    return pal[i%len(pal)]


def overlay(image,items,alpha=.42):
    out=image.copy();layer=image.copy()
    for z in items: layer[z.mask]=color_for_class(z.cls)
    out=cv2.addWeighted(layer,alpha,out,1-alpha,0)
    for z in items:
        color=color_for_class(z.cls); cnts,_=cv2.findContours(z.mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(out,cnts,-1,color,2)
        if cnts:
            x,y,_,_=cv2.boundingRect(max(cnts,key=cv2.contourArea)); text=CLASS_NAMES[z.cls] + (f" {z.score:.2f}" if z.score<.999 else "")
            cv2.putText(out,text,(x,max(18,y-5)),cv2.FONT_HERSHEY_SIMPLEX,.52,(255,255,255),3,cv2.LINE_AA);cv2.putText(out,text,(x,max(18,y-5)),cv2.FONT_HERSHEY_SIMPLEX,.52,color,1,cv2.LINE_AA)
    return out


def error_img(gt,pred,shape):
    h,w=shape;g=np.zeros((h,w),bool);p=np.zeros((h,w),bool)
    for z in gt:g|=z.mask
    for z in pred:p|=z.mask
    out=np.full((h,w,3),28,np.uint8);out[g&p]=(60,170,60);out[g&~p]=(45,45,230);out[p&~g]=(0,150,255);return out


def title(img,text):
    bar=np.full((44,img.shape[1],3),250,np.uint8);cv2.putText(bar,text,(12,29),cv2.FONT_HERSHEY_SIMPLEX,.7,(35,35,35),2,cv2.LINE_AA);return np.vstack([bar,img])


def resize(img,H=420):
    h,w=img.shape[:2];return cv2.resize(img,(max(1,int(w*H/max(h,1))),H),interpolation=cv2.INTER_AREA)


def comparison(image,gt,pred):
    panels=[title(resize(image),"Original"),title(resize(overlay(image,gt)),"Ground Truth overlay"),title(resize(overlay(image,pred)),"Prediction overlay"),title(resize(error_img(gt,pred,image.shape[:2])),"Pixel errors: Green=correct, Red=missed, Orange=false")]
    return cv2.hconcat(panels)


def save_cm(M,labels,path,normalize=False):
    D=M.astype(float)
    if normalize:
        den=D.sum(1,keepdims=True);D=np.divide(D,den,out=np.zeros_like(D),where=den!=0)
    fw=max(10,len(labels)*.72);fig,ax=plt.subplots(figsize=(fw,fw*.92));im=ax.imshow(D,cmap="Blues");ax.set_xticks(range(len(labels)));ax.set_yticks(range(len(labels)));ax.set_xticklabels(labels,rotation=55,ha="right",fontsize=7);ax.set_yticklabels(labels,fontsize=7);ax.set_xlabel("Predicted");ax.set_ylabel("Ground truth");ax.set_title("Normalized confusion matrix" if normalize else "Confusion matrix");fig.colorbar(im,ax=ax,fraction=.046,pad=.04);fig.tight_layout();fig.savefig(path,dpi=180);plt.close(fig)


def save_chart(df,metric,path):
    if metric not in df:return
    v=pd.to_numeric(df[metric],errors="coerce").fillna(0).to_numpy();labels=df["class_name"].tolist();fig,ax=plt.subplots(figsize=(max(10,len(labels)*.65),5.2));ax.bar(np.arange(len(labels)),v);ax.set_xticks(np.arange(len(labels)));ax.set_xticklabels(labels,rotation=45,ha="right",fontsize=8);ax.set_ylim(0,1.05);ax.set_ylabel(metric);ax.set_title(f"Per-class {metric}");ax.grid(axis="y",alpha=.25);fig.tight_layout();fig.savefig(path,dpi=180);plt.close(fig)


def table_html(df):
    d=df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]): d[c]=d[c].map(lambda x:"—" if pd.isna(x) else f"{x:.4f}")
    return d.to_html(index=False,classes="data",border=0,escape=True)


def generate_reports(out_dir,jsonl,images,labels,overall,per_class,pixel,dataset,plots,visual_rows,elapsed):
    row=overall.iloc[0].to_dict();cards=[]
    for k in ["segm_AP50_95","segm_AP50","segm_AP75","instance_precision","instance_recall","instance_F1","pixel_mIoU","pixel_mDice"]:
        v=row.get(k,np.nan);cards.append(f"<div class='card'><span>{k}</span><strong>{'—' if pd.isna(v) else f'{float(v):.3f}'}</strong></div>")
    rel=lambda p:Path(os.path.relpath(p,out_dir)).as_posix(); plots_html="".join(f"<figure><img src='{rel(p)}'><figcaption>{p.stem}</figcaption></figure>" for p in plots if p.exists())
    css="""*{box-sizing:border-box}body{font-family:Segoe UI,Arial,sans-serif;margin:0;background:#f4f6f8;color:#18212b}header{background:#17212b;color:white;padding:24px 34px}header h1{margin:0 0 7px;font-size:24px}header p{margin:3px 0;color:#c8d1da;font-size:12px}main{padding:24px 32px;max-width:1500px;margin:auto}section{background:white;border:1px solid #e1e6eb;border-radius:12px;padding:20px;margin-bottom:18px}h2{font-size:17px;margin:0 0 14px;border-bottom:1px solid #e5e7eb;padding-bottom:9px}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(145px,1fr));gap:10px}.card{background:#f8fafc;border:1px solid #e2e8f0;border-radius:9px;padding:13px}.card span{display:block;font-size:11px;color:#667085}.card strong{font-size:23px}table.data{border-collapse:collapse;width:100%;font-size:12px}table.data th{background:#eef2f6;text-align:left;padding:8px;border:1px solid #dce2e8}table.data td{padding:8px;border:1px solid #e2e7ec}.plots{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:14px}.plots figure{margin:0;border:1px solid #e1e6eb;border-radius:9px;padding:8px}.plots img{width:100%;display:block}a.button{display:inline-block;background:#2563eb;color:white;text-decoration:none;padding:10px 15px;border-radius:8px;font-weight:600}.meta{font-size:12px;color:#596574}.visual{background:white;border:1px solid #dfe5eb;border-radius:12px;padding:12px;margin-bottom:16px}.visual img{width:100%;display:block}"""
    (out_dir/"report.html").write_text(f"""<!doctype html><html><head><meta charset='utf-8'><title>OpenAI Car Parts Segmentation Evaluation</title><style>{css}</style></head><body><header><h1>OpenAI Car Parts Segmentation Evaluation</h1><p>Predictions: {html.escape(str(jsonl))}</p><p>Images: {html.escape(str(images))}</p><p>GT labels: {html.escape(str(labels))}</p></header><main><section><h2>Overall metrics</h2><div class='cards'>{''.join(cards)}</div><p class='meta'>Official pycocotools COCOeval for segmentation AP/AR. OpenAI polygons are rasterized to masks; YOLO GT is converted to COCO for evaluation only.</p></section><section><h2>Dataset</h2>{table_html(dataset)}</section><section><h2>COCO + instance metrics by class</h2>{table_html(per_class)}</section><section><h2>Pixel metrics by class</h2>{table_html(pixel)}</section><section><h2>Charts</h2><div class='plots'>{plots_html}</div></section><section><h2>Visual comparisons</h2><a class='button' href='visual_comparisons.html'>Open Original / GT / Prediction / Errors</a><p class='meta'>{len(visual_rows)} examples · evaluation runtime {elapsed:.1f}s</p></section></main></body></html>""",encoding="utf-8")
    blocks="".join(f"<div class='visual'><h3>{html.escape(n)}</h3><div>Union mask IoU: {q:.3f}</div><img src='{rel(p)}'></div>" for n,p,q in visual_rows)
    (out_dir/"visual_comparisons.html").write_text(f"<!doctype html><html><head><meta charset='utf-8'><title>Visual comparisons</title><style>{css}</style></head><body><header><h1>Visual comparisons</h1><p>Original · Ground Truth · Prediction · Pixel errors</p></header><main><p><a class='button' href='report.html'>Back to metrics</a></p>{blocks}</main></body></html>",encoding="utf-8")


def parse_args():
    p=argparse.ArgumentParser();p.add_argument("--images",default=TEST_IMAGES_DIR);p.add_argument("--labels",default=GT_REFERENCES_DIR);p.add_argument("--predictions",default=OPENAI_RESULTS_JSONL);p.add_argument("--out",default=OUTPUT_DIR);p.add_argument("--pred-conf",type=float,default=PREDICTION_CONF);p.add_argument("--visual-conf",type=float,default=VISUAL_CONF);p.add_argument("--match-iou",type=float,default=MATCH_MASK_IOU);p.add_argument("--max-detections",type=int,default=MAX_DETECTIONS);p.add_argument("--max-visuals",type=int,default=MAX_VISUALS);return p.parse_args()


def main():
    global PREDICTION_CONF,VISUAL_CONF,MATCH_MASK_IOU,MAX_DETECTIONS
    a=parse_args();PREDICTION_CONF=a.pred_conf;VISUAL_CONF=a.visual_conf;MATCH_MASK_IOU=a.match_iou;MAX_DETECTIONS=a.max_detections
    images=Path(a.images);labels=Path(a.labels);jsonl=Path(a.predictions);out=Path(a.out);plots=ensure_dir(out/"plots");visuals=ensure_dir(out/"visuals")
    if not images.is_dir():raise FileNotFoundError(images)
    if not labels.is_dir():raise FileNotFoundError(labels)
    if not jsonl.is_file():raise FileNotFoundError(jsonl)
    image_paths=sorted(p for p in images.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not image_paths:raise RuntimeError("No test images")
    print("[1/7] Loading OpenAI JSONL...");records,issues=load_jsonl(jsonl);print(" records:",len(records)," test images:",len(image_paths));start=time.time()
    bg=NUM_CLASSES;cm=np.zeros((NUM_CLASSES+1,NUM_CLASSES+1),np.int64);ic=[{"TP":0,"FP":0,"FN":0,"ious":[],"dices":[]} for _ in range(NUM_CLASSES)];pc=[{"TP":0,"FP":0,"FN":0,"TN":0} for _ in range(NUM_CLASSES)];gt_by={};pred_by={};vis=[];total_gt=0;ok=0;models=set();prompts=set()
    print("[2/7] Converting predictions + instance/pixel evaluation...")
    for idx,p in enumerate(image_paths,1):
        with Image.open(p) as im:w,h=im.size; bgr=cv2.cvtColor(np.array(im.convert("RGB")),cv2.COLOR_RGB2BGR)
        gt=load_gt(find_label_file(p,labels),w,h);rec=records.get(p.name)
        if rec:
            rr=rec.get("response") or {};ok+=int(rr.get("status") in (None,"ok"));meta=rec.get("meta") or {}
            if meta.get("model"):models.add(str(meta["model"]))
            if meta.get("promptVersion"):prompts.add(str(meta["promptVersion"]))
        raw=record_to_predictions(rec,w,h,p.name,issues);pred_all=[z for z in raw if z.score>=PREDICTION_CONF][:MAX_DETECTIONS];pred=[z for z in pred_all if z.score>=VISUAL_CONF];gt_by[p.name]=gt;pred_by[p.name]=pred_all;total_gt+=len(gt)
        m,ug,up=greedy_all(gt,pred,MATCH_MASK_IOU)
        for gi,pi,_ in m:cm[gt[gi].cls,pred[pi].cls]+=1
        for gi in ug:cm[gt[gi].cls,bg]+=1
        for pi in up:cm[bg,pred[pi].cls]+=1
        for c in range(NUM_CLASSES):
            mm,gmiss,pmiss=greedy_class(gt,pred,c,MATCH_MASK_IOU);ic[c]["TP"]+=len(mm);ic[c]["FN"]+=len(gmiss);ic[c]["FP"]+=len(pmiss)
            for gi,pi,iou in mm:ic[c]["ious"].append(iou);ic[c]["dices"].append(mask_dice(gt[gi].mask,pred[pi].mask))
            gu=np.zeros((h,w),bool);pu=np.zeros((h,w),bool)
            for z in gt:
                if z.cls==c:gu|=z.mask
            for z in pred:
                if z.cls==c:pu|=z.mask
            pc[c]["TP"]+=int((gu&pu).sum());pc[c]["FP"]+=int((~gu&pu).sum());pc[c]["FN"]+=int((gu&~pu).sum());pc[c]["TN"]+=int((~gu&~pu).sum())
        gu=np.zeros((h,w),bool);pu=np.zeros((h,w),bool)
        for z in gt:gu|=z.mask
        for z in pred:pu|=z.mask
        vis.append((mask_iou(gu,pu),p.name,bgr,gt,pred))
        if idx%25==0 or idx==len(image_paths):print(f" {idx}/{len(image_paths)}")
    print("[3/7] Official COCOeval...");gt_dict,name2id=build_coco_gt(image_paths,gt_by);gt_path=out/"coco_ground_truth.json";gt_path.write_text(json.dumps(gt_dict),encoding="utf-8");preds=build_coco_pred(pred_by,name2id);(out/"coco_predictions.json").write_text(json.dumps(preds),encoding="utf-8");coco_overall,ap_df=coco_eval(gt_path,preds)
    print("[4/7] Writing tables...");ir=[]
    for c,n in enumerate(CLASS_NAMES):
        z=ic[c];tp,fp,fn=z["TP"],z["FP"],z["FN"];pr=safe_div(tp,tp+fp);rc=safe_div(tp,tp+fn);f1=safe_div(2*pr*rc,pr+rc);ir.append({"class_id":c,"class_name":n,"GT_instances":tp+fn,"Pred_instances":tp+fp,"TP":tp,"FP":fp,"FN":fn,"Precision@matchIoU":pr,"Recall@matchIoU":rc,"F1@matchIoU":f1,"Mean_matched_mask_IoU":float(np.mean(z["ious"])) if z["ious"] else 0.,"Mean_matched_mask_Dice":float(np.mean(z["dices"])) if z["dices"] else 0.})
    instance=pd.DataFrame(ir);per_class=ap_df.merge(instance,on=["class_id","class_name"],how="outer");pr=[]
    for c,n in enumerate(CLASS_NAMES):
        z=pc[c];tp,fp,fn,tn=z["TP"],z["FP"],z["FN"],z["TN"];pr.append({"class_id":c,"class_name":n,"Pixel_Precision":safe_div(tp,tp+fp),"Pixel_Recall":safe_div(tp,tp+fn),"Pixel_IoU":safe_div(tp,tp+fp+fn),"Pixel_Dice":safe_div(2*tp,2*tp+fp+fn),"Pixel_Accuracy":safe_div(tp+tn,tp+tn+fp+fn),"TP_pixels":tp,"FP_pixels":fp,"FN_pixels":fn})
    pixel=pd.DataFrame(pr);T=int(instance.TP.sum());F=int(instance.FP.sum());N=int(instance.FN.sum());ip=safe_div(T,T+F);ire=safe_div(T,T+N);if1=safe_div(2*ip*ire,ip+ire);elapsed=time.time()-start;overall=dict(coco_overall);overall.update({"instance_precision":ip,"instance_recall":ire,"instance_F1":if1,"pixel_mIoU":float(pixel.Pixel_IoU.mean()),"pixel_mDice":float(pixel.Pixel_Dice.mean()),"prediction_conf":float(PREDICTION_CONF),"visual_conf":float(VISUAL_CONF),"mask_threshold":np.nan,"match_mask_iou":float(MATCH_MASK_IOU),"max_detections":int(MAX_DETECTIONS),"images":len(image_paths),"GT_instances":total_gt,"predictions_COCO":len(preds),"runtime_seconds":elapsed,"seconds_per_image":elapsed/len(image_paths),"metric_method":"Official pycocotools COCOeval segm; OpenAI polygons rasterized to masks; GT=YOLO segmentation TXT converted to COCO","prediction_source":"OpenAI JSONL","source_models":", ".join(sorted(models)),"prompt_versions":", ".join(sorted(prompts)),"jsonl_records":len(records),"jsonl_ok_test_images":ok,"invalid_or_missing_prediction_rows":len(issues)})
    overall_df=pd.DataFrame([overall]);dataset=pd.DataFrame([{"images":len(image_paths),"annotations":total_gt,"classes":NUM_CLASSES,"class_names":", ".join(CLASS_NAMES),"model_classes":", ".join(CLASS_NAMES),"jsonl_records":len(records),"jsonl_ok_test_images":ok}]);overall_df.to_csv(out/"metrics_overall.csv",index=False);per_class.to_csv(out/"metrics_per_class.csv",index=False);pixel.to_csv(out/"pixel_metrics_per_class.csv",index=False);names=CLASS_NAMES+["background"];pd.DataFrame(cm,index=names,columns=names).to_csv(out/"confusion_matrix.csv");pd.DataFrame(issues).to_csv(out/"invalid_predictions.csv",index=False)
    print("[5/7] Plots...");cm1=plots/"confusion_matrix.png";cm2=plots/"confusion_matrix_normalized.png";save_cm(cm,names,cm1);save_cm(cm,names,cm2,True);chart=[cm1,cm2]
    for metric in ["AP50_95","AP50","AP75","Precision@matchIoU","Recall@matchIoU","F1@matchIoU","Mean_matched_mask_IoU"]:
        q=plots/f"per_class_{metric.replace('@','_')}.png";save_chart(per_class,metric,q);chart.append(q)
    for metric in ["Pixel_IoU","Pixel_Dice"]:
        q=plots/f"per_class_{metric}.png";save_chart(pixel,metric,q);chart.append(q)
    print("[6/7] Visuals...");vis.sort(key=lambda x:x[0]);sel=vis if a.max_visuals==0 else vis[:a.max_visuals];vr=[]
    for quality,name,bgr,gt,pred in sel:
        q=visuals/f"{Path(name).stem}_comparison.jpg";cv2.imwrite(str(q),comparison(bgr,gt,pred));vr.append((name,q,quality))
    generate_reports(out,jsonl,images,labels,overall_df,per_class,pixel,dataset,chart,vr,elapsed);print("[7/7] Done:",out);return 0

if __name__=="__main__":raise SystemExit(main())

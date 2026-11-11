import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BG="#fcfcfb"; INK="#0b0b0b"; INK2="#52514e"; MUTED="#a3a29c"; GRID="#e6e5e0"
BLUE="#2a78d6"; ORANGE="#eb6834"; AQUA="#1baf7a"; RED="#e34948"
plt.rcParams.update({"font.family":"DejaVu Sans","text.color":INK,"axes.edgecolor":MUTED,
    "axes.labelcolor":INK2,"xtick.color":INK2,"ytick.color":INK2,"figure.facecolor":BG,"axes.facecolor":BG})
def clean(ax):
    for s in ("top","right","left"): ax.spines[s].set_visible(False)
    ax.tick_params(length=0)

# 1. HERO --------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14,7)); ax.set_xlim(0,14); ax.set_ylim(0,7); ax.axis("off")
ax.text(0.7,6.0,"Same 7B model. Four sizes.",fontsize=34,weight="bold")
ax.text(0.7,5.3,"FP32  →  FP16  →  INT8  →  INT4   ·   what you keep, what you lose",fontsize=17,color=INK2)
fmts=[("FP32",32,MUTED),("FP16",16,BLUE),("INT8 (Q8_0)",8.5,AQUA),("INT4 (Q4_0)",4.5,ORANGE)]
y=4.2
for name,bits,c in fmts:
    gb=7e9*bits/8/1e9; w=gb/28*11.5
    ax.add_patch(FancyBboxPatch((0.7,y-0.32),w,0.64,boxstyle="round,pad=0,rounding_size=0.08",fc=c,ec="none"))
    ax.text(0.7+w+0.2,y,f"{name}   {gb:.1f} GB",va="center",fontsize=16,weight="bold" if bits<5 else None)
    y-=0.95
ax.plot([0.7+4/28*11.5]*2,[0.15,4.7],ls="--",color=RED,lw=2)
ax.text(0.7+4/28*11.5+0.1,0.25,"my RTX 3050: 4 GB VRAM",color=RED,fontsize=13)
fig.savefig("img/01_hero.png",dpi=110,bbox_inches="tight",pad_inches=0.3); plt.close()

# 2. ROUNDING ----------------------------------------------------------
fig, ax = plt.subplots(figsize=(13,3.6)); ax.axis("off")
levels=np.arange(-7,8)
ax.hlines(0,-7.6,7.6,color=MUTED,lw=1.5)
for l in levels:
    ax.vlines(l,-0.15,0.15,color=INK2,lw=1.5); ax.text(l,-0.42,str(l),ha="center",fontsize=11,color=INK2)
vals=[-5.62,-1.31,0.44,2.79,6.18]
for v in vals:
    q=round(v); ax.plot(v,0.95,"o",ms=10,color=BLUE); ax.text(v,1.25,f"{v}",ha="center",fontsize=12,color=INK)
    ax.annotate("",xy=(q,0.2),xytext=(v,0.85),arrowprops=dict(arrowstyle="->",color=ORANGE,lw=1.8))
    ax.plot(q,0,"o",ms=9,color=ORANGE)
ax.text(-7.6,1.85,"Original float values (blue) snap to the nearest of only 15 INT4 levels (orange)",fontsize=14,weight="bold")
ax.text(-7.6,-0.95,"The gap between blue and orange = quantization error. Fewer bits → fewer levels → bigger jumps.",fontsize=12,color=INK2)
ax.set_xlim(-7.8,7.8); ax.set_ylim(-1.1,2.1)
fig.savefig("img/02_rounding.png",dpi=110,bbox_inches="tight",pad_inches=0.25); plt.close()

# 3. OUTLIER: per-tensor vs block-wise ---------------------------------
rng=np.random.default_rng(7)
w=rng.normal(0,0.09,16).round(3); w[11]=0.9  # one outlier
def q4(x):
    s=np.abs(x).max()/7; return np.clip(np.round(x/s),-7,7)*s
pt=q4(w); bw=np.concatenate([q4(w[:8]),q4(w[8:])])
fig, axes = plt.subplots(3,1,figsize=(13,6.4),sharex=True)
for ax,data,title,c in [(axes[0],w,"Original weights (one outlier at position 11)",MUTED),
                        (axes[1],pt,"INT4, ONE scale for everything → the outlier sets the scale, small weights get crushed (most turn into 0)",RED),
                        (axes[2],bw,"INT4, one scale PER BLOCK (what GGUF does) → only the outlier's block suffers",BLUE)]:
    ax.bar(range(16),data,color=c,width=0.7); ax.axhline(0,color=MUTED,lw=1)
    ax.set_ylim(-0.25,1.0); clean(ax); ax.set_yticks([])
    ax.set_title(title,loc="left",fontsize=12.5,color=INK,weight="bold")
axes[2].axvspan(-0.5,7.5,color=AQUA,alpha=0.07); axes[2].axvspan(7.5,15.5,color=ORANGE,alpha=0.07)
axes[2].text(3.5,0.8,"block 1",ha="center",color=INK2); axes[2].text(13.5,0.8,"block 2",ha="center",color=INK2)
axes[2].set_xticks(range(16))
fig.tight_layout(); fig.savefig("img/03_outlier.png",dpi=110,bbox_inches="tight",pad_inches=0.25); plt.close()

# 4. ERROR CHART (real numbers) ----------------------------------------
r=json.load(open("quant_error_results.json"))["with_outliers"]
rows=[("FP16","16",False),("GGUF Q8_0","8.5",True),("INT8 per-channel","8",False),("INT8 per-tensor","8",False),
      ("GGUF Q5_0","5.5",True),("GGUF Q4_0","4.5",True),("INT4 per-channel","4",False),("INT4 per-tensor","4",False)]
fig, ax = plt.subplots(figsize=(12,6))
ys=np.arange(len(rows))[::-1]
for y,(k,bits,g) in zip(ys,rows):
    v=r[k]*100; ax.barh(y,v,color=BLUE if g else ORANGE,height=0.62)
    ax.text(v+1,y,f"{v:.1f}%",va="center",fontsize=12,color=INK)
ax.set_yticks(ys,[f"{k}  ({b} bits)" for k,b,_ in rows],fontsize=12)
ax.set_xlabel("Relative error vs original weights (%) — lower is better",fontsize=12)
ax.set_xlim(0,92); clean(ax); ax.xaxis.grid(True,color=GRID); ax.set_axisbelow(True)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=BLUE,label="GGUF block-wise (32 weights per scale)"),Patch(color=ORANGE,label="Naive / single scale")],
          loc="center right",bbox_to_anchor=(1.0,0.62),frameon=False,fontsize=12)
ax.set_title("Same 1024×4096 weight matrix, 0.1% outliers — measured, not guessed",loc="left",fontsize=15,weight="bold",pad=14)
fig.tight_layout(); fig.savefig("img/04_error.png",dpi=110,bbox_inches="tight",pad_inches=0.25); plt.close()

# 5. SIZE CHART ----------------------------------------------------------
fmts=[("FP32",32),("FP16",16),("Q8_0",8.5),("Q5_0",5.5),("Q4_0",4.5)]
fig, axes = plt.subplots(1,3,figsize=(14,5),sharey=True)
for ax,(lab,p) in zip(axes,[("1.5B params",1.5e9),("3B params",3e9),("7B params",7e9)]):
    gbs=[p*b/8/1e9 for _,b in fmts]
    cols=[BLUE if g<=3.5 else MUTED for g in gbs]
    ax.bar(range(5),gbs,color=cols,width=0.65)
    for i,g in enumerate(gbs): ax.text(i,g+0.4,f"{g:.1f}",ha="center",fontsize=11)
    ax.axhline(4,color=RED,ls="--",lw=1.8); ax.set_xticks(range(5),[f for f,_ in fmts],fontsize=11)
    ax.set_title(lab,fontsize=14,weight="bold"); clean(ax); ax.set_yticks([])
axes[0].text(4.45,4.3,"4 GB VRAM",color=RED,fontsize=11,ha="right")
fig.suptitle("Weights-only size in GB — blue = fits a 4 GB card with ~0.5 GB left for KV cache",x=0.02,ha="left",fontsize=15,weight="bold")
fig.tight_layout(); fig.savefig("img/05_size.png",dpi=110,bbox_inches="tight",pad_inches=0.25); plt.close()

# 6. CHEAT SHEET ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(16,6.2)); ax.axis("off"); ax.set_xlim(0,16); ax.set_ylim(0,6.2)
ax.text(0.2,5.75,"Which format should I pick?",fontsize=24,weight="bold")
rows=[("You want…","Pick","Why"),
      ("Run a chat LLM locally (Ollama / llama.cpp / LM Studio)","GGUF Q4_K_M","Best size-vs-quality default; runs on CPU + small GPUs"),
      ("Max quality, you have the memory","GGUF Q8_0 / FP16","Q8_0 is close to FP16 at about half the size"),
      ("Deploy a model inside an app / server on CPU","ONNX INT8","Portable runtime, works from Python, C#, JS, mobile"),
      ("Squeeze a model onto tiny hardware","GGUF Q3/Q2 or INT4","Fits, but test quality — it drops fast below 4 bits"),
      ("Training / fine-tuning","Stay FP16 / BF16","Quantize AFTER, for inference")]
y=4.95
for i,(a,b,c) in enumerate(rows):
    if i==0:
        for x,t in [(0.2,a),(7.2,b),(10.2,c)]: ax.text(x,y,t,fontsize=12.5,color=INK2,weight="bold")
        ax.plot([0.2,15.8],[y-0.3]*2,color=MUTED,lw=1)
    else:
        ax.text(0.2,y,a,fontsize=12.5,va="center"); ax.text(7.2,y,b,fontsize=12.5,weight="bold",color=BLUE,va="center")
        ax.text(10.2,y,c,fontsize=11,color=INK2,va="center",wrap=True)
        ax.plot([0.2,15.8],[y-0.42]*2,color=GRID,lw=1)
    y-=0.85 if i else 0.75
fig.savefig("img/06_cheatsheet.png",dpi=110,bbox_inches="tight",pad_inches=0.3); plt.close()
print("done")

from pathlib import Path
import sys, os
import numpy as np, math
import rasterio, geopandas as gpd
import matplotlib as mpl, matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch, Rectangle, Patch, Wedge
from matplotlib.ticker import FuncFormatter

mpl.rcParams.update({
    "font.family": ["Arial", "DejaVu Sans"],"font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"], "mathtext.fontset": "custom", "mathtext.rm": "sans", "mathtext.it": "sans:italic", "mathtext.bf": "sans:bold",
    "svg.fonttype":"none","pdf.fonttype":42,"font.size":7,"axes.linewidth":0.6,
    "legend.frameon":False,"axes.unicode_minus":False,
    "xtick.major.width":0.6,"ytick.major.width":0.6,
    "xtick.major.size":2.0,"ytick.major.size":2.0,
})

FS = dict(
    map_tick   = 13,
    map_title  = 14,
    row_label  = 14,
    legend     = 14,
    scale      = 11,
    grade_leg  = 11,
    chord_pct  = 11,
    chord_grp  = 13,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_palette as PP

from config_1004 import PRED_MODE, AREA_DIR, CHINA_SHP, POLY_SHP, out_dir, fig_dir
ZONE   = out_dir("quality_zones") / PRED_MODE
WEIGHT = AREA_DIR
OUTFIG = fig_dir("fig4_migration"); OUTFIG.mkdir(parents=True, exist_ok=True)
SHP_CHINA    = CHINA_SHP
SHP_NINEDASH = POLY_SHP

RI    = ["ri1","ri2"]
RILAB = {"ri1":"Single/early rice","ri2":"Late rice"}
PERIODS = ["baseline","ssp126","ssp585"]
PLAB    = {"baseline":"Baseline","ssp126":"SSP1-2.6","ssp585":"SSP5-8.5"}
SANKEY_ORDER = ["ssp126","baseline","ssp585"]

GRADES = ["Premium","Moderate","Common"]
GCODE  = {"Premium":3,"Moderate":2,"Common":1}
GCOL   = dict(PP.GRADE)
CMAP = ListedColormap([GCOL["Common"],GCOL["Moderate"],GCOL["Premium"]])
NORM = BoundaryNorm([.5,1.5,2.5,3.5],3)
DCOL = dict(PP.DIR)
MAP_EXTENT   = [73,136,17,54]
MAP_ASPECT   = 1/np.cos(np.deg2rad(np.mean([MAP_EXTENT[2],MAP_EXTENT[3]])))
INSET_EXTENT = [105,125,2,26]
SCALE_KM = 1000; LAT_REF = 35.0
SANKEY_XS={"ssp126":0.05,"baseline":0.44,"ssp585":0.83}; SANKEY_WN=0.150
SANKEY_XLIM=(-0.02,1.02)

def read(p):
    p=Path(p)
    if not p.exists(): return None
    with rasterio.open(p) as ds:
        a=ds.read(1).astype("float64"); nd=ds.nodata
        if nd is not None: a[a==nd]=np.nan
        return a, ds.transform

def _shp(p):
    try:
        if not Path(p).exists(): return None
        g=gpd.read_file(p)
        if g.crs is not None and g.crs.to_epsg()!=4326: g=g.to_crs(4326)
        return g
    except Exception as e:
        print(f"  [warn] shp {Path(p).name}: {e}"); return None
GCHN=_shp(SHP_CHINA); GPOLY=_shp(SHP_NINEDASH)
if GPOLY is not None:
    GNANHAI=GPOLY[GPOLY.geometry.apply(lambda g:g.bounds[1]<25)]
    GJD=GPOLY[GPOLY["adchar"]=="JD"] if "adchar" in GPOLY.columns else GNANHAI
else:
    GNANHAI=GJD=None

def zone_path(ri, period):
    return ZONE/(f"zone_{ri}_baseline.tif" if period=="baseline"
                 else f"zone_{ri}_{period}_future.tif")

def load_zones(ri):
    r=read(WEIGHT/f"area_weight_{ri}_2000_2010.tif")
    if r is None: return None,None,None
    w,t=r; Z={}
    for p in PERIODS:
        z=read(zone_path(ri,p))
        if z is None: return None,None,None
        Z[p]=z[0]
    ok=np.isfinite(w)&(w>0)
    for z in Z.values(): ok &= np.isin(z,[1,2,3])
    return w[ok], {p:Z[p][ok].astype(int) for p in PERIODS}, t

def add_scale_bar(ax, x_left, cy, length_km=SCALE_KM, lat_ref=LAT_REF, n_seg=4, h=0.0040):
    fig=ax.figure
    deg=length_km/(111.320*np.cos(np.deg2rad(lat_ref)))
    x_lo,x_hi=ax.get_xlim(); pos=ax.get_position()
    w=deg/(x_hi-x_lo)*pos.width; seg=w/n_seg
    for i in range(n_seg):
        fig.patches.append(Rectangle((x_left+i*seg,cy),seg,h,
                           facecolor="black" if i%2==0 else "white",
                           edgecolor="black",linewidth=0.4,
                           transform=fig.transFigure,zorder=10,figure=fig))
    fig.text(x_left+w+0.006,cy+h/2,f"{length_km:,} km",
             ha="left",va="center",fontsize=FS["scale"],zorder=10)

def add_grade_legend(ax, fs, areas=None):
    """地图左上角竖排色块；[1106] 每幅地图标注三个等级的面积（Mha）。
    areas: {grade: Mha}；为 None 时只画色块 + 等级名（旧行为）。"""
    x0=0.03; sw=0.055; sh=0.055; ys={"Premium":0.90,"Moderate":0.82,"Common":0.74}
    shown=[g for g in GRADES if areas is None or areas[g]>=0.05]
    for k,g in enumerate(shown):
        y=list(ys.values())[k]
        ax.add_patch(Rectangle((x0,y),sw,sh,transform=ax.transAxes,
                     facecolor=GCOL[g],edgecolor="none",zorder=12,clip_on=False))
        if areas is None: lab = g
        else:
            lab=f"{areas[g]:.1f} Mha"
        ax.text(x0+sw+0.02,y+sh/2,lab,transform=ax.transAxes,ha="left",va="center",
                fontsize=fs,zorder=12)
    if areas is not None:
        tot=sum(areas.values()); bw=0.32; bh=0.024
        yb=list(ys.values())[len(shown)-1]-0.03-bh
        left=x0
        for g in GRADES:
            w=bw*areas[g]/tot
            if w<=0: continue
            ax.add_patch(Rectangle((left,yb),w,bh,transform=ax.transAxes,facecolor=GCOL[g],
                                   edgecolor="white",linewidth=0.3,zorder=12,clip_on=False))
            left+=w

def grade_areas(ri):
    """[1106] 每个时期三个等级的面积（Mha）：面积权重 tif 为 km²，÷1e4 → Mha。"""
    w,Z,_=load_zones(ri)
    if w is None: return None
    return {p:{g:float(w[Z[p]==GCODE[g]].sum())/1e4 for g in GRADES} for p in PERIODS}

_MEMBER_SHARE = {}
def member_share_df(ri):
    import pandas as pd
    if ri not in _MEMBER_SHARE:
        fp=ZONE/f"member_zone_share_{ri}.csv"
        _MEMBER_SHARE[ri]=pd.read_csv(fp) if fp.exists() else None
    return _MEMBER_SHARE[ri]

def add_member_violin(ax, ri, period, areas, fs=9):
    """[1106] 每幅地图左下角（西南无水稻区）小提琴插图：该时期三个等级面积占比在成员间的分布
    （基准 = 5 个 GCM，情景 = 25 个 GGCM×GCM 成员）；颜色 = 等级；三角 = 集合平均分级（地图所示）。
    数据：11 号输出 member_zone_share_{ri}.csv（premium/moderate/common_pct）。"""
    d=member_share_df(ri)
    if d is None: return
    d=d[d.period==period]
    if len(d)==0: return
    iax=ax.inset_axes([0.07,0.13,0.24,0.42]); iax.patch.set_alpha(0.0)
    tot=sum(areas.values()) if areas else None
    for k,(g,colname) in enumerate(zip(GRADES,["premium_pct","moderate_pct","common_pct"])):
        v=d[colname].values
        if np.nanmax(v)<0.5 and (areas is None or areas[g]<0.05):
            iax.scatter([k],[0],s=6,color=GCOL[g],zorder=4); continue
        vp=iax.violinplot([v],positions=[k],widths=0.75,showextrema=False,showmedians=False)
        for b_ in vp["bodies"]:
            b_.set_facecolor(GCOL[g]); b_.set_edgecolor(GCOL[g]); b_.set_alpha(0.55); b_.set_linewidth(0.5)
        iax.hlines(np.median(v),k-0.25,k+0.25,color=GCOL[g],linewidth=1.2,zorder=4)
        iax.vlines(k,v.min(),v.max(),color=GCOL[g],linewidth=0.6,zorder=3)
    iax.set_xlim(-0.6,2.6); iax.set_ylim(-5,100); iax.set_xticks([])
    iax.set_yticks([0,50,100]); iax.tick_params(axis="y",labelsize=fs-2,length=1.5,pad=1)
    n=len(d); iax.set_title(f"Share (%), n = {n}",fontsize=fs-2,pad=1.5,loc="left")
    for sp in ("top","right"): iax.spines[sp].set_visible(False)
    for sp in ("left","bottom"): iax.spines[sp].set_linewidth(0.5)

def draw_zone_map(ax, ri, period):
    z=read(zone_path(ri,period))
    if z is not None:
        a,t=z; ext=(t.c,t.c+a.shape[1]*t.a,t.f+a.shape[0]*t.e,t.f)
        ax.imshow(np.where(np.isin(a,[1,2,3]),a,np.nan),extent=ext,origin="upper",
                  cmap=CMAP,norm=NORM,alpha=0.92,zorder=1,interpolation="nearest")
    if GCHN is not None: GCHN.boundary.plot(ax=ax,color="0.35",linewidth=0.28,zorder=3)
    ax.set_xlim(MAP_EXTENT[0],MAP_EXTENT[1]); ax.set_ylim(MAP_EXTENT[2],MAP_EXTENT[3])
    ax.set_aspect(MAP_ASPECT)
    ax.set_xticks(np.arange(80,131,20)); ax.set_yticks(np.arange(20,51,10))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v,_:f"{v:.0f}\u00B0E"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_:f"{v:.0f}\u00B0N"))
    ax.tick_params(labelsize=FS["map_tick"],pad=2)
    for sp in ax.spines.values(): sp.set_linewidth(0.7)
    if z is not None:
        a,t=z; ext=(t.c,t.c+a.shape[1]*t.a,t.f+a.shape[0]*t.e,t.f)
        iax=ax.inset_axes([0.785,0.025,0.190,0.300])
        iax.imshow(np.where(np.isin(a,[1,2,3]),a,np.nan),extent=ext,origin="upper",
                   cmap=CMAP,norm=NORM,alpha=0.92,zorder=1,interpolation="nearest")
        if GCHN is not None: GCHN.boundary.plot(ax=iax,color="black",linewidth=0.22,zorder=4)
        if GNANHAI is not None and len(GNANHAI): GNANHAI.boundary.plot(ax=iax,color="black",linewidth=0.28,zorder=4)
        if GJD is not None and len(GJD): GJD.boundary.plot(ax=iax,color="black",linewidth=0.5,zorder=5)
        iax.set_xlim(INSET_EXTENT[0],INSET_EXTENT[1]); iax.set_ylim(INSET_EXTENT[2],INSET_EXTENT[3])
        iax.set_xticks([]); iax.set_yticks([])
        for sp in iax.spines.values(): sp.set_linewidth(0.5)

def stage_matrix(w, za, zb):
    M=np.zeros((3,3))
    for i,g in enumerate(GRADES):
        for j,h in enumerate(GRADES):
            M[i,j]=w[(za==GCODE[g])&(zb==GCODE[h])].sum()
    return M

def draw_sankey3(ax, ri):
    out=load_zones(ri)
    if out[0] is None:
        ax.axis("off"); print(f"  [warn] {ri} sankey skipped"); return
    w,Z,_=out; tot=w.sum()
    xs=SANKEY_XS; wN=SANKEY_WN
    TOP,BOT,GAP=0.90,0.06,0.055
    span=(TOP-BOT)-GAP*(len(GRADES)-1)
    gi={g:i for i,g in enumerate(GRADES)}
    LBL_MIN=0.5

    def layout(marg):
        y=TOP; pos={}
        for g in GRADES:
            hh=span*marg[g]/tot; pos[g]=(y-hh,hh); y-=hh+GAP
        return pos
    marg={p:{g:float(w[Z[p]==GCODE[g]].sum()) for g in GRADES} for p in PERIODS}
    POS={p:layout(marg[p]) for p in PERIODS}

    def band_heights(vals, node_h):
        """等分：某节点等级下，出/入的各条色带宽度一致(=节点高/条数)。
        宽度不再表示迁移量，量级由色带上的数字承载。"""
        keys=[k for k,v in vals.items() if v>0]
        if not keys: return {}
        h=node_h/len(keys)
        return {k:h for k in keys}

    strok=[pe.withStroke(linewidth=1.0,foreground="white")]
    dstrok=[pe.withStroke(linewidth=1.3,foreground="white")]

    def flows(M, src, tgt, x_src, x_tgt):
        side="left" if x_tgt<x_src else "right"
        xlbl = x_tgt+0.008 if side=="left" else x_tgt-0.008
        halbl= "left"     if side=="left" else "right"
        sh={i:band_heights({j:M[i][j] for j in range(3)}, POS[src][GRADES[i]][1]) for i in range(3)}
        dh={j:band_heights({i:M[i][j] for i in range(3)}, POS[tgt][GRADES[j]][1]) for j in range(3)}
        soff={g:POS[src][g][0]+POS[src][g][1] for g in GRADES}
        toff={g:POS[tgt][g][0]+POS[tgt][g][1] for g in GRADES}
        for i,sg in enumerate(GRADES):
            for j,tg in enumerate(GRADES):
                if j not in sh[i] or i not in dh[j]: continue
                h0=sh[i][j]; h1=dh[j][i]
                sy1=soff[sg]; sy0=sy1-h0; soff[sg]=sy0
                dy1=toff[tg]; dy0=dy1-h1; toff[tg]=dy0
                xm=(x_src+x_tgt)/2
                verts=[(x_src,sy0),(xm,sy0),(xm,dy0),(x_tgt,dy0),
                       (x_tgt,dy1),(xm,dy1),(xm,sy1),(x_src,sy1),(x_src,sy0)]
                codes=[MPath.MOVETO,MPath.CURVE4,MPath.CURVE4,MPath.CURVE4,MPath.LINETO,
                       MPath.CURVE4,MPath.CURVE4,MPath.CURVE4,MPath.CLOSEPOLY]
                d="down" if gi[tg]>gi[sg] else ("up" if gi[tg]<gi[sg] else "same")
                ax.add_patch(PathPatch(MPath(verts,codes),facecolor=DCOL[d],
                             alpha=0.85 if d!="same" else 0.55,edgecolor="white",
                             linewidth=0.3,zorder=2))
                pct=100*M[i][j]/tot
                if pct>=LBL_MIN:
                    ax.text(xlbl,dy0+h1/2,f"{pct:.0f}" if pct>=1 else f"{pct:.1f}",
                            ha=halbl,va="center",fontsize=4.3,color="#2E2E2E",
                            zorder=6,path_effects=strok)

    flows(stage_matrix(w,Z["baseline"],Z["ssp126"]),"baseline","ssp126",
          xs["baseline"], xs["ssp126"]+wN)
    flows(stage_matrix(w,Z["baseline"],Z["ssp585"]),"baseline","ssp585",
          xs["baseline"]+wN, xs["ssp585"])

    for p in SANKEY_ORDER:
        for g in GRADES:
            yb,hh=POS[p][g]
            ax.add_patch(Rectangle((xs[p],yb),wN,hh,color=GCOL[g],zorder=3))

    OUTSIDE={"ssp126":("left",  lambda x:xs["ssp126"]-0.012),
             "baseline":("right",lambda x:xs["baseline"]+wN+0.012),
             "ssp585":("right", lambda x:xs["ssp585"]+wN+0.012)}
    for p in PERIODS:
        for g in GRADES:
            yb,hh=POS[p][g]; pct=100*marg[p][g]/tot
            if pct<0.5: continue
            txt=f"{pct:.0f}%"
            if hh>=0.055:
                ax.text(xs[p]+wN/2,yb+hh/2,txt,ha="center",va="center",fontsize=5.0,
                        color="white",fontweight="bold",zorder=5,path_effects=None)
            else:
                sidelab,fx=OUTSIDE[p]; ha="left" if sidelab=="right" else "right"
                ax.text(fx(None),yb+hh/2,txt,ha=ha,va="center",fontsize=5.2,
                        color="0.25",zorder=6,path_effects=dstrok)
    ax.set_xlim(*SANKEY_XLIM); ax.set_ylim(0,1.0); ax.axis("off")

def _arc(a0,a1,r,n=None):
    if n is None: n=max(2,int(abs(a1-a0)/0.045))
    t=np.linspace(a0,a1,n); return list(zip(r*np.cos(t), r*np.sin(t)))

def _sub(a0,a1,items,minspan=0.0):
    """把 [a0,a1] 按 items=[(key,weight)] 顺序切成子弧；weight<=0 跳过(不留空段)。
    minspan>0 时给每条一个最小角宽(细带也能显丝)，再归一化填满。"""
    items=[(k,v) for k,v in items if v>0]
    W=sum(v for _,v in items); out={}
    if W<=0: return out
    span=a1-a0
    hs={k: span*v/W for k,v in items}
    if minspan>0:
        hs={k:max(h,minspan) for k,h in hs.items()}
        s=sum(hs.values()); hs={k:h*span/s for k,h in hs.items()}
    a=a0
    for k,_ in items:
        e=a+hs[k]; out[k]=(a,e); a=e
    return out

def _sub_gap(a0,a1,items,gap):
    """同 _sub，但每个子弧之间留 gap(弧度)，用于等级段之间的留白。"""
    items=[(k,v) for k,v in items if v>0]; n=len(items)
    if n==0: return {}
    span=(a1-a0)-gap*(n-1); W=sum(v for _,v in items); out={}; a=a0
    for k,v in items:
        e=a+span*v/W; out[k]=(a,e); a=e+gap
    return out

def draw_chord(ax, ri):
    out=load_zones(ri)
    if out[0] is None: ax.axis("off"); print(f"  [warn] {ri} chord skipped"); return
    w,Z,_=out; tot=w.sum()
    SCEN=["ssp126","ssp585"]
    B  ={g:float(w[Z["baseline"]==GCODE[g]].sum()) for g in GRADES}
    Sar={s:{g:float(w[Z[s]==GCODE[g]].sum()) for g in GRADES} for s in SCEN}
    T  ={s:stage_matrix(w,Z["baseline"],Z[s]) for s in SCEN}
    draw_chord_core(ax, B, Sar, T, tot)

def draw_chord_core(ax, B, Sar, T, tot, SCEN=("ssp126","ssp585"), MINSHARE=0.004):
    """按 (baseline 各等级面积 B, 情景各等级面积 Sar[s][g], 转移矩阵 T[s][gb][gs], 总面积 tot)
    画圆形 chord。MINSHARE: 占比< 该值(默认 0.4%)的等级视为不存在(不画段)。"""
    SCEN=list(SCEN)
    gi={g:i for i,g in enumerate(GRADES)}

    bl_grades=[g for g in GRADES if B[g]/tot>MINSHARE]
    sc_grades={s:[g for g in GRADES if Sar[s][g]/tot>MINSHARE] for s in SCEN}

    Wg={"baseline":sum(B[g] for g in bl_grades)*len(SCEN),
        "ssp126":sum(Sar["ssp126"][g] for g in sc_grades["ssp126"]),
        "ssp585":sum(Sar["ssp585"][g] for g in sc_grades["ssp585"])}
    order=["baseline","ssp585","ssp126"]
    gap=math.radians(5); avail=2*math.pi-gap*len(order); W=sum(Wg.values())
    ang={}; a=0.0
    for k in order:
        sp=avail*Wg[k]/W; ang[k]=(a,a+sp); a+=sp+gap
    bc=(ang["baseline"][0]+ang["baseline"][1])/2; rot=math.radians(-90)-bc
    ang={k:(s+rot,e+rot) for k,(s,e) in ang.items()}

    R,dr=1.0,0.08
    GGAP=math.radians(0.0)
    bl_g=_sub_gap(*ang["baseline"], [(g,B[g]*len(SCEN)) for g in bl_grades], GGAP)
    sc_g={s:_sub_gap(*ang[s], [(g,Sar[s][g]) for g in sc_grades[s]], GGAP) for s in SCEN}

    bl_slot={}
    for g in bl_grades:
        items=[]
        for s in ["ssp585","ssp126"]:
            for gs in GRADES:
                items.append(((s,gs), T[s][gi[g]][gi[gs]]))
        for k,sl in _sub(*bl_g[g], items).items(): bl_slot[(g,)+k]=sl
    sc_slot={}
    for s in SCEN:
        for gs in sc_grades[s]:
            items=[(gb, T[s][gi[gb]][gi[gs]]) for gb in bl_grades]
            for gb,sl in _sub(*sc_g[s][gs], items).items(): sc_slot[(s,gb,gs)]=sl

    for g in bl_grades:
        for s in SCEN:
            for gs in GRADES:
                v=T[s][gi[g]][gi[gs]]
                if v<=0: continue
                if (g,s,gs) not in bl_slot or (s,g,gs) not in sc_slot: continue
                a0,a1=bl_slot[(g,s,gs)]; b0,b1=sc_slot[(s,g,gs)]
                pa=_arc(a0,a1,R); pb=_arc(b1,b0,R)
                verts=list(pa)+[(0,0),pb[0]]+list(pb[1:])+[(0,0),pa[0]]
                codes=[MPath.MOVETO]+[MPath.LINETO]*(len(pa)-1)+[MPath.CURVE3,MPath.CURVE3]\
                      +[MPath.LINETO]*(len(pb)-1)+[MPath.CURVE3,MPath.CURVE3]
                d="down" if gi[gs]>gi[g] else ("up" if gi[gs]<gi[g] else "same")
                ax.add_patch(PathPatch(MPath(verts,codes),facecolor=DCOL[d],
                             alpha=0.80 if d!="same" else 0.42,edgecolor="white",
                             linewidth=0.25,zorder=2))

    def deg(a): return math.degrees(a)
    seg_all=[("baseline",g,se) for g,se in bl_g.items()]
    for s in SCEN:
        seg_all+=[(s,g,se) for g,se in sc_g[s].items()]
    for grp,g,(s0,s1) in seg_all:
        ax.add_patch(Wedge((0,0),R+dr,deg(s0),deg(s1),width=dr,
                     facecolor=GCOL[g],edgecolor="white",linewidth=0.6,zorder=3))

    GRP_COL=dict(PP.SSP_RING)
    Rg=R+dr+0.19; gdr=0.045
    for k,(s0,s1) in ang.items():
        ax.add_patch(Wedge((0,0),Rg+gdr,deg(s0),deg(s1),width=gdr,
                     facecolor=GRP_COL[k],edgecolor="white",linewidth=0.6,zorder=3))


    AREA={("baseline",g):B[g] for g in bl_grades}
    for s in SCEN:
        for g in sc_grades[s]: AREA[(s,g)]=Sar[s][g]
    lstrk=[pe.withStroke(linewidth=1.8,foreground="white")]
    rr_pct=R+dr+0.08
    groups={"baseline":list(bl_grades)}
    for s in SCEN: groups[s]=list(sc_grades[s])
    for grp,gl in groups.items():
        a0,a1=ang[grp]; n=len(gl)
        for i,g in enumerate(gl):
            m=a0+(i+0.5)/n*(a1-a0)
            pct=100*AREA[(grp,g)]/tot
            rot=deg(m)-90
            if rot>90: rot-=180
            if rot<-90: rot+=180
            ax.text(rr_pct*math.cos(m),rr_pct*math.sin(m),f"{pct:.0f}%",
                    ha="center",va="center",rotation=rot,rotation_mode="anchor",
                    fontsize=FS["chord_pct"],color=GCOL[g],fontweight="normal",
                    zorder=7,path_effects=lstrk)
    PLAB2={"baseline":"Baseline","ssp126":"SSP1-2.6","ssp585":"SSP5-8.5"}
    for k,(s0,s1) in ang.items():
        m=(s0+s1)/2; rr=Rg+gdr+0.085; rot=deg(m)-90
        if rot>90: rot-=180
        if rot<-90: rot+=180
        ax.text(rr*math.cos(m),rr*math.sin(m),PLAB2[k],ha="center",va="center",
                rotation=rot,rotation_mode="anchor",fontsize=FS["chord_grp"],fontweight="normal")
    ax.set_xlim(-1.62,1.62); ax.set_ylim(-1.62,1.62); ax.set_aspect("equal"); ax.axis("off")


def save_pub(fig,stem,dpi=600):
    for e in ("svg","pdf"): fig.savefig(OUTFIG/f"{stem}.{e}",bbox_inches="tight")
    fig.savefig(OUTFIG/f"{stem}.tiff",dpi=dpi,bbox_inches="tight")
    fig.savefig(OUTFIG/f"{stem}.png",dpi=200,bbox_inches="tight")

SHOW_CHORD = True
if __name__=="__main__":
    if SHOW_CHORD:
        fig=plt.figure(figsize=(200/25.4,285/25.4))
        gs=fig.add_gridspec(4,2,height_ratios=[1.2,1.2,1.2,1.5],
                            left=0.135,right=0.995,top=0.965,bottom=0.045,
                            hspace=0.03,wspace=0.0)
    else:
        fig=plt.figure(figsize=(200/25.4,208/25.4))
        gs=fig.add_gridspec(3,2,left=0.135,right=0.995,top=0.955,bottom=0.055,
                            hspace=0.03,wspace=0.0)
    ROWLAB=["Baseline","SSP1-2.6","SSP5-8.5","Grade transitions"]
    base_axes=[]; map_axes=[]
    AREAS={ri:grade_areas(ri) for ri in RI}
    for row,p in enumerate(PERIODS):
        for col,ri in enumerate(RI):
            ax=fig.add_subplot(gs[row,col])
            draw_zone_map(ax,ri,p)
            if AREAS[ri] is not None:
                add_grade_legend(ax,FS["grade_leg"],AREAS[ri][p])
                add_member_violin(ax,ri,p,AREAS[ri][p],fs=FS["grade_leg"]-1)
            if row==0: ax.set_title(RILAB[ri],fontsize=FS["map_title"],pad=4)
            if row<2: ax.set_xticklabels([])
            if col>0: ax.set_yticklabels([])
            if col==0 and row==0: base_axes.append(ax)
            if col==0: map_axes.append(ax)
    pl=[Patch(facecolor=GCOL[g],label=g) for g in GRADES]
    if SHOW_CHORD:
        for col,ri in enumerate(RI):
            ax=fig.add_subplot(gs[3,col]); draw_chord(ax,ri)
            if col==0: map_axes.append(ax)
        fl=[Patch(facecolor=DCOL["down"],alpha=.80,label="Downgrade"),
            Patch(facecolor=DCOL["same"],alpha=.42,label="Stable"),
            Patch(facecolor=DCOL["up"],alpha=.80,label="Upgrade")]
        l1=fig.legend(handles=pl,loc="lower center",ncol=3,fontsize=FS["legend"],
                      bbox_to_anchor=(0.35,0.004),handlelength=0.6,handletextpad=0.1,columnspacing=0.5)
        fig.add_artist(l1)
        fig.legend(handles=fl,loc="lower center",ncol=3,fontsize=FS["legend"],
                   bbox_to_anchor=(0.78,0.004),handlelength=0.6,handletextpad=0.1,columnspacing=0.5)
    else:
        fig.legend(handles=pl,loc="lower center",ncol=3,fontsize=FS["legend"],
                   bbox_to_anchor=(0.565,0.004),handlelength=0.8,handletextpad=0.3,columnspacing=2.0)

    for row,ax0 in enumerate(map_axes):
        p0=ax0.get_position()
        fig.text(0.028,(p0.y0+p0.y1)/2,ROWLAB[row],rotation=90,
                 fontsize=FS["row_label"],va="center",ha="center")
    p0=base_axes[0].get_position()
    add_scale_bar(base_axes[0],x_left=p0.x0+0.006,cy=p0.y0+0.006)

    save_pub(fig,"fig4_grid"); plt.close(fig)
    print("[saved] fig4_grid ->",OUTFIG)

    figc,axc=plt.subplots(1,2,figsize=(200/25.4,112/25.4))
    for ax,ri in zip(axc,RI):
        draw_chord(ax,ri); ax.set_title(RILAB[ri],fontsize=9,pad=4)
    figc.tight_layout()
    save_pub(figc,"fig4_chord"); plt.close(figc)
    print("[saved] fig4_chord (standae big chords) ->",OUTFIG)

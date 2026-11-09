"""Scanner Ichimoku — Perps Hyperliquid (daily), version automatisée du notebook
scanner_ichimoku_hyperliquid.ipynb (étapes 1 à 3 ; l'étape 4 est faite par la tâche planifiée Claude).

1) univers : les NB_PERPS perps au plus gros volume moyen sur JOURS_VOLUME jours (HIP-3 inclus)
2) Ichimoku 9/26/52 en daily, bougies clôturées uniquement
3) Kumo Breakout du jour, trié par liquidité → data/latest.json et data/AAAA-MM-JJ.json
"""

import json
import os
import sys
import time
from datetime import datetime, timezone

import pandas as pd
import requests

# ============ PARAMÈTRES (identiques au notebook) ============
NB_PERPS = 100
JOURS_VOLUME = 90
INCLURE_HIP3 = True
PARAMS_ICHIMOKU = {"9/26/52": (9, 26, 52)}
JOURS_FENETRE = 90
INCLURE_BOUGIE_EN_COURS = False
PAUSE_API_S = 0.7
MARGE_CALCUL = 110

HL_URL = "https://api.hyperliquid.xyz/info"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def hl_post(payload, essais=6):
    for i in range(essais):
        r = requests.post(HL_URL, json=payload, timeout=30)
        if r.status_code == 429:
            time.sleep(2 ** i)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError("Limite de débit Hyperliquid persistante")


# ---------- Étape 1 : univers ----------
def lister_perps():
    dexs = [""]  # "" = perps principaux
    if INCLURE_HIP3:
        try:
            for d in hl_post({"type": "perpDexs"}):
                if d and d.get("name"):
                    dexs.append(d["name"])
        except Exception as e:
            print("perpDexs indisponible, HIP-3 ignoré :", e)
    lignes = []
    for dex in dexs:
        try:
            meta, ctxs = hl_post({"type": "metaAndAssetCtxs", "dex": dex})
        except Exception as e:
            print(f"dex '{dex or 'core'}' ignoré :", e)
            continue
        for actif, ctx in zip(meta["universe"], ctxs):
            if actif.get("isDelisted"):
                continue
            lignes.append({
                "coin": actif["name"],
                "dex": dex or "core",
                "volume_24h_usd": float(ctx.get("dayNtlVlm") or 0),
                "max_leverage": actif.get("maxLeverage"),
            })
        time.sleep(PAUSE_API_S)
    return pd.DataFrame(lignes)


def charger_bougies(coin):
    maintenant = int(time.time() * 1000)
    debut = maintenant - (JOURS_FENETRE + MARGE_CALCUL) * 86_400_000
    data = hl_post({"type": "candleSnapshot",
                    "req": {"coin": coin, "interval": "1d", "startTime": debut, "endTime": maintenant}})
    if not data:
        return None
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    df["cloturee"] = df["T"] < maintenant
    return df[["date", "open", "high", "low", "close", "volume", "cloturee"]].reset_index(drop=True)


# ---------- Étape 2 : Ichimoku ----------
def ichimoku(df, t, k, s):
    h, l, c = df["high"], df["low"], df["close"]
    tenkan = (h.rolling(t).max() + l.rolling(t).min()) / 2
    kijun = (h.rolling(k).max() + l.rolling(k).min()) / 2
    span_a_brut = (tenkan + kijun) / 2
    span_b_brut = (h.rolling(s).max() + l.rolling(s).min()) / 2
    out = df[["date", "open", "high", "low", "close"]].copy()
    out["tenkan"], out["kijun"] = tenkan, kijun
    out["span_a"], out["span_b"] = span_a_brut.shift(k), span_b_brut.shift(k)  # nuage au jour J
    out["span_a_futur"], out["span_b_futur"] = span_a_brut, span_b_brut        # nuage projeté (J+k)
    out["kumo_haut"] = out[["span_a", "span_b"]].max(axis=1, skipna=False)
    out["kumo_bas"] = out[["span_a", "span_b"]].min(axis=1, skipna=False)
    out["close_moins_k"] = c.shift(k)                                           # pour la Chikou
    return out


# ---------- Étape 3 : signaux ----------
def detecter_signaux(ich):
    if len(ich) < 2:
        return []
    a, b = ich.iloc[-2], ich.iloc[-1]
    valeurs = [a.kumo_haut, a.kumo_bas, b.kumo_haut, b.kumo_bas, a.close, b.close]
    if any(pd.isna(v) for v in valeurs):
        return []
    res = []
    if a.close <= a.kumo_haut and b.close > b.kumo_haut:
        res.append("LONG")
    if a.close >= a.kumo_bas and b.close < b.kumo_bas:
        res.append("SHORT")
    return res


def decrire(coin, params, sens, ich, info):
    b = ich.iloc[-1]
    long_ = sens == "LONG"
    sl = b.kumo_bas if long_ else b.kumo_haut
    futur_vert = b.span_a_futur > b.span_b_futur
    tk = b.tenkan > b.kijun
    chikou = None if pd.isna(b.close_moins_k) else bool(b.close > b.close_moins_k if long_ else b.close < b.close_moins_k)
    return {
        "coin": coin,
        "dex": info["dex"],
        "params": params,
        "direction": sens,
        "candle_date": b["date"].strftime("%Y-%m-%d"),
        "close": float(b.close),
        "kumo_top": float(b.kumo_haut),
        "kumo_bottom": float(b.kumo_bas),
        "kumo_thickness_pct": round(float((b.kumo_haut - b.kumo_bas) / b.close * 100), 2),
        "sl_opposite_edge": float(sl),
        "sl_distance_pct": round(float(abs(b.close - sl) / b.close * 100), 2),
        "tenkan": float(b.tenkan),
        "kijun": float(b.kijun),
        "tenkan_above_kijun": bool(tk),
        "chikou_confirms": chikou,
        "future_kumo": "vert" if futur_vert else "rouge",
        "future_kumo_confirms": bool(futur_vert if long_ else not futur_vert),
        "vol_moy_90j_usd": round(info["vol_moy_90j_usd"]),
        "volume_24h_usd": round(info["volume_24h_usd"]),
        "max_leverage": None if pd.isna(info["max_leverage"]) else int(info["max_leverage"]),
    }


def main():
    univers = lister_perps()
    univers = univers[univers["volume_24h_usd"] > 0].reset_index(drop=True)
    print(f"{len(univers)} perps listés avec du volume — chargement des bougies daily...")

    bougies_toutes, volumes, erreurs = {}, [], {}
    for i, row in enumerate(univers.itertuples(), 1):
        try:
            b = charger_bougies(row.coin)
            if b is not None and len(b):
                bougies_toutes[row.coin] = b
                cl = b[b["cloturee"]].tail(JOURS_VOLUME)
                volumes.append({"coin": row.coin, "dex": row.dex,
                                "vol_moy_90j_usd": float((cl["volume"] * cl["close"]).sum() / JOURS_VOLUME),
                                "volume_24h_usd": row.volume_24h_usd,
                                "max_leverage": row.max_leverage})
        except Exception as e:
            erreurs[row.coin] = str(e)
        time.sleep(PAUSE_API_S)
        if i % 50 == 0:
            print(f"  {i}/{len(univers)} perps chargés...")

    perps = (pd.DataFrame(volumes).sort_values("vol_moy_90j_usd", ascending=False)
             .head(NB_PERPS).reset_index(drop=True))
    print(f"{len(perps)} perps retenus (volume moyen 90 j ≥ {perps['vol_moy_90j_usd'].min():,.0f} $)")

    signaux = []
    for info in perps.to_dict("records"):
        coin = info["coin"]
        bougies = bougies_toutes[coin]
        if not INCLURE_BOUGIE_EN_COURS:
            bougies = bougies[bougies["cloturee"]]
        bougies = bougies.reset_index(drop=True)
        if len(bougies) < 10:
            continue
        for nom, p in PARAMS_ICHIMOKU.items():
            ich = ichimoku(bougies, *p).tail(JOURS_FENETRE).reset_index(drop=True)
            for sens in detecter_signaux(ich):
                signaux.append(decrire(coin, nom, sens, ich, info))

    signaux.sort(key=lambda s: -s["vol_moy_90j_usd"])
    run_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "strategy": "Kumo Breakout " + ", ".join(PARAMS_ICHIMOKU) + " (daily, bougies clôturées)",
        "universe": f"top {NB_PERPS} perps Hyperliquid par volume moyen {JOURS_VOLUME} j"
                    + (", HIP-3 inclus" if INCLURE_HIP3 else ""),
        "scanned": int(len(perps)),
        "signals": signaux,
        "errors": dict(list(erreurs.items())[:20]),
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    for name in ("latest.json", f"{run_date}.json"):
        with open(os.path.join(OUT_DIR, name), "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"{len(signaux)} signaux, {len(erreurs)} erreurs de chargement")
    return 0 if len(perps) else 1


if __name__ == "__main__":
    sys.exit(main())

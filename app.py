"""
GOLD QUANT DESK v2  |  Futuros de Oro (GC=F)
---------------------------------------------------------------------------
Tab 1  Order Flow (Volume Delta + CVD) y Smart Money Concepts (OB / FVG)
Tab 2  Matriz macro y correlaciones cruzadas
Tab 3  Walk-Forward: VAR (statsmodels) vs MLPRegressor (scikit-learn) vs precio real
Tab 4  Matriz estadistica: Win Rate y EV (R) con R:R fijo 1:3 por nivel de confluencia

Despliegue: GitHub -> Streamlit Community Cloud (archivo principal: app.py).
Los datos provienen de Yahoo Finance (retrasados); el Volume Delta es una estimacion OHLCV.
Material analitico; no constituye asesoria financiera.
"""

import time
import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# CONSTANTES
# ---------------------------------------------------------------------------
GOLD = "GC=F"
MACRO = {  # clave -> tickers candidatos (el primero que responda se usa)
    "DXY": ["DX-Y.NYB", "DX=F"],
    "TNX": ["^TNX"],
    "SPX": ["^GSPC"],
    "HG": ["HG=F"],
    "VIX": ["^VIX"],
}
MACRO_LABEL = {"XAU": "Oro", "DXY": "Dolar DXY", "TNX": "Tasa 10Y", "SPX": "S&P 500",
               "HG": "Cobre", "VIX": "VIX"}
C_UP, C_DN, C_GOLD = "#26a69a", "#ef5350", "#f1c40f"
C_VAR, C_NN, C_REAL = "#ff9800", "#2ecc71", "#1e90ff"
RR = 3.0  # ratio R:R fijo 1:3


def fmt(x: float) -> str:
    return f"{x:,.2f}"


# ---------------------------------------------------------------------------
# 1. INGESTA DE DATOS
# ---------------------------------------------------------------------------
def _flatten(df: Optional[pd.DataFrame], ticker: str) -> Optional[pd.DataFrame]:
    """Aplana columnas multinivel de yfinance (Price, Ticker) o (Ticker, Price)."""
    if df is None or df.empty or not isinstance(df.columns, pd.MultiIndex):
        return df
    price_lvl = next((i for i in range(df.columns.nlevels)
                      if "Close" in df.columns.get_level_values(i)), 0)
    for lv in [i for i in range(df.columns.nlevels) if i != price_lvl]:
        if ticker in df.columns.get_level_values(lv):
            return df.xs(ticker, axis=1, level=lv)
    out = df.copy()
    out.columns = out.columns.get_level_values(price_lvl)
    return out.loc[:, ~out.columns.duplicated()]


@st.cache_data(ttl=900, show_spinner=False)
def fetch_ohlcv(ticker: str, years: int) -> pd.DataFrame:
    """OHLCV diario normalizado (indice sin zona horaria, a medianoche). Vacio si falla."""
    period = f"{years}y"
    df = pd.DataFrame()
    for attempt in range(3):
        for mode in ("history", "download"):
            try:
                if mode == "history":
                    d = yf.Ticker(ticker).history(period=period, interval="1d",
                                                  auto_adjust=False, actions=False)
                else:
                    d = yf.download(ticker, period=period, interval="1d", auto_adjust=False,
                                    progress=False, threads=False)
                d = _flatten(d, ticker)
                if d is not None and not d.empty and "Close" in d.columns:
                    df = d
                    break
            except Exception:
                continue
        if not df.empty:
            break
        time.sleep(1.0 + attempt)
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    for col in ["Open", "High", "Low"]:
        if col not in df.columns:
            df[col] = df["Close"]
    if "Volume" not in df.columns:
        df["Volume"] = 0.0
    df = df[["Open", "High", "Low", "Close", "Volume"]].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["Close"])
    for col in ["Open", "High", "Low"]:
        df[col] = df[col].fillna(df["Close"])
    df["Volume"] = df["Volume"].fillna(0.0)
    return df[~df.index.duplicated(keep="last")].sort_index()


@st.cache_data(ttl=900, show_spinner=False)
def load_data(years: int):
    """Devuelve (df_raw, df_model, faltantes). df_model: oro + macro alineado al calendario del oro."""
    df_raw = fetch_ohlcv(GOLD, years)
    if df_raw.empty:
        return pd.DataFrame(), pd.DataFrame(), ["GC=F"]
    df_model = df_raw.rename(columns={c: f"XAU_{c}" for c in df_raw.columns})
    missing = []
    for key, cands in MACRO.items():
        s = None
        for tk in cands:
            d = fetch_ohlcv(tk, years)
            if not d.empty:
                s = d["Close"]
                break
        if s is None:
            missing.append(key)
            continue
        df_model[key] = s.reindex(df_model.index).ffill(limit=3)
    if missing:
        return df_raw, pd.DataFrame(), missing
    if df_model["TNX"].median() > 25:     # Yahoo a veces cotiza ^TNX en decimas de %
        df_model["TNX"] = df_model["TNX"] / 10.0
    df_model = df_model.dropna()
    return df_raw, df_model, []


# ---------------------------------------------------------------------------
# 2. ORDER FLOW Y SMC
# ---------------------------------------------------------------------------
def compute_atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["Close"].shift(1)
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - pc).abs(), (df["Low"] - pc).abs()],
                   axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def add_order_flow(df: pd.DataFrame) -> pd.DataFrame:
    """Volume Delta por vela = Vol*(C-L)/(H-L) - Vol*(H-C)/(H-L); CVD = acumulado."""
    out = df.copy()
    rng = (out["High"] - out["Low"]).replace(0, np.nan)
    buy = out["Volume"] * (out["Close"] - out["Low"]) / rng
    sell = out["Volume"] * (out["High"] - out["Close"]) / rng
    out["Delta"] = (buy - sell).fillna(0.0)
    out["CVD"] = out["Delta"].cumsum()
    return out


def _first_touch(h: np.ndarray, l: np.ndarray, kind: str, top: float, bottom: float, avail: int) -> int:
    if avail >= len(h):
        return -1
    hits = np.where(l[avail:] <= top)[0] if kind == "demand" else np.where(h[avail:] >= bottom)[0]
    return int(avail + hits[0]) if len(hits) else -1


def _zone(kind, typ, start, top, bottom, avail, touch, pos) -> Dict:
    return dict(kind=kind, type=typ, start=start, top=float(top), bottom=float(bottom),
                avail=int(avail), touch=int(touch), mitigated=touch >= 0, pos=int(pos))


def detect_fvg(df: pd.DataFrame, min_atr_frac: float = 0.10) -> List[Dict]:
    """Fair Value Gap de 3 velas. Utilizable desde la vela siguiente a la formacion."""
    h, l = df["High"].values, df["Low"].values
    n = len(df)
    if n < 5:
        return []
    atr = compute_atr(df).values
    idx = df.index
    zones = []
    for i in range(2, n):
        a = atr[i - 1]
        thr = (0.0 if np.isnan(a) else a) * min_atr_frac
        if l[i] - h[i - 2] > 0 and l[i] - h[i - 2] >= thr:
            top, bot = l[i], h[i - 2]
            zones.append(_zone("demand", "FVG", idx[i - 1], top, bot, i + 1,
                               _first_touch(h, l, "demand", top, bot, i + 1), i))
        elif l[i - 2] - h[i] > 0 and l[i - 2] - h[i] >= thr:
            top, bot = l[i - 2], h[i]
            zones.append(_zone("supply", "FVG", idx[i - 1], top, bot, i + 1,
                               _first_touch(h, l, "supply", top, bot, i + 1), i))
    return zones


def detect_order_blocks(df: pd.DataFrame, disp_atr: float = 1.5, lookback: int = 5) -> List[Dict]:
    """OB = ultima vela contraria previa a un desplazamiento (cuerpo >= disp_atr*ATR) con ruptura de estructura."""
    o, h, l, c = (df[k].values for k in ["Open", "High", "Low", "Close"])
    n = len(df)
    if n < 20:
        return []
    atr = compute_atr(df).values
    idx = df.index
    zones, used = [], set()
    for k in range(4, n - 1):
        a = atr[k - 1]
        if np.isnan(a) or a <= 0 or abs(c[k] - o[k]) < disp_atr * a:
            continue
        if c[k] > o[k] and c[k] > np.max(h[k - 3:k]):
            for j in range(k - 1, max(k - 1 - lookback, -1), -1):
                if c[j] < o[j] and j not in used:
                    used.add(j)
                    zones.append(_zone("demand", "OB", idx[j], h[j], l[j], k + 1,
                                       _first_touch(h, l, "demand", h[j], l[j], k + 1), j))
                    break
        elif c[k] < o[k] and c[k] < np.min(l[k - 3:k]):
            for j in range(k - 1, max(k - 1 - lookback, -1), -1):
                if c[j] > o[j] and j not in used:
                    used.add(j)
                    zones.append(_zone("supply", "OB", idx[j], h[j], l[j], k + 1,
                                       _first_touch(h, l, "supply", h[j], l[j], k + 1), j))
                    break
    return zones


def nearest_zone(zones: List[Dict], price: float, kind: str) -> Optional[Dict]:
    cand = [z for z in zones if z["kind"] == kind and not z["mitigated"]]
    if kind == "demand":
        cand = [z for z in cand if z["bottom"] < price]
        key = lambda z: (max(0.0, price - z["top"]), 0 if z["type"] == "OB" else 1)
    else:
        cand = [z for z in cand if z["top"] > price]
        key = lambda z: (max(0.0, z["bottom"] - price), 0 if z["type"] == "OB" else 1)
    return min(cand, key=key) if cand else None


def zone_distance_atr(z: Dict, price: float, atr: float) -> float:
    d = max(0.0, price - z["top"]) if z["kind"] == "demand" else max(0.0, z["bottom"] - price)
    return d / atr if atr > 0 else float("nan")


def cvd_divergence(df: pd.DataFrame, window: int = 20) -> str:
    if len(df) <= window + 1:
        return "Datos insuficientes para evaluar divergencia."
    dp = df["Close"].iloc[-1] - df["Close"].iloc[-window - 1]
    dc = df["CVD"].iloc[-1] - df["CVD"].iloc[-window - 1]
    if dp > 0 and dc > 0:
        return f"Convergencia alcista ({window}d): precio y CVD al alza (acumulacion confirmada)."
    if dp < 0 and dc < 0:
        return f"Convergencia bajista ({window}d): precio y CVD a la baja (distribucion confirmada)."
    if dp > 0 and dc < 0:
        return f"DIVERGENCIA BAJISTA ({window}d): el precio sube con CVD cayendo (posible distribucion)."
    return f"DIVERGENCIA ALCISTA ({window}d): el precio cae con CVD subiendo (posible acumulacion)."


def build_order_flow_chart(df: pd.DataFrame, zones: List[Dict], view_bars: int,
                           show_mitigated: bool) -> go.Figure:
    view = df.tail(view_bars)
    v0, v1 = view.index[0], view.index[-1]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28],
                        vertical_spacing=0.03, specs=[[{}], [{"secondary_y": True}]])
    fig.add_trace(go.Candlestick(x=view.index, open=view["Open"], high=view["High"], low=view["Low"],
                                 close=view["Close"], increasing_line_color=C_UP,
                                 decreasing_line_color=C_DN, name="GC=F"), row=1, col=1)

    idx = df.index
    for kind in ["demand", "supply"]:
        for typ in ["OB", "FVG"]:
            sel = [z for z in zones if z["kind"] == kind and z["type"] == typ]
            live = sorted([z for z in sel if not z["mitigated"]], key=lambda z: z["pos"])[-6:]
            hist = []
            if show_mitigated:
                hist = [z for z in sel if z["mitigated"] and idx[z["touch"]] >= v0]
                hist = sorted(hist, key=lambda z: z["pos"])[-4:]
            for z in live + hist:
                x0 = max(z["start"], v0)
                x1 = idx[z["touch"]] if z["mitigated"] else v1
                base = (38, 166, 154) if kind == "demand" else (239, 83, 80)
                alpha = 0.07 if z["mitigated"] else (0.24 if typ == "OB" else 0.12)
                color = C_UP if kind == "demand" else C_DN
                fig.add_shape(type="rect", x0=x0, x1=x1, y0=z["bottom"], y1=z["top"],
                              fillcolor=f"rgba({base[0]},{base[1]},{base[2]},{alpha})",
                              line=dict(color=color, width=1, dash="solid" if typ == "OB" else "dot"),
                              layer="below", row=1, col=1)
                arrow = "↑" if kind == "demand" else "↓"
                fig.add_annotation(x=x0, y=z["top"], text=f"{typ}{arrow}", showarrow=False,
                                   font=dict(size=9, color=color), xanchor="left", yanchor="bottom",
                                   row=1, col=1)

    colors = [C_UP if d >= 0 else C_DN for d in view["Delta"]]
    fig.add_trace(go.Bar(x=view.index, y=view["Delta"], marker_color=colors, opacity=0.85,
                         name="Volume Delta"), row=2, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=view.index, y=view["CVD"], mode="lines",
                             line=dict(color=C_GOLD, width=2), name="CVD"),
                  row=2, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Delta", row=2, col=1, secondary_y=False, showgrid=False)
    fig.update_yaxes(title_text="CVD", row=2, col=1, secondary_y=True, showgrid=False,
                     title_font=dict(color=C_GOLD), tickfont=dict(color=C_GOLD))
    fig.update_xaxes(rangeslider_visible=False, rangebreaks=[dict(bounds=["sat", "mon"])])
    fig.update_layout(template="plotly_dark", height=720, margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", y=1.05), bargap=0.15)
    return fig


# ---------------------------------------------------------------------------
# 3. MACRO
# ---------------------------------------------------------------------------
def make_returns(dfm: pd.DataFrame) -> pd.DataFrame:
    R = pd.DataFrame(index=dfm.index)
    R[MACRO_LABEL["XAU"]] = np.log(dfm["XAU_Close"]).diff()
    R[MACRO_LABEL["DXY"]] = np.log(dfm["DXY"]).diff()
    R[MACRO_LABEL["TNX"]] = dfm["TNX"].diff()
    R[MACRO_LABEL["SPX"]] = np.log(dfm["SPX"]).diff()
    R[MACRO_LABEL["HG"]] = np.log(dfm["HG"]).diff()
    R[MACRO_LABEL["VIX"]] = np.log(dfm["VIX"]).diff()
    return R.replace([np.inf, -np.inf], np.nan).dropna()


def make_levels(dfm: pd.DataFrame) -> pd.DataFrame:
    L = pd.DataFrame(index=dfm.index)
    L[MACRO_LABEL["XAU"]] = dfm["XAU_Close"]
    for k in ["DXY", "TNX", "SPX", "HG", "VIX"]:
        L[MACRO_LABEL[k]] = dfm[k]
    return L


def pct_change(s: pd.Series, n: int) -> float:
    return float(s.iloc[-1] / s.iloc[-n - 1] - 1.0) * 100.0 if len(s) > n else float("nan")


# ---------------------------------------------------------------------------
# 4. WALK-FORWARD: VAR vs MLP
# ---------------------------------------------------------------------------
def build_supervised(dfm: pd.DataFrame, n_lags: int):
    """X[p] usa informacion hasta p (con n_lags rezagos); y[p] = retorno log del Oro en p+1."""
    F = pd.DataFrame(index=dfm.index)
    pc = dfm["XAU_Close"].shift(1)
    F["g_open"] = np.log(dfm["XAU_Open"] / pc)
    F["g_high"] = np.log(dfm["XAU_High"] / pc)
    F["g_low"] = np.log(dfm["XAU_Low"] / pc)
    F["g_close"] = np.log(dfm["XAU_Close"] / pc)
    vol = dfm["XAU_Volume"].replace(0, np.nan)
    F["g_vol"] = np.log(vol / vol.rolling(20, min_periods=5).mean())
    F["dxy"] = np.log(dfm["DXY"]).diff()
    F["tnx"] = dfm["TNX"].diff()
    F["spx"] = np.log(dfm["SPX"]).diff()
    F["hg"] = np.log(dfm["HG"]).diff()
    F["vix"] = np.log(dfm["VIX"]).diff()
    F = F.replace([np.inf, -np.inf], np.nan)
    F["g_vol"] = F["g_vol"].fillna(0.0).clip(-3, 3)
    X = pd.concat([F.shift(k).add_suffix(f"_l{k}") for k in range(n_lags)], axis=1)
    y = F["g_close"].shift(-1)
    return X, y


def fit_mlp(X: np.ndarray, y: np.ndarray, hidden: tuple, seeds: int):
    sx = MinMaxScaler().fit(X)
    sy = MinMaxScaler().fit(y.reshape(-1, 1))
    Xs, ys = sx.transform(X), sy.transform(y.reshape(-1, 1)).ravel()
    models = []
    for s in range(seeds):
        m = MLPRegressor(hidden_layer_sizes=hidden, activation="tanh", solver="adam", alpha=1e-2,
                         learning_rate_init=1e-3, max_iter=300, early_stopping=False, random_state=s)
        m.fit(Xs, ys)
        models.append(m)
    return sx, sy, models


def predict_mlp(pack, X: np.ndarray) -> np.ndarray:
    sx, sy, models = pack
    Xs = np.clip(sx.transform(X), -0.5, 1.5)
    p = np.mean([m.predict(Xs) for m in models], axis=0)
    return sy.inverse_transform(p.reshape(-1, 1)).ravel()


def _metrics(actual, pred, a_ret, p_ret) -> Dict:
    ok = ~(np.isnan(pred) | np.isnan(p_ret))
    actual, pred, a_ret, p_ret = actual[ok], pred[ok], a_ret[ok], p_ret[ok]
    return {"RMSE (USD)": float(np.sqrt(np.mean((actual - pred) ** 2))),
            "MAE (USD)": float(np.mean(np.abs(actual - pred))),
            "Precision direccional (%)": float(np.mean(np.sign(p_ret) == np.sign(a_ret)) * 100.0)}


@st.cache_resource(show_spinner="Walk-forward: reentrenando VAR y MLP paso a paso (cache de servidor)...")
def run_walk_forward(_dfm: pd.DataFrame, data_key: str, test_size: int, train_window: int,
                     retrain_every: int, var_maxlags: int, ic: str, n_lags: int,
                     hidden: tuple, seeds: int) -> Dict:
    dfm = _dfm
    N = len(dfm)
    s = N - test_size
    tw = int(min(train_window, s - n_lags - 40))
    if tw < 250:
        raise ValueError("Historia insuficiente: aumente los anos de historia o reduzca el periodo de prueba.")
    close = dfm["XAU_Close"].values.astype(float)
    lnp = np.log(dfm[["XAU_Open", "XAU_High", "XAU_Low", "XAU_Close"]].values.astype(float))
    R = np.vstack([np.full((1, 4), np.nan), np.diff(lnp, axis=0)])      # R[t]: retorno log en t
    X_df, y_ser = build_supervised(dfm, n_lags)
    Xv, yv = X_df.values, y_ser.values
    xok, yok = ~np.isnan(Xv).any(axis=1), ~np.isnan(yv)

    # --- Lag del VAR (se selecciona una vez sobre la primera ventana) y ADF ---
    p0 = s - 1
    first = R[max(1, p0 + 1 - tw):p0 + 1]
    try:
        sel = VAR(first).select_order(maxlags=var_maxlags)
        ic_df = pd.DataFrame({k.upper(): v for k, v in sel.ics.items()})
        ic_df.index.name = "Lag"
        k = int(max(1, sel.selected_orders.get(ic, 1)))
    except Exception:
        ic_df, k = pd.DataFrame(), 1
    adf_rows = []
    for j, nm in enumerate(["Open", "High", "Low", "Close"]):
        for tag, ser in [(f"ln({nm}) nivel", lnp[max(0, p0 - tw):p0 + 1, j]),
                         (f"retorno {nm}", first[:, j])]:
            try:
                stat, pv = adfuller(ser, autolag="AIC")[:2]
                adf_rows.append({"Serie": tag, "ADF": round(float(stat), 3), "p-valor": round(float(pv), 4),
                                 "Estacionaria (5%)": "SI" if pv < 0.05 else "NO"})
            except Exception:
                pass

    # --- Bucle walk-forward: en p se entrena con datos <= p y se pronostica el cierre de p+1 ---
    rows, var_res, pack, n_fits = [], None, None, 0
    for p in range(s - 1, N):
        step = p - (s - 1)
        if var_res is None or pack is None or step % retrain_every == 0 or p == N - 1:
            try:
                var_res = VAR(R[max(1, p + 1 - tw):p + 1]).fit(k)
            except Exception:
                if var_res is None:
                    raise
            tr = np.arange(max(0, p - tw), p)                  # features en q, target en q+1 <= p
            tr = tr[xok[tr] & yok[tr]]
            if len(tr) < 100:
                raise ValueError("Muestra de entrenamiento insuficiente para la red neuronal.")
            pack = fit_mlp(Xv[tr], yv[tr], hidden, seeds)
            n_fits += 1
        v = float(var_res.forecast(R[p + 1 - k:p + 1], 1)[0, 3])
        m = float(predict_mlp(pack, Xv[[p]])[0]) if xok[p] else np.nan
        rows.append((dfm.index[p], v, m))
    fc = pd.DataFrame(rows, columns=["date", "VAR_ret", "NN_ret"]).set_index("date")   # indexado por fecha de senal

    # --- Evaluacion con alineacion exacta de fechas ---
    test_data = dfm.iloc[s:]
    sig = fc.iloc[:-1]                                         # senales de s-1 .. N-2 -> objetivos s .. N-1
    ev = pd.DataFrame(index=test_data.index)
    ev["Real"] = dfm["XAU_Close"].reindex(ev.index)            # precio real por indice exacto
    prev = close[s - 1:N - 1]
    ev["Prev"] = prev
    ev["VAR_ret"], ev["NN_ret"] = sig["VAR_ret"].values, sig["NN_ret"].values
    ev["VAR"] = prev * np.exp(ev["VAR_ret"].values)
    ev["NN"] = prev * np.exp(ev["NN_ret"].values)
    ev["A_ret"] = np.log(ev["Real"].values / prev)
    ev["RW_ret"] = np.log(prev / close[s - 2:N - 2])
    aligned = bool(ev.index.equals(test_data.index) and not ev["Real"].isna().any()
                   and np.isclose(ev["Real"].iloc[0], dfm.loc[test_data.index[0], "XAU_Close"]))

    a = ev["Real"].values
    metrics = pd.DataFrame({
        "VAR (walk-forward)": _metrics(a, ev["VAR"].values, ev["A_ret"].values, ev["VAR_ret"].values),
        "Red Neuronal (walk-forward)": _metrics(a, ev["NN"].values, ev["A_ret"].values, ev["NN_ret"].values),
        "Random Walk (ingenuo)": _metrics(a, prev, ev["A_ret"].values, ev["RW_ret"].values),
    }).T

    folds = []
    for i, part in enumerate(np.array_split(np.arange(len(ev)), 5), 1):
        e = ev.iloc[part]
        mv = _metrics(e["Real"].values, e["VAR"].values, e["A_ret"].values, e["VAR_ret"].values)
        mn = _metrics(e["Real"].values, e["NN"].values, e["A_ret"].values, e["NN_ret"].values)
        mr = _metrics(e["Real"].values, e["Prev"].values, e["A_ret"].values, e["RW_ret"].values)
        folds.append({"Bloque": f"{i} ({e.index[0].date()} a {e.index[-1].date()})",
                      "RMSE VAR": round(mv["RMSE (USD)"], 2), "RMSE Red": round(mn["RMSE (USD)"], 2),
                      "RMSE RW": round(mr["RMSE (USD)"], 2),
                      "Dir. VAR %": round(mv["Precision direccional (%)"], 1),
                      "Dir. Red %": round(mn["Precision direccional (%)"], 1)})
    return dict(ev=ev, fc=fc, metrics=metrics, folds=pd.DataFrame(folds), lag=k, ic=ic_df,
                adf=pd.DataFrame(adf_rows), tw=tw, n_fits=n_fits, aligned=aligned,
                var_next=float(close[-1] * np.exp(fc["VAR_ret"].iloc[-1])),
                nn_next=float(close[-1] * np.exp(fc["NN_ret"].iloc[-1])) if not np.isnan(fc["NN_ret"].iloc[-1]) else np.nan)


def build_wf_chart(ev: pd.DataFrame, zoom_last: Optional[int]) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ev.index, y=ev["VAR"], mode="lines", name="Pronostico VAR",
                             line=dict(color=C_VAR, width=2, dash="dot")))
    fig.add_trace(go.Scatter(x=ev.index, y=ev["NN"], mode="lines", name="Red Neuronal",
                             line=dict(color=C_NN, width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=ev.index, y=ev["Real"], mode="lines", name="Precio Real",
                             line=dict(color=C_REAL, width=3)))             # ultima traza = capa superior
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    if zoom_last and len(ev) > zoom_last:
        z = ev.tail(zoom_last)
        lo = float(np.nanmin(z[["Real", "VAR", "NN"]].values))
        hi = float(np.nanmax(z[["Real", "VAR", "NN"]].values))
        pad = (hi - lo) * 0.08 + 1e-9
        fig.update_xaxes(range=[z.index[0], z.index[-1] + pd.Timedelta(days=2)])
        fig.update_yaxes(range=[lo - pad, hi + pad])
    fig.update_layout(template="plotly_dark", height=470, hovermode="x unified",
                      margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=1.1),
                      title="Walk-forward a 1 dia vs precio real (USD/oz)")
    return fig


# ---------------------------------------------------------------------------
# 5. MATRIZ ESTADISTICA: EVENTOS SMC, WIN RATE, EV
# ---------------------------------------------------------------------------
def _simulate(direction: int, entry: float, stop: float, target: float,
              h: np.ndarray, l: np.ndarray, t: int, horizon: int) -> float:
    """1 = TP primero, 0 = SL primero (si ambos en la misma vela, se asume SL), NaN = sin resolver."""
    for j in range(t + 1, min(t + 1 + horizon, len(h))):
        if direction == 1:
            if l[j] <= stop:
                return 0.0
            if h[j] >= target:
                return 1.0
        else:
            if h[j] >= stop:
                return 0.0
            if l[j] <= target:
                return 1.0
    return float("nan")


EVENT_COLS = ["date", "pos", "dir", "type", "entry", "stop", "risk", "outcome", "cvd_ok",
              "model_avail", "model_ok", "live"]


def build_events(df: pd.DataFrame, zones: List[Dict], atr: pd.Series, cvd: pd.Series,
                 var_sig: np.ndarray, nn_sig: np.ndarray, horizon: int) -> pd.DataFrame:
    """Un evento por primer toque de zona (causal): entrada al cierre de la vela de toque, SL estructural, TP = 3R."""
    h, l, c = df["High"].values, df["Low"].values, df["Close"].values
    a, cv, n = atr.values, cvd.values, len(df)
    seen, rows = set(), []
    for z in sorted(zones, key=lambda z: z["avail"]):
        t = z["touch"]
        if t < 0:
            continue
        d = 1 if z["kind"] == "demand" else -1
        if (d == 1 and c[t] < z["bottom"]) or (d == -1 and c[t] > z["top"]) or (t, d) in seen:
            continue
        if np.isnan(a[t]):
            continue
        seen.add((t, d))
        entry = c[t]
        stop = (min(z["bottom"], l[t]) - 0.25 * a[t]) if d == 1 else (max(z["top"], h[t]) + 0.25 * a[t])
        risk = abs(entry - stop)
        if risk < 0.5 * a[t]:
            risk = 0.5 * a[t]
            stop = entry - d * risk
        target = entry + d * RR * risk
        out = _simulate(d, entry, stop, target, h, l, t, horizon)
        cvd_ok = bool(t >= 5 and np.sign(cv[t] - cv[t - 5]) == d)
        mav = not (np.isnan(var_sig[t]) or np.isnan(nn_sig[t]))
        mok = bool(mav and var_sig[t] == d and nn_sig[t] == d)
        rows.append((df.index[t], t, d, z["type"], entry, stop, risk, out, cvd_ok, mav, mok, t == n - 1))
    return pd.DataFrame(rows, columns=EVENT_COLS)


def wilson_lb(w: float, n: int, z: float = 1.96) -> float:
    if n == 0:
        return float("nan")
    p = w / n
    den = 1 + z * z / n
    centre = p + z * z / (2 * n)
    adj = z * np.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return float((centre - adj) / den)


def level_stats(ev: pd.DataFrame, level: int, direction: int) -> Dict:
    sub = ev[ev["outcome"].notna() & (ev["dir"] == direction)]
    if level >= 2:
        sub = sub[sub["cvd_ok"]]
    if level >= 3:
        sub = sub[sub["model_avail"] & sub["model_ok"]]
    n = int(len(sub))
    w = float(sub["outcome"].sum()) if n else 0.0
    wr = w / n if n else float("nan")
    lb = wilson_lb(w, n)
    return dict(n=n, wr=wr, ev=(wr * RR - (1 - wr)) if n else float("nan"),
                ev_lb=(lb * RR - (1 - lb)) if n else float("nan"), lb=lb)


def trade_plan(d: int, price: float, atr: float, zone: Optional[Dict], live_ev: Optional[pd.Series]) -> Optional[Dict]:
    if live_ev is not None:
        entry, stop, src = float(live_ev["entry"]), float(live_ev["stop"]), "Toque de zona en la ultima vela"
    elif zone is not None:
        mid = (zone["top"] + zone["bottom"]) / 2.0
        entry = min(mid, price) if d == 1 else max(mid, price)
        stop = zone["bottom"] - 0.25 * atr if d == 1 else zone["top"] + 0.25 * atr
        src = f"Mitigacion {zone['type']} [{fmt(zone['bottom'])} - {fmt(zone['top'])}]"
    else:
        return None
    risk = abs(entry - stop)
    if risk < 0.5 * atr:
        risk = 0.5 * atr
        stop = entry - d * risk
    return dict(dir=d, entry=entry, stop=stop, risk=risk, tp=entry + d * RR * risk, src=src)


def plan_chart(df: pd.DataFrame, plan: Dict, title: str) -> go.Figure:
    d = df.tail(90)
    fig = go.Figure(go.Candlestick(x=d.index, open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"],
                                   increasing_line_color=C_UP, decreasing_line_color=C_DN, name="GC=F"))
    for nm, lvl, col in [("Entrada", plan["entry"], C_GOLD), ("Stop Loss", plan["stop"], C_DN),
                         ("TP 1:3", plan["tp"], C_UP)]:
        fig.add_hline(y=lvl, line_dash="dash", line_color=col, annotation_text=f"{nm} {lvl:,.1f}",
                      annotation_position="top left", annotation_font_size=10)
    fig.update_xaxes(rangeslider_visible=False, rangebreaks=[dict(bounds=["sat", "mon"])])
    fig.update_layout(template="plotly_dark", height=430, title=title, margin=dict(l=10, r=10, t=50, b=10))
    return fig


# ---------------------------------------------------------------------------
# 6. APLICACION
# ---------------------------------------------------------------------------
CSS = """
<style>
.block-container {padding-top: 1.2rem; max-width: 1500px;}
div[data-testid="stMetric"] {background: rgba(128,128,128,0.08); border: 1px solid rgba(128,128,128,0.25);
    border-radius: 10px; padding: 10px 14px;}
button[data-baseweb="tab"] {font-weight: 600;}
</style>
"""


def main() -> None:
    st.set_page_config(page_title="Gold Quant Desk", page_icon="🥇", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.title("🥇 Gold Quant Desk · Futuros de Oro (GC=F)")
    st.caption("Order Flow · SMC · Macro · Walk-Forward VAR vs Red Neuronal · Matriz de probabilidad con R:R 1:3. "
               "Datos de Yahoo Finance con retraso; material analitico, no asesoria financiera.")

    with st.sidebar:
        st.header("Parametros")
        years = st.slider("Historia (anos)", 3, 10, 6)
        with st.expander("Order Flow y SMC", expanded=True):
            view_bars = st.slider("Velas visibles", 60, 500, 150, 10)
            disp_atr = st.slider("Desplazamiento OB (x ATR)", 1.0, 3.0, 1.5, 0.1)
            show_mit = st.checkbox("Mostrar zonas mitigadas", value=False)
        with st.expander("Walk-forward", expanded=True):
            test_size = st.slider("Sesiones de prueba", 120, 500, 250, 10)
            train_window = st.slider("Ventana de entrenamiento rodante", 400, 1000, 750, 50)
            retrain_every = st.select_slider("Reentrenar cada (sesiones)", options=[1, 5, 10, 20], value=10)
            ic = st.selectbox("Criterio de lags VAR", ["aic", "bic", "hqic", "fpe"], index=0)
            var_maxlags = st.slider("Max lags VAR", 2, 10, 6)
            n_lags = st.slider("Rezagos de la red (dias)", 1, 5, 3)
            hidden_name = st.selectbox("Arquitectura MLP", ["(16, 8)", "(32, 16)", "(16,)"], index=0)
            seeds = st.slider("Semillas del ensamble", 1, 3, 2)
        with st.expander("Matriz estadistica y riesgo", expanded=True):
            horizon = st.slider("Horizonte de resolucion (sesiones)", 10, 40, 20)
            min_n = st.slider("Muestra minima para operar (n)", 5, 60, 15)
            prox = st.slider("Proximidad a zona (x ATR)", 0.1, 1.5, 0.5, 0.1)
            capital = st.number_input("Capital (USD)", min_value=1000.0, value=50000.0, step=1000.0)
            risk_pct = st.number_input("Riesgo por operacion (%)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
        if st.button("Actualizar datos y modelos"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()
    hidden = {"(16, 8)": (16, 8), "(32, 16)": (32, 16), "(16,)": (16,)}[hidden_name]

    # ---- Datos ----
    with st.spinner("Descargando datos de mercado..."):
        df_raw, df_model, missing = load_data(years)
    if df_raw.empty:
        st.error("No se pudo descargar GC=F desde Yahoo Finance. Pulse 'Actualizar datos' o reintente en unos minutos.")
        st.stop()
    if missing:
        st.error(f"Faltan series macro: {', '.join(missing)}. Pulse 'Actualizar datos' o reintente.")
        st.stop()
    if len(df_model) < 700:
        st.error(f"Muestra alineada insuficiente ({len(df_model)} sesiones). Aumente los anos de historia.")
        st.stop()

    daily = df_model[["XAU_Open", "XAU_High", "XAU_Low", "XAU_Close", "XAU_Volume"]].copy()
    daily.columns = ["Open", "High", "Low", "Close", "Volume"]
    df = add_order_flow(daily)
    atr = compute_atr(df)
    atr_now = float(atr.iloc[-1])
    price = float(df["Close"].iloc[-1])
    zones = detect_order_blocks(df, disp_atr) + detect_fvg(df)

    # ---- Walk-forward (cacheado) ----
    wf, wf_err = None, None
    try:
        key = f"{df_model.index[-1].date()}_{len(df_model)}_{years}"
        wf = run_walk_forward(df_model, key, test_size, train_window, retrain_every, var_maxlags, ic,
                              n_lags, hidden, seeds)
    except Exception as e:
        wf_err = str(e)

    # ---- KPIs ----
    chg = pct_change(df["Close"], 1)
    k = st.columns(5)
    k[0].metric("GC=F cierre", fmt(price), f"{chg:+.2f}%")
    k[1].metric("ATR(14)", fmt(atr_now))
    k[2].metric("DXY 20d", f"{pct_change(df_model['DXY'], 20):+.2f}%")
    k[3].metric("10Y (pb, 20d)", f"{(df_model['TNX'].iloc[-1] - df_model['TNX'].iloc[-21]) * 100:+.0f}")
    k[4].metric("VIX", f"{df_model['VIX'].iloc[-1]:.1f}")
    st.caption(f"Ultima sesion: {df_model.index[-1].date()} · {len(df_model)} sesiones alineadas · {cvd_divergence(df, 20)}")

    tab1, tab2, tab3, tab4 = st.tabs(["① Order Flow & SMC", "② Matriz Macro",
                                      "③ VAR vs Red Neuronal", "④ Escenarios & Riesgo"])

    # ================= TAB 1 =================
    with tab1:
        live = [z for z in zones if not z["mitigated"]]
        c = st.columns(5)
        c[0].metric("Delta ultima vela", f"{df['Delta'].iloc[-1]:,.0f}")
        c[1].metric("CVD cambio 20d", f"{df['CVD'].iloc[-1] - df['CVD'].iloc[-21]:,.0f}")
        c[2].metric("OB vigentes (D/S)", f"{sum(z['type'] == 'OB' and z['kind'] == 'demand' for z in live)} / "
                                          f"{sum(z['type'] == 'OB' and z['kind'] == 'supply' for z in live)}")
        c[3].metric("FVG vigentes (D/S)", f"{sum(z['type'] == 'FVG' and z['kind'] == 'demand' for z in live)} / "
                                           f"{sum(z['type'] == 'FVG' and z['kind'] == 'supply' for z in live)}")
        c[4].metric("Zonas totales", f"{len(zones)}")
        st.plotly_chart(build_order_flow_chart(df, zones, view_bars, show_mit))
        near = sorted(live, key=lambda z: zone_distance_atr(z, price, atr_now) if (
            (z["kind"] == "demand" and z["bottom"] < price) or (z["kind"] == "supply" and z["top"] > price))
            else 1e9)[:8]
        near = [z for z in near if (z["kind"] == "demand" and z["bottom"] < price) or
                (z["kind"] == "supply" and z["top"] > price)]
        if near:
            st.markdown("**Zonas vigentes mas cercanas**")
            st.dataframe(pd.DataFrame([{
                "Tipo": z["type"], "Lado": "Demanda (largo)" if z["kind"] == "demand" else "Oferta (corto)",
                "Inferior": fmt(z["bottom"]), "Superior": fmt(z["top"]),
                "Distancia (ATR)": round(zone_distance_atr(z, price, atr_now), 2),
                "Formada": z["start"].date()} for z in near]), hide_index=True)
        st.caption("Volume Delta estimado por vela: Vol·(C−L)/(H−L) − Vol·(H−C)/(H−L); CVD = acumulado (desde el inicio de la muestra). "
                   "OB: ultima vela contraria previa a un desplazamiento con ruptura de estructura. FVG: hueco de 3 velas. "
                   "Se dibujan las zonas aun no mitigadas (sin retorno del precio).")

    # ================= TAB 2 =================
    with tab2:
        mode = st.radio("Base de calculo", ["Retornos diarios", "Niveles"], horizontal=True)
        base = make_returns(df_model) if mode == "Retornos diarios" else make_levels(df_model)
        win = st.slider("Ventana (sesiones)", 60, min(750, len(base)), min(252, len(base)), 10)
        corr = base.tail(win).corr()
        st.dataframe(corr.round(3))
        heat = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, zmin=-1, zmax=1,
                                    colorscale="RdBu", reversescale=True, text=np.round(corr.values, 2),
                                    texttemplate="%{text}"))
        heat.update_layout(template="plotly_dark", height=480, margin=dict(l=10, r=10, t=40, b=10),
                           title=f"Correlacion ({mode.lower()}, ultimas {win} sesiones)")
        st.plotly_chart(heat)
        R = make_returns(df_model)
        rw = st.slider("Ventana de correlacion rodante vs Oro", 20, 120, 60, 5)
        rf = go.Figure()
        for col in [c for c in R.columns if c != MACRO_LABEL["XAU"]]:
            rc = R[MACRO_LABEL["XAU"]].rolling(rw).corr(R[col])
            rf.add_trace(go.Scatter(x=rc.index, y=rc, mode="lines", name=col))
        rf.add_hline(y=0, line_dash="dot", line_color="#90a4ae")
        rf.update_layout(template="plotly_dark", height=380, yaxis=dict(range=[-1, 1]),
                         margin=dict(l=10, r=10, t=40, b=10), title=f"Correlacion rodante de retornos vs Oro ({rw} sesiones)")
        st.plotly_chart(rf)

    # ================= TAB 3 =================
    with tab3:
        if wf is None:
            st.error(f"Walk-forward no disponible: {wf_err}")
        else:
            ev = wf["ev"]
            st.markdown(f"**Walk-forward:** {len(ev)} sesiones de prueba ({ev.index[0].date()} a {ev.index[-1].date()}) · "
                        f"ventana rodante {wf['tw']} sesiones · reentrenamiento cada {retrain_every} · "
                        f"{wf['n_fits']} reentrenamientos · lag VAR ({ic.upper()}) = {wf['lag']} · "
                        f"alineacion de fechas con precio real: {'OK' if wf['aligned'] else 'REVISAR'}")
            zoom = st.radio("Zoom", ["Ultimas 60 sesiones", "Todo el periodo de prueba"], horizontal=True)
            st.plotly_chart(build_wf_chart(ev, 60 if zoom.startswith("Ultimas") else None))

            m = wf["metrics"].copy()
            m["RMSE (USD)"] = m["RMSE (USD)"].map(lambda v: f"{v:,.2f}")
            m["MAE (USD)"] = m["MAE (USD)"].map(lambda v: f"{v:,.2f}")
            m["Precision direccional (%)"] = m["Precision direccional (%)"].map(lambda v: f"{v:.1f}")
            st.subheader("Cuadro comparativo out-of-sample")
            st.dataframe(m)
            st.caption("Cada pronostico usa solo informacion hasta el dia previo y se ancla al ultimo cierre real. "
                       "El Random Walk (cierre previo) es el benchmark: un modelo que no lo supera no aporta ventaja. "
                       "Una precision direccional cercana a 50% es consistente con eficiencia debil.")
            c = st.columns(3)
            c[0].metric("Cierre actual", fmt(price))
            c[1].metric("VAR t+1", fmt(wf["var_next"]), f"{(wf['var_next'] / price - 1) * 100:+.2f}%")
            if not np.isnan(wf["nn_next"]):
                c[2].metric("Red Neuronal t+1", fmt(wf["nn_next"]), f"{(wf['nn_next'] / price - 1) * 100:+.2f}%")
            with st.expander("Validacion temporal por bloques (5 folds consecutivos)"):
                st.dataframe(wf["folds"], hide_index=True)
            with st.expander("Ultimas observaciones del walk-forward (verificacion de alineacion)"):
                tail = ev[["Real", "VAR", "NN", "Prev"]].tail(10).round(2)
                tail.index = tail.index.date
                st.dataframe(tail)
            with st.expander("Estacionariedad (ADF) y seleccion de lags"):
                st.dataframe(wf["adf"], hide_index=True)
                if not wf["ic"].empty:
                    st.dataframe(wf["ic"])

    # ================= TAB 4 =================
    with tab4:
        n_df = len(df)
        var_sig, nn_sig = np.full(n_df, np.nan), np.full(n_df, np.nan)
        if wf is not None:
            pos = df.index.get_indexer(wf["fc"].index)
            ok = pos >= 0
            var_sig[pos[ok]] = np.sign(wf["fc"]["VAR_ret"].values[ok])
            nn_sig[pos[ok]] = np.sign(wf["fc"]["NN_ret"].values[ok])
        events = build_events(df, zones, atr, df["CVD"], var_sig, nn_sig, horizon)

        # ---- Estado actual de confluencia por direccion ----
        cvd5 = float(df["CVD"].iloc[-1] - df["CVD"].iloc[-6])
        state = {}
        for d, kind in [(1, "demand"), (-1, "supply")]:
            live_ev = None
            if not events.empty:
                m_ = events[(events["live"]) & (events["dir"] == d)]
                live_ev = m_.iloc[0] if len(m_) else None
            z = nearest_zone(zones, price, kind)
            dist = zone_distance_atr(z, price, atr_now) if z is not None else float("nan")
            smc = live_ev is not None or (z is not None and dist <= prox)
            cvd_ok = bool(np.sign(cvd5) == d)
            mod_ok = False
            if wf is not None and not np.isnan(wf["nn_next"]):
                mod_ok = bool(np.sign(wf["var_next"] - price) == d and np.sign(wf["nn_next"] - price) == d)
            level = 0 if not smc else (1 if not cvd_ok else (2 if not mod_ok else 3))
            cond = ("Toque en ultima vela" if live_ev is not None else
                    (f"Zona a {dist:.2f} ATR" if z is not None else "Sin zona vigente"))
            state[d] = dict(level=level, cond=cond, plan=trade_plan(d, price, atr_now, z if smc else None, live_ev),
                            smc=smc, cvd=cvd_ok, mod=mod_ok)

        # ---- Tabla de probabilidad / EV ----
        names = {1: "Largo", -1: "Corto"}
        lvl_names = {1: "N1 · SMC", 2: "N2 · SMC + CVD", 3: "N3 · SMC + CVD + Modelos"}
        rows, qualified = [], []
        for L in (1, 2, 3):
            for d in (1, -1):
                s_ = level_stats(events, L, d) if not events.empty else dict(n=0, wr=np.nan, ev=np.nan, ev_lb=np.nan, lb=np.nan)
                active = state[d]["level"] >= L
                go_ = bool(active and s_["n"] >= min_n and s_["ev"] > 0)
                if go_:
                    qualified.append((L, s_["ev"], d))
                rows.append({
                    "Nivel de confluencia": lvl_names[L], "Direccion": names[d], "Muestras (n)": s_["n"],
                    "Win Rate %": "n/d" if s_["n"] == 0 else f"{s_['wr'] * 100:.1f}",
                    "WR inf. 95%": "n/d" if s_["n"] == 0 else f"{s_['lb'] * 100:.1f}",
                    "EV (R) 1:3": "n/d" if s_["n"] == 0 else f"{s_['ev']:+.2f}",
                    "EV conservador (R)": "n/d" if s_["n"] == 0 else f"{s_['ev_lb']:+.2f}",
                    "Condicion actual": state[d]["cond"] if L == 1 else ("Cumple" if active else "No cumple"),
                    "Operacion": "SI" if go_ else "NO"})
        table = pd.DataFrame(rows)

        if qualified:
            best = sorted(qualified, key=lambda q: (q[0], q[1]), reverse=True)[0]
            st.success(f"Operacion habilitada: {names[best[2]]} con confluencia N{best[0]} · EV {best[1]:+.2f} R · R:R 1:3")
            sel_dir = best[2]
        else:
            st.warning("Sin operacion: ninguna confluencia activa combina EV positivo y muestra minima.")
            sel_dir = max(state, key=lambda d: state[d]["level"])
        st.dataframe(table, hide_index=True)
        st.caption(f"Break-even con R:R 1:3 = 25% de aciertos. EV = WR·3 − (1−WR)·1 (en unidades de riesgo R). "
                   f"Backtest causal de {len(events)} eventos de primer toque de zona OB/FVG sobre {years} anos "
                   f"(entrada al cierre de la vela de toque, SL estructural a 0.25 ATR de la zona, TP = 3R, horizonte {horizon} sesiones; "
                   "si SL y TP caen en la misma vela se asume SL; eventos sin resolver se excluyen). "
                   "N3 solo existe dentro del periodo out-of-sample del walk-forward, por lo que su muestra es menor. "
                   "'EV conservador' usa el limite inferior de Wilson al 95%. Operacion = SI exige confluencia activa hoy, "
                   f"n ≥ {min_n} y EV > 0. Resultados pasados no garantizan resultados futuros.")

        # ---- Grafico de WR y plan ----
        left, right = st.columns([1, 1])
        with left:
            bar_x, bar_y = [], []
            for L in (1, 2, 3):
                for d in (1, -1):
                    s_ = level_stats(events, L, d) if not events.empty else dict(n=0, wr=np.nan)
                    bar_x.append(f"N{L} {names[d]} (n={s_['n']})")
                    bar_y.append(0.0 if s_["n"] == 0 else s_["wr"] * 100)
            bf = go.Figure(go.Bar(x=bar_x, y=bar_y, marker_color=[C_UP if i % 2 == 0 else C_DN for i in range(6)]))
            bf.add_hline(y=25, line_dash="dash", line_color=C_GOLD, annotation_text="Break-even 25%")
            bf.update_layout(template="plotly_dark", height=430, title="Win Rate por nivel de confluencia (%)",
                             margin=dict(l=10, r=10, t=50, b=10), yaxis=dict(range=[0, 100]))
            st.plotly_chart(bf)
        with right:
            pick = st.radio("Plan a visualizar", ["Largo", "Corto"], horizontal=True,
                            index=0 if sel_dir == 1 else 1)
            d_sel = 1 if pick == "Largo" else -1
            plan = state[d_sel]["plan"]
            if plan is None:
                st.info("No hay zona vigente en esa direccion para construir un plan.")
            else:
                st.plotly_chart(plan_chart(df, plan, f"Plan {pick}: {plan['src']}"))

        if state[d_sel]["plan"] is not None:
            plan = state[d_sel]["plan"]
            risk_usd = capital * risk_pct / 100.0
            gc = int(risk_usd // (plan["risk"] * 100.0)) if plan["risk"] > 0 else 0
            mgc = int(risk_usd // (plan["risk"] * 10.0)) if plan["risk"] > 0 else 0
            p = st.columns(6)
            p[0].metric("Entrada", fmt(plan["entry"]))
            p[1].metric("Stop Loss", fmt(plan["stop"]))
            p[2].metric("TP (1:3)", fmt(plan["tp"]))
            p[3].metric("Riesgo (pts)", fmt(plan["risk"]))
            p[4].metric("Contratos GC (100 oz)", gc)
            p[5].metric("Contratos MGC (10 oz)", mgc)
            st.caption(f"Confluencia actual {pick}: SMC {'✔' if state[d_sel]['smc'] else '✘'} · "
                       f"CVD(5d) {'✔' if state[d_sel]['cvd'] else '✘'} · "
                       f"Modelos alineados {'✔' if state[d_sel]['mod'] else '✘'}. "
                       f"Riesgo por operacion: {fmt(risk_usd)} USD.")


if __name__ == "__main__":
    main()

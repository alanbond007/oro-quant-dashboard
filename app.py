"""
GOLD QUANT DESK  |  Futuros de Oro (GC=F)
---------------------------------------------------------------------------
Dashboard institucional: Order Flow + SMC + Matriz Macro + VAR vs MLP + Escenarios.
Despliegue: GitHub -> Streamlit Community Cloud (archivo principal: app.py).

Notas de diseno
- Datos: Yahoo Finance (yfinance). Cache de descarga con @st.cache_data.
- Modelos: entrenamiento cacheado con @st.cache_resource (VAR + MLP ligero).
- Volume Delta: estimacion por vela (no es tick data); es un proxy, no CVD real de L2.
- Ningun contenido constituye asesoria financiera.
"""

import time
import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
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
MACRO_TICKERS = {
    "TNX": "^TNX",        # Rendimiento Tesoro 10Y
    "IRX": "^IRX",        # T-Bill 13 semanas (proxy de tasa corta; Yahoo no ofrece 2Y)
    "TIP": "TIP",         # ETF TIPS (proxy de expectativas de inflacion / tasas reales)
    "HG": "HG=F",         # Cobre (Doctor Copper)
    "SPX": "^GSPC",       # S&P 500
    "DXY": "DX-Y.NYB",    # Indice del Dolar
    "VIX": "^VIX",        # Volatilidad / estres
    "SI": "SI=F",         # Plata
}
LABELS = {
    "Oro": "Oro",
    "TNX": "Tasa 10Y",
    "SPREAD": "Curva 10Y-3M",
    "TIP": "TIPS (inflacion)",
    "HG": "Cobre (PIB)",
    "SPX": "S&P 500",
    "DXY": "Dolar DXY",
    "VIX": "VIX",
    "SI": "Plata",
}
TF_CONFIG = {
    "Diario (1D)": ("1y", "1d"),
    "Horario (1H)": ("60d", "1h"),
    "15 minutos (15M)": ("30d", "15m"),
}
COLOR_UP = "#26a69a"
COLOR_DN = "#ef5350"
GOLD_COLOR = "#d4af37"


def usd(x: float) -> str:
    return f"{x:,.2f}"


# ---------------------------------------------------------------------------
# 1. EXTRACCION DE DATOS
# ---------------------------------------------------------------------------
def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert("America/New_York").tz_localize(None)
    df = df.copy()
    df.index = idx
    return df


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_ohlcv(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """Descarga OHLCV de un ticker con reintentos. Devuelve DataFrame vacio si falla."""
    df = None
    for attempt in range(3):
        try:
            df = yf.Ticker(ticker).history(
                period=period, interval=interval, auto_adjust=False, actions=False
            )
            if df is not None and not df.empty:
                break
        except Exception:
            df = None
        time.sleep(1.0 + attempt)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = _normalize_index(df[cols])
    if "Volume" not in df.columns:
        df["Volume"] = 0.0
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df["Volume"] = df["Volume"].fillna(0.0).astype(float)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def load_macro_matrix(years: int) -> pd.DataFrame:
    """Matriz macro extendida alineada al calendario del Oro (diario)."""
    period = f"{years}y"
    gold = fetch_ohlcv(GOLD, period, "1d")
    if gold.empty:
        return pd.DataFrame()
    M = gold.copy()
    M.index = M.index.normalize()
    M = M[~M.index.duplicated(keep="last")]
    for key, tk in MACRO_TICKERS.items():
        d = fetch_ohlcv(tk, period, "1d")
        if d.empty:
            M[key] = np.nan
            continue
        s = d["Close"].copy()
        s.index = s.index.normalize()
        s = s[~s.index.duplicated(keep="last")]
        M[key] = s.reindex(M.index).ffill(limit=3)
    # Yahoo ha cotizado ^TNX/^IRX tanto en % como en decimas de %.
    for k in ["TNX", "IRX"]:
        if M[k].notna().any() and M[k].median() > 25:
            M[k] = M[k] / 10.0
    M["SPREAD"] = M["TNX"] - M["IRX"]
    required = ["Open", "High", "Low", "Close", "TNX", "SPREAD", "TIP", "HG", "SPX", "DXY", "VIX", "SI"]
    M = M.dropna(subset=required)
    return M


# ---------------------------------------------------------------------------
# 2. ORDER FLOW / SMC / VOLUME PROFILE
# ---------------------------------------------------------------------------
def compute_atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["Close"].shift(1)
    tr = pd.concat(
        [df["High"] - df["Low"], (df["High"] - pc).abs(), (df["Low"] - pc).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def add_order_flow(df: pd.DataFrame) -> pd.DataFrame:
    """Volume Delta estimado: volumen repartido segun la posicion del cierre dentro del rango."""
    out = df.copy()
    rng = (out["High"] - out["Low"]).replace(0, np.nan)
    buy = out["Volume"] * (out["Close"] - out["Low"]) / rng
    sell = out["Volume"] * (out["High"] - out["Close"]) / rng
    out["Delta"] = (buy - sell).fillna(0.0)
    out["CVD"] = out["Delta"].cumsum()
    return out


def detect_fvg(df: pd.DataFrame, min_atr_frac: float = 0.10) -> List[Dict]:
    """Fair Value Gaps de 3 velas. Marca si fueron mitigados (precio retorno a la zona)."""
    h, l = df["High"].values, df["Low"].values
    n = len(df)
    if n < 5:
        return []
    atr = compute_atr(df).values
    idx = df.index
    suf_min = np.minimum.accumulate(l[::-1])[::-1]
    suf_max = np.maximum.accumulate(h[::-1])[::-1]
    zones = []
    for i in range(2, n):
        a = atr[i - 1]
        thr = (0.0 if np.isnan(a) else a) * min_atr_frac
        gap_bull = l[i] - h[i - 2]
        gap_bear = l[i - 2] - h[i]
        if gap_bull > 0 and gap_bull >= thr:
            top, bot = l[i], h[i - 2]
            mit = (i + 1 < n) and (suf_min[i + 1] <= top)
            zones.append(dict(type="FVG", kind="demand", start=idx[i - 1], top=float(top),
                              bottom=float(bot), mitigated=bool(mit), pos=i))
        elif gap_bear > 0 and gap_bear >= thr:
            top, bot = l[i - 2], h[i]
            mit = (i + 1 < n) and (suf_max[i + 1] >= bot)
            zones.append(dict(type="FVG", kind="supply", start=idx[i - 1], top=float(top),
                              bottom=float(bot), mitigated=bool(mit), pos=i))
    return zones


def detect_order_blocks(df: pd.DataFrame, disp_atr: float = 1.5, lookback: int = 5) -> List[Dict]:
    """
    Order Block = ultima vela contraria previa a una vela de desplazamiento
    (cuerpo >= disp_atr * ATR) que rompe la estructura de las 3 velas previas.
    """
    o, h, l, c = (df[k].values for k in ["Open", "High", "Low", "Close"])
    n = len(df)
    if n < 20:
        return []
    atr = compute_atr(df).values
    idx = df.index
    suf_min = np.minimum.accumulate(l[::-1])[::-1]
    suf_max = np.maximum.accumulate(h[::-1])[::-1]
    zones, used = [], set()
    for k in range(4, n - 1):
        a = atr[k - 1]
        if np.isnan(a) or a <= 0:
            continue
        body = abs(c[k] - o[k])
        if body < disp_atr * a:
            continue
        if c[k] > o[k] and c[k] > np.max(h[k - 3:k]):          # desplazamiento alcista + BOS
            for j in range(k - 1, max(k - 1 - lookback, -1), -1):
                if c[j] < o[j] and j not in used:
                    used.add(j)
                    mit = suf_min[k + 1] <= h[j]
                    zones.append(dict(type="OB", kind="demand", start=idx[j], top=float(h[j]),
                                      bottom=float(l[j]), mitigated=bool(mit), pos=j))
                    break
        elif c[k] < o[k] and c[k] < np.min(l[k - 3:k]):         # desplazamiento bajista + BOS
            for j in range(k - 1, max(k - 1 - lookback, -1), -1):
                if c[j] > o[j] and j not in used:
                    used.add(j)
                    mit = suf_max[k + 1] >= l[j]
                    zones.append(dict(type="OB", kind="supply", start=idx[j], top=float(h[j]),
                                      bottom=float(l[j]), mitigated=bool(mit), pos=j))
                    break
    return zones


def volume_profile(df: pd.DataFrame, bins: int = 48, va_pct: float = 0.70) -> Dict:
    """VPVR: volumen distribuido proporcionalmente al rango de cada vela. POC, VAH, VAL."""
    lo, hi = float(df["Low"].min()), float(df["High"].max())
    if hi <= lo:
        hi = lo + 1.0
    edges = np.linspace(lo, hi, bins + 1)
    lows = df["Low"].values[:, None]
    highs = df["High"].values[:, None]
    vol = df["Volume"].fillna(0).values[:, None].astype(float)
    e_lo, e_hi = edges[:-1][None, :], edges[1:][None, :]
    overlap = np.clip(np.minimum(highs, e_hi) - np.maximum(lows, e_lo), 0, None)
    rng = np.where((highs - lows) <= 0, 1e-9, highs - lows)
    prof = (vol * overlap / rng).sum(axis=0)
    centers = (edges[:-1] + edges[1:]) / 2.0
    poc_i = int(prof.argmax())
    target = va_pct * prof.sum()
    lo_i = hi_i = poc_i
    acc = prof[poc_i]
    while acc < target and (lo_i > 0 or hi_i < bins - 1):
        up = prof[hi_i + 1] if hi_i < bins - 1 else -1.0
        dn = prof[lo_i - 1] if lo_i > 0 else -1.0
        if up >= dn:
            hi_i += 1
            acc += prof[hi_i]
        else:
            lo_i -= 1
            acc += prof[lo_i]
    return dict(centers=centers, profile=prof, poc=float(centers[poc_i]),
                vah=float(edges[hi_i + 1]), val=float(edges[lo_i]), poc_idx=poc_i)


def cvd_divergence(df: pd.DataFrame, window: int = 20) -> str:
    if len(df) <= window + 1:
        return "Datos insuficientes para evaluar divergencia."
    p0, p1 = df["Close"].iloc[-window - 1], df["Close"].iloc[-1]
    c0, c1 = df["CVD"].iloc[-window - 1], df["CVD"].iloc[-1]
    dp, dc = p1 - p0, c1 - c0
    if dp > 0 and dc > 0:
        return "Convergencia alcista: precio y CVD al alza (acumulacion confirmada)."
    if dp < 0 and dc < 0:
        return "Convergencia bajista: precio y CVD a la baja (distribucion confirmada)."
    if dp > 0 and dc < 0:
        return "DIVERGENCIA BAJISTA: precio sube con CVD cayendo (posible distribucion institucional)."
    return "DIVERGENCIA ALCISTA: precio cae con CVD subiendo (posible acumulacion institucional)."


def nearest_zone(zones: List[Dict], price: float, kind: str) -> Optional[Dict]:
    cand = [z for z in zones if z["kind"] == kind and not z["mitigated"]]
    if kind == "demand":
        cand = [z for z in cand if z["bottom"] < price]
        key = lambda z: (max(0.0, price - z["top"]), 0 if z["type"] == "OB" else 1)
    else:
        cand = [z for z in cand if z["top"] > price]
        key = lambda z: (max(0.0, z["bottom"] - price), 0 if z["type"] == "OB" else 1)
    return min(cand, key=key) if cand else None


def build_institutional_chart(df: pd.DataFrame, zones: List[Dict], vp: Dict, daily: bool) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=2, shared_xaxes=True,
        column_widths=[0.84, 0.16], row_heights=[0.74, 0.26],
        specs=[[{}, {}], [{}, None]],
        horizontal_spacing=0.008, vertical_spacing=0.03,
    )
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        increasing_line_color=COLOR_UP, decreasing_line_color=COLOR_DN, name="GC=F",
    ), row=1, col=1)

    x_end = df.index[-1]
    live = [z for z in zones if not z["mitigated"]]
    for kind in ["demand", "supply"]:
        for typ in ["OB", "FVG"]:
            sel = [z for z in live if z["kind"] == kind and z["type"] == typ]
            sel = sorted(sel, key=lambda z: z["pos"])[-5:]
            for z in sel:
                if kind == "demand":
                    fill = "rgba(38,166,154,0.22)" if typ == "OB" else "rgba(38,166,154,0.10)"
                    line = COLOR_UP
                else:
                    fill = "rgba(239,83,80,0.22)" if typ == "OB" else "rgba(239,83,80,0.10)"
                    line = COLOR_DN
                fig.add_shape(type="rect", x0=z["start"], x1=x_end, y0=z["bottom"], y1=z["top"],
                              fillcolor=fill, line=dict(color=line, width=1,
                                                        dash="solid" if typ == "OB" else "dot"),
                              layer="below", row=1, col=1)
                fig.add_annotation(x=z["start"], y=z["top"], text=f"{typ} {'D' if kind == 'demand' else 'S'}",
                                   showarrow=False, font=dict(size=9, color=line), xanchor="left",
                                   yanchor="bottom", row=1, col=1)

    # Perfil de volumen lateral
    colors = [GOLD_COLOR if i == vp["poc_idx"] else "rgba(120,144,156,0.55)"
              for i in range(len(vp["centers"]))]
    fig.add_trace(go.Bar(x=vp["profile"], y=vp["centers"], orientation="h",
                         marker_color=colors, name="VPVR", showlegend=False), row=1, col=2)
    for lvl, nm, col in [(vp["poc"], "POC", GOLD_COLOR), (vp["vah"], "VAH", "#90a4ae"), (vp["val"], "VAL", "#90a4ae")]:
        fig.add_hline(y=lvl, line_dash="dash", line_color=col, line_width=1, row=1, col=1,
                      annotation_text=f"{nm} {lvl:,.1f}", annotation_position="top left",
                      annotation_font_size=9)
        fig.add_hline(y=lvl, line_dash="dash", line_color=col, line_width=1, row=1, col=2)

    # CVD
    fig.add_trace(go.Scatter(x=df.index, y=df["CVD"], mode="lines", line=dict(color=GOLD_COLOR, width=1.6),
                             fill="tozeroy", fillcolor="rgba(212,175,55,0.10)", name="CVD"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["CVD"].ewm(span=20, adjust=False).mean(), mode="lines",
                             line=dict(color="#90a4ae", width=1, dash="dot"), name="CVD EMA20"), row=2, col=1)

    fig.update_yaxes(matches="y", showticklabels=False, row=1, col=2)
    fig.update_xaxes(rangeslider_visible=False)
    if daily:
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])], row=1, col=1)
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])], row=2, col=1)
    fig.update_layout(height=760, template="plotly_dark", margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", y=1.04), xaxis_rangeslider_visible=False,
                      bargap=0.05)
    return fig


# ---------------------------------------------------------------------------
# 3. MACRO: RETORNOS, CORRELACIONES, SESGO DE VARIABLES OMITIDAS
# ---------------------------------------------------------------------------
def make_returns(M: pd.DataFrame) -> pd.DataFrame:
    R = pd.DataFrame(index=M.index)
    R[LABELS["Oro"]] = np.log(M["Close"]).diff()
    R[LABELS["TNX"]] = M["TNX"].diff()
    R[LABELS["SPREAD"]] = M["SPREAD"].diff()
    for k in ["TIP", "HG", "SPX", "DXY", "VIX", "SI"]:
        R[LABELS[k]] = np.log(M[k]).diff()
    return R.replace([np.inf, -np.inf], np.nan).dropna()


def omitted_variable_table(R: pd.DataFrame) -> (pd.DataFrame, float):
    y = R[LABELS["Oro"]]
    X = R.drop(columns=[LABELS["Oro"]])
    multi = sm.OLS(y, sm.add_constant(X)).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    rows = []
    for col in X.columns:
        uni = sm.OLS(y, sm.add_constant(X[[col]])).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
        b_u, b_m = float(uni.params[col]), float(multi.params[col])
        rows.append({
            "Variable": col,
            "Beta univariada": round(b_u, 4),
            "Beta multivariada": round(b_m, 4),
            "t (multivariada, HAC)": round(float(multi.tvalues[col]), 2),
            "p-valor": round(float(multi.pvalues[col]), 4),
            "Sesgo si se omite el resto": round(b_u - b_m, 4),
        })
    return pd.DataFrame(rows), float(multi.rsquared)


# ---------------------------------------------------------------------------
# 4. MODELOS: VAR vs MLP
# ---------------------------------------------------------------------------
def build_supervised(M: pd.DataFrame, n_lags: int):
    """Features estacionarias (retornos/diferencias) + rezagos. Target: retorno logaritmico t+1 del Oro."""
    F = pd.DataFrame(index=M.index)
    pc = M["Close"].shift(1)
    F["g_open"] = np.log(M["Open"] / pc)
    F["g_high"] = np.log(M["High"] / pc)
    F["g_low"] = np.log(M["Low"] / pc)
    F["g_close"] = np.log(M["Close"] / pc)
    vol = M["Volume"].replace(0, np.nan)
    F["g_vol"] = np.log(vol / vol.rolling(20, min_periods=5).mean())
    F["tip"] = np.log(M["TIP"]).diff()
    F["tnx"] = M["TNX"].diff()
    F["spread"] = M["SPREAD"].diff()
    F["hg"] = np.log(M["HG"]).diff()
    F["spx"] = np.log(M["SPX"]).diff()
    F["dxy"] = np.log(M["DXY"]).diff()
    F["vix"] = np.log(M["VIX"]).diff()
    F["si"] = np.log(M["SI"]).diff()
    F = F.replace([np.inf, -np.inf], np.nan)
    F["g_vol"] = F["g_vol"].fillna(0.0).clip(-3, 3)
    blocks = [F.shift(k).add_suffix(f"_l{k}") for k in range(n_lags)]
    X = pd.concat(blocks, axis=1)
    y = F["g_close"].shift(-1)
    return X, y


def fit_mlp(X: np.ndarray, y: np.ndarray, hidden: tuple, seeds: int):
    sx = MinMaxScaler().fit(X)
    sy = MinMaxScaler().fit(y.reshape(-1, 1))
    Xs = sx.transform(X)
    ys = sy.transform(y.reshape(-1, 1)).ravel()
    models = []
    for s in range(seeds):
        m = MLPRegressor(hidden_layer_sizes=hidden, activation="tanh", solver="adam", alpha=1e-2,
                         learning_rate_init=1e-3, max_iter=400, early_stopping=False, random_state=s)
        m.fit(Xs, ys)
        models.append(m)
    return sx, sy, models


def predict_mlp(pack, X: np.ndarray) -> np.ndarray:
    sx, sy, models = pack
    Xs = np.clip(sx.transform(X), -0.5, 1.5)
    p = np.mean([m.predict(Xs) for m in models], axis=0)
    return sy.inverse_transform(p.reshape(-1, 1)).ravel()


def _metrics(actual: np.ndarray, pred: np.ndarray, a_ret: np.ndarray, p_ret: np.ndarray) -> Dict:
    return {
        "RMSE (USD)": float(np.sqrt(np.mean((actual - pred) ** 2))),
        "MAE (USD)": float(np.mean(np.abs(actual - pred))),
        "Precision direccional (%)": float(np.mean(np.sign(p_ret) == np.sign(a_ret)) * 100.0),
    }


@st.cache_resource(show_spinner="Entrenando VAR y red neuronal (cache de servidor)...")
def train_models(_M: pd.DataFrame, data_key: str, var_maxlags: int, ic: str,
                 test_frac: float, n_lags: int, hidden: tuple, seeds: int) -> Dict:
    M = _M
    N = len(M)
    s = int(N * (1.0 - test_frac))
    close = M["Close"].values.astype(float)
    lnp = np.log(M[["Open", "High", "Low", "Close"]].values.astype(float))
    R = np.vstack([np.full((1, 4), np.nan), np.diff(lnp, axis=0)])   # R[t] = retorno log en t

    # ---------------- VAR ----------------
    train_arr = R[1:s]
    adf_rows = []
    for j, nm in enumerate(["Open", "High", "Low", "Close"]):
        for tag, series in [(f"ln({nm}) nivel", lnp[:s, j]), (f"d ln({nm}) retorno", train_arr[:, j])]:
            try:
                stat, pval = adfuller(series, autolag="AIC")[:2]
                adf_rows.append({"Serie": tag, "Estadistico ADF": round(float(stat), 3),
                                 "p-valor": round(float(pval), 4),
                                 "Estacionaria (5%)": "SI" if pval < 0.05 else "NO"})
            except Exception:
                pass
    adf_df = pd.DataFrame(adf_rows)

    var_model = VAR(train_arr)
    try:
        sel = var_model.select_order(maxlags=var_maxlags)
        ic_df = pd.DataFrame({k.upper(): v for k, v in sel.ics.items()}, index=range(len(next(iter(sel.ics.values())))))
        ic_df.index.name = "Lag"
        k_star = int(max(1, sel.selected_orders.get(ic, 1)))
    except Exception:
        ic_df, k_star = pd.DataFrame(), 1
    var_res = var_model.fit(k_star)

    n_test = N - s
    var_ret = np.full(n_test, np.nan)
    for i, t in enumerate(range(s, N)):
        var_ret[i] = var_res.forecast(R[t - k_star:t], 1)[0, 3]

    # ---------------- MLP ----------------
    X_df, y_ser = build_supervised(M, n_lags)
    pos = np.arange(N)
    valid = X_df.notna().all(axis=1).values & y_ser.notna().values
    tr_mask = valid & (pos + 1 < s)
    te_mask = valid & (pos + 1 >= s)
    if tr_mask.sum() < 150 or te_mask.sum() < 20:
        raise ValueError("Muestra insuficiente: aumente los anos de historia o reduzca la fraccion de test.")
    pack = fit_mlp(X_df.values[tr_mask], y_ser.values[tr_mask], hidden, seeds)
    mlp_ret_te = predict_mlp(pack, X_df.values[te_mask])

    p_te = pos[te_mask]
    t_te = p_te + 1
    dates = M.index[t_te]
    actual = close[t_te]
    prev = close[p_te]
    a_ret = np.log(actual / prev)
    var_te = var_ret[t_te - s]
    prev_ret = np.log(prev / close[p_te - 1])

    ev = pd.DataFrame({
        "Real": actual,
        "VAR": prev * np.exp(var_te),
        "Red Neuronal": prev * np.exp(mlp_ret_te),
        "Random Walk": prev,
    }, index=dates)
    metrics = pd.DataFrame({
        "VAR": _metrics(actual, ev["VAR"].values, a_ret, var_te),
        "Red Neuronal Macro": _metrics(actual, ev["Red Neuronal"].values, a_ret, mlp_ret_te),
        "Random Walk (ingenuo)": _metrics(actual, prev, a_ret, prev_ret),
    }).T

    # ---------------- Proyeccion en vivo (reentrenado con toda la muestra) ----------------
    live = {"var_path": None, "var_next": np.nan, "mlp_next": np.nan}
    try:
        var_full = VAR(R[1:]).fit(k_star)
        fc = var_full.forecast(R[-k_star:], 5)[:, 3]
        live["var_path"] = close[-1] * np.exp(np.cumsum(fc))
        live["var_next"] = float(live["var_path"][0])
        x_live = X_df.iloc[[-1]]
        if x_live.notna().all(axis=1).iloc[0]:
            pack_full = fit_mlp(X_df.values[valid], y_ser.values[valid], hidden, seeds)
            live["mlp_next"] = float(close[-1] * np.exp(predict_mlp(pack_full, x_live.values)[0]))
    except Exception:
        pass

    return dict(adf=adf_df, ic=ic_df, lag=k_star, stable=bool(var_res.is_stable()),
                eval=ev, metrics=metrics, live=live, n_train=int(tr_mask.sum()),
                n_test=int(te_mask.sum()), split_date=M.index[s])


# ---------------------------------------------------------------------------
# 5. ESCENARIOS Y GESTION DE RIESGO
# ---------------------------------------------------------------------------
def _trade(direction: int, entry: float, sl: float, atr: float, struct: Optional[float],
           label: str, rr_targets=(3.0, 4.5)) -> Dict:
    risk = abs(entry - sl)
    if risk < 0.3 * atr:
        sl = entry - direction * 0.3 * atr
        risk = abs(entry - sl)
    tps = [entry + direction * risk * r for r in rr_targets]
    struct_rr = (abs(struct - entry) / risk) if struct is not None else np.nan
    return dict(label=label, direction=direction, entry=entry, sl=sl, risk=risk, tps=tps,
                struct=struct, struct_rr=struct_rr)


def build_scenarios(price: float, atr: float, zones: List[Dict], vp: Dict,
                    hi_n: float, lo_n: float) -> Dict[str, List[Dict]]:
    out = {}
    # --- Alcista: mitigacion de demanda ---
    z = nearest_zone(zones, price, "demand")
    if z is not None:
        entry = min((z["top"] + z["bottom"]) / 2.0, price)
        sl = z["bottom"] - 0.25 * atr
        lab = f"Mitigacion {z['type']} demanda [{usd(z['bottom'])} - {usd(z['top'])}]"
    else:
        entry, sl = price - atr, price - 2.0 * atr
        lab = "Sin zona vigente: retroceso 1 ATR (fallback)"
    sz = nearest_zone(zones, entry, "supply")
    struct = sz["bottom"] if sz is not None else (vp["vah"] if vp["vah"] > entry else hi_n)
    out["Alcista"] = [_trade(+1, entry, sl, atr, struct, lab)]

    # --- Bajista: mitigacion de oferta ---
    z = nearest_zone(zones, price, "supply")
    if z is not None:
        entry = max((z["top"] + z["bottom"]) / 2.0, price)
        sl = z["top"] + 0.25 * atr
        lab = f"Mitigacion {z['type']} oferta [{usd(z['bottom'])} - {usd(z['top'])}]"
    else:
        entry, sl = price + atr, price + 2.0 * atr
        lab = "Sin zona vigente: rebote 1 ATR (fallback)"
    dz = nearest_zone(zones, entry, "demand")
    struct = dz["top"] if dz is not None else (vp["val"] if vp["val"] < entry else lo_n)
    out["Bajista"] = [_trade(-1, entry, sl, atr, struct, lab)]

    # --- Lateral: reversion a la media dentro del Value Area ---
    long_t = _trade(+1, vp["val"], vp["val"] - 0.5 * atr, atr, vp["vah"], "Largo en VAL (reversion)",
                    rr_targets=(0.0, 0.0))
    short_t = _trade(-1, vp["vah"], vp["vah"] + 0.5 * atr, atr, vp["val"], "Corto en VAH (reversion)",
                     rr_targets=(0.0, 0.0))
    long_t["tps"] = [vp["poc"], vp["vah"]]
    short_t["tps"] = [vp["poc"], vp["val"]]
    out["Lateral"] = [long_t, short_t]
    return out


def bias_score(components: Dict[str, float]) -> float:
    vals = [np.sign(v) for v in components.values() if v is not None and not np.isnan(v)]
    return float(np.mean(vals)) if vals else 0.0


def scenario_chart(daily: pd.DataFrame, trade: Dict, vp: Dict, title: str) -> go.Figure:
    d = daily.tail(90)
    fig = go.Figure(go.Candlestick(x=d.index, open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"],
                                   increasing_line_color=COLOR_UP, decreasing_line_color=COLOR_DN, name="GC=F"))
    levels = [("Entrada", trade["entry"], GOLD_COLOR), ("Stop Loss", trade["sl"], COLOR_DN)]
    for i, tp in enumerate(trade["tps"], 1):
        levels.append((f"TP{i}", tp, COLOR_UP))
    for nm, lvl, col in levels:
        fig.add_hline(y=lvl, line_dash="dash", line_color=col, annotation_text=f"{nm} {lvl:,.1f}",
                      annotation_position="top left", annotation_font_size=10)
    fig.add_hline(y=vp["poc"], line_dash="dot", line_color="#90a4ae", annotation_text="POC",
                  annotation_position="bottom right", annotation_font_size=9)
    fig.update_xaxes(rangeslider_visible=False, rangebreaks=[dict(bounds=["sat", "mon"])])
    fig.update_layout(template="plotly_dark", height=460, title=title, margin=dict(l=10, r=10, t=50, b=10))
    return fig


# ---------------------------------------------------------------------------
# 6. DIAGNOSTICO AUTOMATICO
# ---------------------------------------------------------------------------
def pct_change(s: pd.Series, n: int) -> float:
    if len(s) <= n:
        return float("nan")
    return float(s.iloc[-1] / s.iloc[-n - 1] - 1.0) * 100.0


def macro_diagnosis(M: pd.DataFrame, of_daily: pd.DataFrame) -> List[str]:
    n = 20
    gold = pct_change(M["Close"], n)
    dxy = pct_change(M["DXY"], n)
    tnx_bp = float(M["TNX"].iloc[-1] - M["TNX"].iloc[-n - 1]) * 100.0
    tip = pct_change(M["TIP"], n)
    cu = pct_change(M["HG"], n)
    spx = pct_change(M["SPX"], n)
    vix = float(M["VIX"].iloc[-1])
    vix_avg = float(M["VIX"].tail(60).mean())
    spread = float(M["SPREAD"].iloc[-1])
    lines = [f"Oro {n}d: {gold:+.2f}% | Cierre {usd(M['Close'].iloc[-1])} USD/oz (GC=F)."]
    if dxy > 0 and tnx_bp > 0:
        lines.append(f"Regimen de endurecimiento: DXY {dxy:+.2f}% y 10Y {tnx_bp:+.0f} pb en {n}d. "
                     "Costo de oportunidad y dolar en contra del metal.")
    elif dxy < 0 and tnx_bp < 0:
        lines.append(f"Regimen de relajacion: DXY {dxy:+.2f}% y 10Y {tnx_bp:+.0f} pb en {n}d. Viento de cola para el oro.")
    else:
        lines.append(f"Senales mixtas: DXY {dxy:+.2f}% vs 10Y {tnx_bp:+.0f} pb en {n}d (sin regimen dominante).")
    lines.append(f"Inflacion/tasas reales (TIPS): {tip:+.2f}% en {n}d. "
                 f"Curva 10Y-3M: {spread:+.2f} pp.")
    lines.append(f"Crecimiento: cobre {cu:+.2f}% y S&P 500 {spx:+.2f}% en {n}d. "
                 f"VIX {vix:.1f} (media 60d {vix_avg:.1f}): "
                 f"{'estres por encima de lo normal' if vix > vix_avg * 1.15 else 'estres contenido'}.")
    lines.append(f"Order Flow diario: {cvd_divergence(of_daily, 20)}")
    return lines


# ---------------------------------------------------------------------------
# 7. APLICACION
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="Gold Quant Desk", page_icon="🥇", layout="wide")
    st.title("🥇 Gold Quant Desk | Futuros de Oro (GC=F)")
    st.caption("Order Flow · Smart Money Concepts · Matriz Macro · VAR vs Red Neuronal · Escenarios con R:R ≥ 1:3. "
               "Material analitico; no constituye asesoria financiera.")

    # ---- Sidebar ----
    with st.sidebar:
        st.header("Parametros")
        tf_name = st.selectbox("Temporalidad (Tab 1)", list(TF_CONFIG.keys()), index=0)
        years = st.slider("Historia macro (anos)", 3, 10, 5)
        disp_atr = st.slider("Desplazamiento OB (xATR)", 1.0, 3.0, 1.5, 0.1)
        st.subheader("Modelos")
        test_frac = st.slider("Fraccion out-of-sample", 0.10, 0.40, 0.20, 0.05)
        var_maxlags = st.slider("Max lags VAR", 2, 12, 8)
        ic = st.selectbox("Criterio de informacion", ["aic", "bic", "hqic", "fpe"], index=0)
        n_lags = st.slider("Rezagos de la red (dias)", 1, 5, 3)
        hidden_choice = st.selectbox("Arquitectura MLP", ["(32, 16)", "(64, 32)", "(16,)"], index=0)
        seeds = st.slider("Semillas del ensamble", 1, 5, 3)
        st.subheader("Riesgo")
        capital = st.number_input("Capital (USD)", min_value=1000.0, value=50000.0, step=1000.0)
        risk_pct = st.number_input("Riesgo por operacion (%)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
        roll_win = st.slider("Ventana correlacion rodante (dias)", 20, 120, 60, 5)
        if st.button("Refrescar datos y modelos"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()

    hidden = {"(32, 16)": (32, 16), "(64, 32)": (64, 32), "(16,)": (16,)}[hidden_choice]

    # ---- Datos ----
    with st.spinner("Descargando datos de mercado..."):
        M = load_macro_matrix(years)
        period, interval = TF_CONFIG[tf_name]
        px = fetch_ohlcv(GOLD, period, interval)
    if M.empty or px.empty or len(M) < 300:
        st.error("No se pudieron obtener datos suficientes de Yahoo Finance. Pulse 'Refrescar datos' o intente en unos minutos.")
        st.stop()

    daily = M[["Open", "High", "Low", "Close", "Volume"]]
    daily_of = add_order_flow(daily)
    daily_zones = detect_order_blocks(daily, disp_atr) + detect_fvg(daily)
    daily_vp = volume_profile(daily.tail(90))
    atr_d = float(compute_atr(daily).iloc[-1])
    price = float(daily["Close"].iloc[-1])

    # ---- Diagnostico ----
    with st.expander("Diagnostico estrategico automatico", expanded=True):
        for line in macro_diagnosis(M, daily_of):
            st.markdown(f"- {line}")
        st.caption(f"Ultima observacion diaria: {M.index[-1].date()} | Muestra: {len(M)} sesiones.")

    tab1, tab2, tab3, tab4 = st.tabs([
        "1 · Order Flow & SMC", "2 · Matriz Macro", "3 · VAR vs Red Neuronal", "4 · Escenarios & Riesgo"
    ])

    # ================= TAB 1 =================
    with tab1:
        df = add_order_flow(px)
        zones = detect_order_blocks(df, disp_atr) + detect_fvg(df)
        vp = volume_profile(df)
        live = [z for z in zones if not z["mitigated"]]
        n_od = sum(z["type"] == "OB" and z["kind"] == "demand" for z in live)
        n_os = sum(z["type"] == "OB" and z["kind"] == "supply" for z in live)
        n_fd = sum(z["type"] == "FVG" and z["kind"] == "demand" for z in live)
        n_fs = sum(z["type"] == "FVG" and z["kind"] == "supply" for z in live)
        c = st.columns(6)
        c[0].metric("Ultimo", usd(df["Close"].iloc[-1]))
        c[1].metric("POC", usd(vp["poc"]))
        c[2].metric("VAH", usd(vp["vah"]))
        c[3].metric("VAL", usd(vp["val"]))
        c[4].metric("OB sin mitigar (D/S)", f"{n_od} / {n_os}")
        c[5].metric("FVG sin mitigar (D/S)", f"{n_fd} / {n_fs}")
        st.info(cvd_divergence(df, 20))
        st.plotly_chart(build_institutional_chart(df, zones, vp, daily=(interval == "1d")))
        st.caption("Volume Delta estimado por vela: Volumen × (Cierre−Min)/(Max−Min) menos Volumen × (Max−Cierre)/(Max−Min). "
                   "Es una aproximacion OHLCV, no flujo de ordenes tick a tick. OB: ultima vela contraria previa a un "
                   "desplazamiento con ruptura de estructura. FVG: hueco de 3 velas. Solo se dibujan zonas no mitigadas.")

    # ================= TAB 2 =================
    with tab2:
        R = make_returns(M)
        win = st.slider("Ventana de la matriz de correlacion (sesiones)", 60, min(750, len(R)), min(252, len(R)), 10)
        corr = R.tail(win).corr()
        heat = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, zmin=-1, zmax=1,
                                    colorscale="RdBu", text=np.round(corr.values, 2),
                                    texttemplate="%{text}", reversescale=True))
        heat.update_layout(template="plotly_dark", height=520, title=f"Correlacion de retornos diarios (ultimas {win} sesiones)",
                           margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(heat)

        others = [c for c in R.columns if c != LABELS["Oro"]]
        pick = st.multiselect("Correlacion rodante vs Oro", others,
                              default=[LABELS["DXY"], LABELS["TNX"], LABELS["TIP"], LABELS["VIX"]])
        if pick:
            rf = go.Figure()
            for col in pick:
                rc = R[LABELS["Oro"]].rolling(roll_win).corr(R[col])
                rf.add_trace(go.Scatter(x=rc.index, y=rc, mode="lines", name=col))
            rf.add_hline(y=0, line_color="#90a4ae", line_dash="dot")
            rf.update_layout(template="plotly_dark", height=420, yaxis=dict(range=[-1, 1]),
                             title=f"Correlacion rodante ({roll_win} sesiones)", margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(rf)

        st.subheader("Control de variables omitidas")
        ovb, r2 = omitted_variable_table(R)
        st.dataframe(ovb, hide_index=True)
        st.caption(f"OLS multivariado con errores HAC (Newey-West, 5 rezagos). R² = {r2:.3f}. "
                   "Una diferencia grande entre beta univariada y multivariada indica sesgo por omitir los demas regresores. "
                   "TIP funciona como proxy de tasas reales/inflacion; VIX y DXY como proxies de estres y condiciones financieras "
                   "(no hay serie de desempleo diaria).")

    # ================= TAB 3 =================
    with tab3:
        key = f"{M.index[-1].date()}_{len(M)}_{years}"
        try:
            res = train_models(M, key, var_maxlags, ic, test_frac, n_lags, hidden, seeds)
        except Exception as e:
            st.error(f"Entrenamiento no disponible: {e}")
            res = None
        if res is not None:
            st.markdown(f"**Corte train/test:** {res['split_date'].date()} | Train: {res['n_train']} obs | "
                        f"Test: {res['n_test']} obs | Lag VAR ({ic.upper()}): {res['lag']} | "
                        f"VAR estable: {'SI' if res['stable'] else 'NO'}")
            ev = res["eval"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ev.index, y=ev["Real"], name="Precio real", line=dict(color="white", width=2)))
            fig.add_trace(go.Scatter(x=ev.index, y=ev["VAR"], name="VAR", line=dict(color="#42a5f5", width=1.3)))
            fig.add_trace(go.Scatter(x=ev.index, y=ev["Red Neuronal"], name="Red Neuronal Macro",
                                     line=dict(color=GOLD_COLOR, width=1.3)))
            fig.update_layout(template="plotly_dark", height=430, title="Pronostico a 1 dia fuera de muestra vs precio real",
                              margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=1.08))
            st.plotly_chart(fig)

            m = res["metrics"].copy()
            m["RMSE (USD)"] = m["RMSE (USD)"].map(lambda v: f"{v:,.2f}")
            m["MAE (USD)"] = m["MAE (USD)"].map(lambda v: f"{v:,.2f}")
            m["Precision direccional (%)"] = m["Precision direccional (%)"].map(lambda v: f"{v:.1f}")
            st.subheader("Cuadro comparativo (out-of-sample)")
            st.dataframe(m)
            best = res["metrics"]["RMSE (USD)"].idxmin()
            st.caption(f"Menor RMSE: {best}. Referencia: el Random Walk usa el cierre previo como pronostico y la direccion "
                       "del retorno previo. Un modelo que no mejora al Random Walk no aporta ventaja estadistica; "
                       "una precision direccional cercana a 50% es consistente con eficiencia debil.")

            live = res["live"]
            c1, c2, c3 = st.columns(3)
            c1.metric("Cierre actual", usd(price))
            if not np.isnan(live["var_next"]):
                c2.metric("VAR t+1", usd(live["var_next"]), f"{(live['var_next'] / price - 1) * 100:+.2f}%")
            if not np.isnan(live["mlp_next"]):
                c3.metric("Red Neuronal t+1", usd(live["mlp_next"]), f"{(live['mlp_next'] / price - 1) * 100:+.2f}%")
            if live["var_path"] is not None:
                fut = pd.bdate_range(M.index[-1] + pd.Timedelta(days=1), periods=5)
                pf = go.Figure()
                tail = M["Close"].tail(60)
                pf.add_trace(go.Scatter(x=tail.index, y=tail, name="Historico", line=dict(color="white")))
                pf.add_trace(go.Scatter(x=[tail.index[-1]] + list(fut), y=[price] + list(live["var_path"]),
                                        name="Trayectoria VAR (5d)", line=dict(color="#42a5f5", dash="dash")))
                pf.update_layout(template="plotly_dark", height=340, title="Proyeccion VAR a 5 sesiones",
                                 margin=dict(l=10, r=10, t=50, b=10))
                st.plotly_chart(pf)

            with st.expander("Estacionariedad (ADF) y seleccion de lags"):
                st.dataframe(res["adf"], hide_index=True)
                if not res["ic"].empty:
                    st.dataframe(res["ic"])
                st.caption("Los niveles de ln(precio) son I(1); el VAR se estima sobre retornos logaritmicos (estacionarios). "
                           "La red usa retornos/diferencias de todas las variables macro y min-max fijado solo con datos de entrenamiento.")

    # ================= TAB 4 =================
    with tab4:
        hi_n, lo_n = float(daily["High"].tail(60).max()), float(daily["Low"].tail(60).min())
        scen = build_scenarios(price, atr_d, daily_zones, daily_vp, hi_n, lo_n)

        comp = {
            "CVD 20d": float(daily_of["CVD"].iloc[-1] - daily_of["CVD"].iloc[-21]),
            "Precio vs POC": price - daily_vp["poc"],
            "DXY 20d (inverso)": -pct_change(M["DXY"], 20),
            "10Y 20d (inverso)": -float(M["TNX"].iloc[-1] - M["TNX"].iloc[-21]),
            "TIPS 20d": pct_change(M["TIP"], 20),
        }
        if "res" in locals() and res is not None:
            if not np.isnan(res["live"]["var_next"]):
                comp["VAR t+1"] = res["live"]["var_next"] - price
            if not np.isnan(res["live"]["mlp_next"]):
                comp["Red t+1"] = res["live"]["mlp_next"] - price
        score = bias_score(comp)
        view = "Alcista" if score > 0.3 else ("Bajista" if score < -0.3 else "Lateral")
        st.markdown(f"**Sesgo compuesto (heuristico, {len(comp)} senales):** {score:+.2f} → escenario dominante: **{view}**")
        st.caption("Promedio de signos de: " + ", ".join(comp.keys()) + ". No es una probabilidad calibrada.")
        c = st.columns(3)
        c[0].metric("Precio", usd(price))
        c[1].metric("ATR(14) diario", usd(atr_d))
        c[2].metric("Riesgo por operacion (USD)", usd(capital * risk_pct / 100.0))
        risk_usd = capital * risk_pct / 100.0

        rows = []
        for name, trades in scen.items():
            for t in trades:
                risk_pts = t["risk"]
                gc = int(risk_usd // (risk_pts * 100.0)) if risk_pts > 0 else 0
                mgc = int(risk_usd // (risk_pts * 10.0)) if risk_pts > 0 else 0
                rr1 = abs(t["tps"][0] - t["entry"]) / risk_pts
                rr2 = abs(t["tps"][1] - t["entry"]) / risk_pts
                rows.append({
                    "Escenario": name, "Setup": t["label"],
                    "Entrada": usd(t["entry"]), "Stop Loss": usd(t["sl"]), "Riesgo (pts)": usd(risk_pts),
                    "TP1": f"{usd(t['tps'][0])} (1:{rr1:.1f})", "TP2": f"{usd(t['tps'][1])} (1:{rr2:.1f})",
                    "Objetivo estructural": (f"{usd(t['struct'])} (1:{t['struct_rr']:.1f})"
                                             if t["struct"] is not None else "n/d"),
                    "R:R >= 1:3": "SI" if rr1 >= 3.0 else "NO",
                    "GC (100 oz)": gc, "MGC (10 oz)": mgc,
                })
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.caption("Alcista/Bajista: entrada en el punto medio de la zona OB/FVG sin mitigar mas cercana (limit), stop estructural "
                   "0.25 ATR mas alla de la zona, TP1 = 3R y TP2 = 4.5R. Lateral: reversion en VAL/VAH hacia POC; si el R:R es "
                   "menor a 1:3 se marca 'NO' y no cumple el criterio de la mesa. Tamano: contratos = riesgo USD / (puntos de riesgo × multiplicador).")

        pick_s = st.radio("Visualizar escenario", list(scen.keys()), horizontal=True,
                          index=list(scen.keys()).index(view))
        trade = scen[pick_s][0]
        if pick_s == "Lateral" and len(scen[pick_s]) > 1:
            side = st.radio("Lado", ["Largo en VAL", "Corto en VAH"], horizontal=True)
            trade = scen[pick_s][0] if side.startswith("Largo") else scen[pick_s][1]
        st.plotly_chart(scenario_chart(daily, trade, daily_vp, f"Escenario {pick_s}: {trade['label']}"))


if __name__ == "__main__":
    main()

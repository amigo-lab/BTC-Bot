import os
import json
import math
import requests
import ccxt
import pandas as pd
import numpy as np


# =========================
# 환경변수
# =========================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

# =========================
# 기본 설정
# =========================
SYMBOL = "BTC/USD"
TIMEFRAME = "15m"
LIMIT = 300

STATE_FILE = "alert_state.json"


# =========================
# 유틸
# =========================
def safe_float(value, default=0.0):
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return default
        return float(value)
    except Exception:
        return default


def format_price(value: float) -> str:
    return f"{value:,.2f}"


def load_state() -> dict:
    if not os.path.exists(STATE_FILE):
        return {}
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(state: dict) -> None:
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


# =========================
# 텔레그램 전송
# =========================
def send_telegram(msg: str) -> None:
    if not TELEGRAM_TOKEN or not CHAT_ID:
        raise ValueError("TELEGRAM_TOKEN 또는 CHAT_ID가 설정되지 않았습니다.")

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {
        "chat_id": CHAT_ID,
        "text": msg
    }
    response = requests.post(url, data=data, timeout=20)
    print("Telegram status:", response.status_code)
    print("Telegram response:", response.text)


# =========================
# 거래소 데이터 가져오기
# =========================
def fetch_ohlcv() -> pd.DataFrame:
    exchange = ccxt.kraken()
    ohlcv = exchange.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, limit=LIMIT)

    df = pd.DataFrame(
        ohlcv,
        columns=["timestamp", "open", "high", "low", "close", "volume"]
    )

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.astype({
        "open": float,
        "high": float,
        "low": float,
        "close": float,
        "volume": float
    })
    return df


# =========================
# 지표 계산
# =========================
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # EMA
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()

    # RSI14
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi14"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # 볼린저 밴드
    df["bb_mid"] = df["close"].rolling(window=20).mean()
    bb_std = df["close"].rolling(window=20).std()
    df["bb_upper"] = df["bb_mid"] + (2 * bb_std)
    df["bb_lower"] = df["bb_mid"] - (2 * bb_std)

    # 거래량 평균
    df["vol_ma20"] = df["volume"].rolling(window=20).mean()

    # ATR14
    prev_close = df["close"].shift(1)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    df["tr"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr14"] = df["tr"].rolling(window=14).mean()

    return df


# =========================
# 손절 / 목표가 계산
# =========================
def build_trade_plan(signal_type: str, entry: float, atr: float) -> dict:
    atr = safe_float(atr, 0.0)

    # ATR이 너무 작거나 비정상이면 기본 1% 사용
    if atr <= 0:
        atr = entry * 0.01

    stop_distance = atr * 1.5
    target1_distance = atr * 1.5
    target2_distance = atr * 3.0

    if signal_type == "LONG":
        stop_loss = entry - stop_distance
        target1 = entry + target1_distance
        target2 = entry + target2_distance
    else:
        stop_loss = entry + stop_distance
        target1 = entry - target1_distance
        target2 = entry - target2_distance

    risk_pct = abs((entry - stop_loss) / entry) * 100
    reward1_pct = abs((target1 - entry) / entry) * 100
    reward2_pct = abs((target2 - entry) / entry) * 100

    return {
        "entry": entry,
        "stop_loss": stop_loss,
        "target1": target1,
        "target2": target2,
        "risk_pct": risk_pct,
        "reward1_pct": reward1_pct,
        "reward2_pct": reward2_pct,
    }


# =========================
# 시그널 평가
# =========================
def evaluate_signal(df: pd.DataFrame) -> dict:
    last = df.iloc[-1]
    prev = df.iloc[-2]

    score_long = 0.0
    score_short = 0.0

    long_reasons = []
    short_reasons = []

    # -------------------------
    # 1) 추세
    # -------------------------
    if last["close"] > last["ema50"] > last["ema200"]:
        score_long += 2.0
        long_reasons.append("상승 추세 정렬 (종가 > EMA50 > EMA200)")

    if last["close"] < last["ema50"] < last["ema200"]:
        score_short += 2.0
        short_reasons.append("하락 추세 정렬 (종가 < EMA50 < EMA200)")

    # -------------------------
    # 2) RSI
    # -------------------------
    rsi = safe_float(last["rsi14"])

    if 35 <= rsi <= 50:
        score_long += 1.0
        long_reasons.append(f"RSI 눌림 후 회복 구간 ({rsi:.2f})")
    elif rsi < 35:
        score_long += 0.5
        long_reasons.append(f"RSI 낮음 ({rsi:.2f})")

    if last["close"] < last["ema50"]:
        if 50 <= rsi <= 65:
            score_short += 1.0
            short_reasons.append(f"RSI 반등 후 약세 구간 ({rsi:.2f})")
        elif rsi > 65:
            score_short += 0.5
            short_reasons.append(f"RSI 높음 ({rsi:.2f})")

    # -------------------------
    # 3) MACD
    # -------------------------
    if prev["macd"] < prev["macd_signal"] and last["macd"] > last["macd_signal"]:
        score_long += 1.5
        long_reasons.append("MACD 골든크로스")

    elif last["macd"] > last["macd_signal"] and last["macd_hist"] > prev["macd_hist"]:
        score_long += 1.0
        long_reasons.append("MACD 상승 모멘텀 강화")

    if prev["macd"] > prev["macd_signal"] and last["macd"] < last["macd_signal"]:
        score_short += 1.5
        short_reasons.append("MACD 데드크로스")

    elif last["macd"] < last["macd_signal"] and last["macd_hist"] < prev["macd_hist"]:
        score_short += 1.0
        short_reasons.append("MACD 하락 모멘텀 강화")

    # -------------------------
    # 4) 볼린저 밴드 위치
    # -------------------------
    if safe_float(last["bb_lower"]) > 0 and last["close"] <= last["bb_lower"] * 1.01:
        score_long += 1.0
        long_reasons.append("볼린저 하단 근접")

    if safe_float(last["bb_upper"]) > 0 and last["close"] >= last["bb_upper"] * 0.99:
        score_short += 1.0
        short_reasons.append("볼린저 상단 근접")

    # -------------------------
    # 5) 거래량
    # -------------------------
    vol_ma20 = safe_float(last["vol_ma20"])
    if vol_ma20 > 0 and last["volume"] > vol_ma20 * 1.2:
        if score_long > score_short:
            score_long += 1.0
            long_reasons.append("거래량 증가 확인")
        elif score_short > score_long:
            score_short += 1.0
            short_reasons.append("거래량 증가 확인")

    # -------------------------
    # 6) 단기 방향성
    # -------------------------
    if last["close"] > last["ema20"]:
        score_long += 0.5
        long_reasons.append("단기 가격이 EMA20 위")

    if last["close"] < last["ema20"]:
        score_short += 0.5
        short_reasons.append("단기 가격이 EMA20 아래")

    # -------------------------
    # 7) 신호 단계 판정
    # -------------------------
    signal_type = "NONE"
    signal_level = "NONE"
    final_score = 0.0
    reasons = []

    # 강한 신호
    if score_long >= 5.0 and score_long > score_short + 1.0:
        signal_type = "LONG"
        signal_level = "STRONG"
        final_score = score_long
        reasons = long_reasons

    elif score_short >= 5.0 and score_short > score_long + 1.0:
        signal_type = "SHORT"
        signal_level = "STRONG"
        final_score = score_short
        reasons = short_reasons

    # 관심 신호
    elif score_long >= 4.0 and score_long > score_short + 1.0:
        signal_type = "LONG"
        signal_level = "WATCH"
        final_score = score_long
        reasons = long_reasons

    elif score_short >= 4.0 and score_short > score_long + 1.0:
        signal_type = "SHORT"
        signal_level = "WATCH"
        final_score = score_short
        reasons = short_reasons

    trade_plan = None
    if signal_type in ["LONG", "SHORT"]:
        trade_plan = build_trade_plan(
            signal_type=signal_type,
            entry=safe_float(last["close"]),
            atr=safe_float(last["atr14"])
        )

    return {
        "signal": signal_type,
        "level": signal_level,
        "score_long": round(score_long, 2),
        "score_short": round(score_short, 2),
        "final_score": round(final_score, 2),
        "reasons": reasons,
        "trade_plan": trade_plan,
        "last": last
    }


# =========================
# 중복 알림 방지
# =========================
def build_alert_key(result: dict) -> str:
    if result["signal"] == "NONE":
        return "NONE"

    last = result["last"]
    candle_time = str(last["timestamp"])
    signal_type = result["signal"]
    signal_level = result["level"]

    return f"{SYMBOL}|{TIMEFRAME}|{candle_time}|{signal_type}|{signal_level}"


def is_duplicate_alert(result: dict, state: dict) -> bool:
    if result["signal"] == "NONE":
        return False

    current_key = build_alert_key(result)
    last_key = state.get("last_alert_key")

    return current_key == last_key


def update_alert_state(result: dict, state: dict) -> dict:
    if result["signal"] == "NONE":
        return state

    state["last_alert_key"] = build_alert_key(result)
    state["last_signal"] = result["signal"]
    state["last_level"] = result["level"]
    state["last_score"] = result["final_score"]
    state["last_candle"] = str(result["last"]["timestamp"])
    return state


# =========================
# 메시지 생성
# =========================
def build_message(result: dict) -> str:
    last = result["last"]
    plan = result["trade_plan"]

    if result["signal"] == "LONG" and result["level"] == "STRONG":
        title = "🚀 BTC 강한 롱 신호"
    elif result["signal"] == "LONG" and result["level"] == "WATCH":
        title = "🟡 BTC 롱 관심 신호"
    elif result["signal"] == "SHORT" and result["level"] == "STRONG":
        title = "🔻 BTC 강한 숏 신호"
    elif result["signal"] == "SHORT" and result["level"] == "WATCH":
        title = "🟠 BTC 숏 관심 신호"
    else:
        title = "ℹ️ BTC 신호 없음"

    reasons_text = "\n".join([f"• {r}" for r in result["reasons"]]) if result["reasons"] else "• 없음"

    msg = (
        f"{title}\n"
        f"━━━━━━━━━━\n"
        f"종목: {SYMBOL}\n"
        f"시간봉: {TIMEFRAME}\n"
        f"캔들시간: {last['timestamp']}\n"
        f"현재가: {format_price(last['close'])}\n"
        f"\n"
        f"[신호 점수]\n"
        f"• 최종 점수: {result['final_score']}\n"
        f"• 롱 점수: {result['score_long']}\n"
        f"• 숏 점수: {result['score_short']}\n"
        f"• 신호 단계: {result['level']}\n"
        f"\n"
        f"[핵심 지표]\n"
        f"• RSI14: {safe_float(last['rsi14']):.2f}\n"
        f"• EMA20: {format_price(last['ema20'])}\n"
        f"• EMA50: {format_price(last['ema50'])}\n"
        f"• EMA200: {format_price(last['ema200'])}\n"
        f"• MACD: {safe_float(last['macd']):.4f}\n"
        f"• MACD Signal: {safe_float(last['macd_signal']):.4f}\n"
        f"• 거래량: {safe_float(last['volume']):,.2f}\n"
        f"• 거래량평균20: {safe_float(last['vol_ma20']):,.2f}\n"
        f"• ATR14: {safe_float(last['atr14']):.2f}\n"
        f"\n"
        f"[판단 근거]\n"
        f"{reasons_text}\n"
    )

    if plan:
        msg += (
            f"\n"
            f"[참고 가격 플랜]\n"
            f"• 진입 기준가: {format_price(plan['entry'])}\n"
            f"• 손절가: {format_price(plan['stop_loss'])} (-{plan['risk_pct']:.2f}%)\n"
            f"• 목표가 1차: {format_price(plan['target1'])} (+{plan['reward1_pct']:.2f}%)\n"
            f"• 목표가 2차: {format_price(plan['target2'])} (+{plan['reward2_pct']:.2f}%)\n"
        )

    msg += (
        f"\n"
        f"※ 이 알림은 보조 판단용입니다.\n"
        f"※ 실제 진입 전 손절폭과 포지션 크기를 별도로 확인하세요."
    )

    return msg


def build_no_signal_message(result: dict) -> str:
    last = result["last"]
    return (
        f"ℹ️ BTC 신호 없음\n"
        f"━━━━━━━━━━\n"
        f"종목: {SYMBOL}\n"
        f"시간봉: {TIMEFRAME}\n"
        f"캔들시간: {last['timestamp']}\n"
        f"현재가: {format_price(last['close'])}\n"
        f"\n"
        f"롱 점수: {result['score_long']}\n"
        f"숏 점수: {result['score_short']}\n"
        f"RSI14: {safe_float(last['rsi14']):.2f}\n"
        f"ATR14: {safe_float(last['atr14']):.2f}"
    )


# =========================
# 메인
# =========================
def main() -> None:
    state = load_state()

    try:
        df = fetch_ohlcv()
        df = add_indicators(df)
        result = evaluate_signal(df)

        if result["signal"] in ["LONG", "SHORT"]:
            if is_duplicate_alert(result, state):
                print("중복 알림으로 전송 생략")
            else:
                send_telegram(build_message(result))
                state = update_alert_state(result, state)
                save_state(state)
        else:
            print("유효 신호 없음")
            print(build_no_signal_message(result))

    except Exception as e:
        error_msg = f"⚠️ 실행 오류 발생: {str(e)}"
        print(error_msg)
        send_telegram(error_msg)
        raise


if __name__ == "__main__":
    main()

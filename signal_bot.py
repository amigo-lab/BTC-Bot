import json
import os
from pathlib import Path

import ccxt
import numpy as np
import pandas as pd
import requests


# =========================
# 환경설정
# =========================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

SYMBOL = "BTC/USD"
TIMEFRAME = "15m"
LIMIT = 320
STATE_FILE = Path("alert_state.json")


# =========================
# 텔레그램 전송
# =========================
def send_telegram(msg: str) -> None:
    if not TELEGRAM_TOKEN or not CHAT_ID:
        raise ValueError(
            f"환경변수 누락: TELEGRAM_TOKEN={bool(TELEGRAM_TOKEN)}, CHAT_ID={bool(CHAT_ID)}"
        )

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {
        "chat_id": CHAT_ID,
        "text": msg,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }

    response = requests.post(url, data=data, timeout=20)
    print("Telegram status:", response.status_code)
    print("Telegram response:", response.text)

    response.raise_for_status()

    try:
        result = response.json()
    except Exception as e:
        raise RuntimeError(f"Telegram 응답 JSON 파싱 실패: {e}")

    if not result.get("ok"):
        raise RuntimeError(f"Telegram API error: {result}")


# =========================
# 상태파일 로드/저장
# =========================
def load_state() -> dict:
    if not STATE_FILE.exists():
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
# 거래소 데이터
# =========================
def fetch_ohlcv() -> pd.DataFrame:
    exchange = ccxt.kraken()
    ohlcv = exchange.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, limit=LIMIT)

    df = pd.DataFrame(
        ohlcv,
        columns=["timestamp", "open", "high", "low", "close", "volume"]
    )

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.astype({
        "open": float,
        "high": float,
        "low": float,
        "close": float,
        "volume": float,
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
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi14"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # Bollinger Bands
    df["bb_mid"] = df["close"].rolling(20).mean()
    bb_std = df["close"].rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + (2 * bb_std)
    df["bb_lower"] = df["bb_mid"] - (2 * bb_std)

    # Volume average
    df["vol_ma20"] = df["volume"].rolling(20).mean()

    # ATR14
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14).mean()

    # Recent structure
    df["recent_high_20"] = df["high"].rolling(20).max()
    df["recent_low_20"] = df["low"].rolling(20).min()

    # Momentum
    df["close_change_3"] = df["close"].pct_change(3) * 100
    df["close_change_5"] = df["close"].pct_change(5) * 100

    return df


# =========================
# 리스크 가격 계산
# =========================
def calc_trade_levels(signal_type: str, last: pd.Series) -> dict:
    entry = float(last["close"])
    atr = float(last["atr14"]) if not pd.isna(last["atr14"]) else entry * 0.006

    min_stop_pct = 0.0045  # 0.45%
    min_stop_distance = entry * min_stop_pct
    stop_distance = max(atr * 1.2, min_stop_distance)

    if signal_type == "LONG":
        stop = entry - stop_distance
        target1 = entry + (stop_distance * 1.3)
        target2 = entry + (stop_distance * 2.0)
    elif signal_type == "SHORT":
        stop = entry + stop_distance
        target1 = entry - (stop_distance * 1.3)
        target2 = entry - (stop_distance * 2.0)
    else:
        stop = entry
        target1 = entry
        target2 = entry

    risk = abs(entry - stop)
    reward1 = abs(target1 - entry)
    rr1 = reward1 / risk if risk > 0 else 0

    return {
        "entry": round(entry, 2),
        "stop": round(stop, 2),
        "target1": round(target1, 2),
        "target2": round(target2, 2),
        "rr1": round(rr1, 2),
        "atr": round(atr, 2),
        "stop_pct": round((risk / entry) * 100, 2) if entry > 0 else 0,
    }


# =========================
# 점수 등급
# =========================
def get_grade(level: str, final_score: float, volume_ratio) -> str:
    if level == "STRONG":
        if final_score >= 7.0 and volume_ratio is not None and volume_ratio >= 1.2:
            return "A+"
        return "A"
    if level == "WATCH":
        return "B"
    return "C"


# =========================
# 시장 코멘트
# =========================
def build_market_comment(result: dict) -> str:
    signal = result["signal"]
    level = result["level"]
    volume_ratio = result["volume_ratio"]
    last = result["last"]

    comments = []

    if signal == "LONG":
        comments.append("상승 추세 안에서 눌림 이후 회복 시도가 감지되고 있습니다.")
        if volume_ratio is not None:
            if volume_ratio >= 1.2:
                comments.append("거래량까지 동반되어 비교적 신뢰도 높은 반등 구조로 해석됩니다.")
            elif volume_ratio >= 0.8:
                comments.append("거래량은 보통 수준으로, 추세 확인은 가능하지만 과신은 금물입니다.")
            else:
                comments.append("거래량이 약해 힘 없는 반등일 가능성도 함께 열어둬야 합니다.")
    elif signal == "SHORT":
        comments.append("하락 추세 안에서 반등 이후 재차 약세 전개 가능성이 감지되고 있습니다.")
        if volume_ratio is not None:
            if volume_ratio >= 1.2:
                comments.append("거래량까지 실려 있어 하락 재개 신호의 신뢰도가 높아진 상태입니다.")
            elif volume_ratio >= 0.8:
                comments.append("거래량은 보통 수준으로, 추세 확인은 가능하지만 추격은 주의가 필요합니다.")
            else:
                comments.append("거래량이 약해 일시적 눌림일 가능성도 배제할 수 없습니다.")
    else:
        comments.append("현재는 승률 높은 진입 구간으로 보기 어려워 관망 우위 구간입니다.")

    if level == "STRONG":
        comments.append("단기 진입 후보로 볼 수 있으나 손절 기준은 반드시 선행되어야 합니다.")
    elif level == "WATCH":
        comments.append("즉시 진입보다 다음 캔들의 확인 동작을 보는 접근이 더 유리합니다.")

    comments.append(f"현재 RSI는 {last['rsi14']:.2f}, MACD는 {last['macd']:.4f} 수준입니다.")
    return " ".join(comments)


# =========================
# 최종 신호 평가
# =========================
def evaluate_signal(df: pd.DataFrame) -> dict:
    last = df.iloc[-1]
    prev = df.iloc[-2]

    score_long = 0.0
    score_short = 0.0
    long_reasons = []
    short_reasons = []
    long_blockers = []
    short_blockers = []

    bullish_trend = last["close"] > last["ema50"] > last["ema200"]
    bearish_trend = last["close"] < last["ema50"] < last["ema200"]

    # 큰 추세
    if bullish_trend:
        score_long += 2.5
        long_reasons.append("상승 추세 정렬 (종가 > EMA50 > EMA200)")
    else:
        long_blockers.append("상승 추세 정렬 미충족")

    if bearish_trend:
        score_short += 2.5
        short_reasons.append("하락 추세 정렬 (종가 < EMA50 < EMA200)")
    else:
        short_blockers.append("하락 추세 정렬 미충족")

    # 단기 위치
    if last["close"] > last["ema20"]:
        score_long += 0.7
        long_reasons.append("단기 가격이 EMA20 위")
    else:
        long_blockers.append("단기 가격이 EMA20 아래")

    if last["close"] < last["ema20"]:
        score_short += 0.7
        short_reasons.append("단기 가격이 EMA20 아래")
    else:
        short_blockers.append("단기 가격이 EMA20 위")

    # RSI
    if 40 <= last["rsi14"] <= 52:
        score_long += 1.2
        long_reasons.append(f"RSI 눌림 후 회복 구간 ({last['rsi14']:.2f})")
    elif 35 <= last["rsi14"] < 40:
        score_long += 0.7
        long_reasons.append(f"RSI 낮은 구간 ({last['rsi14']:.2f})")

    if 48 <= last["rsi14"] <= 60 and last["close"] < last["ema50"]:
        score_short += 1.2
        short_reasons.append(f"RSI 반등 후 약세 구간 ({last['rsi14']:.2f})")
    elif last["rsi14"] > 62:
        score_short += 0.7
        short_reasons.append(f"RSI 높은 구간 ({last['rsi14']:.2f})")

    # MACD
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

    # 볼린저
    if last["close"] <= last["bb_lower"] * 1.012:
        score_long += 0.8
        long_reasons.append("볼린저 하단 근접")
    if last["close"] >= last["bb_upper"] * 0.988:
        score_short += 0.8
        short_reasons.append("볼린저 상단 근접")

    # 최근 범위 위치
    recent_range = last["recent_high_20"] - last["recent_low_20"]
    position_in_range = None
    if recent_range > 0:
        position_in_range = (last["close"] - last["recent_low_20"]) / recent_range

        if position_in_range < 0.4:
            score_long += 0.5
            long_reasons.append("최근 20캔들 범위 하단권")
        if position_in_range > 0.6:
            score_short += 0.5
            short_reasons.append("최근 20캔들 범위 상단권")

    # 거래량
    volume_ratio = np.nan
    if pd.notna(last["vol_ma20"]) and last["vol_ma20"] > 0:
        volume_ratio = last["volume"] / last["vol_ma20"]

    if pd.notna(volume_ratio):
        base_long = score_long
        base_short = score_short

        if volume_ratio >= 1.2:
            if base_long >= base_short:
                score_long += 1.2
                long_reasons.append(f"거래량 증가 확인 (평균 대비 {volume_ratio:.2f}배)")
            if base_short > base_long:
                score_short += 1.2
                short_reasons.append(f"거래량 증가 확인 (평균 대비 {volume_ratio:.2f}배)")
        elif volume_ratio >= 0.8:
            if base_long >= base_short:
                score_long += 0.4
                long_reasons.append(f"거래량 보통 수준 (평균 대비 {volume_ratio:.2f}배)")
            if base_short > base_long:
                score_short += 0.4
                short_reasons.append(f"거래량 보통 수준 (평균 대비 {volume_ratio:.2f}배)")
        else:
            long_blockers.append(f"거래량 부족 ({volume_ratio:.2f}배)")
            short_blockers.append(f"거래량 부족 ({volume_ratio:.2f}배)")

    # 가짜 반등 / 가짜 하락 필터
    long_fake_rebound = False
    if bullish_trend:
        if (pd.notna(volume_ratio) and volume_ratio < 0.6) and last["macd"] <= last["macd_signal"]:
            long_fake_rebound = True
            long_blockers.append("가짜 반등 가능성")

    short_fake_break = False
    if bearish_trend:
        if (pd.notna(volume_ratio) and volume_ratio < 0.6) and last["macd"] >= last["macd_signal"]:
            short_fake_break = True
            short_blockers.append("가짜 하락 가능성")

    signal_type = "NONE"
    signal_level = "NONE"
    final_score = 0.0
    reasons = []
    blockers = []

    strong_long_ok = (
        bullish_trend
        and last["close"] > last["ema20"]
        and last["macd"] > last["macd_signal"]
        and 38 <= last["rsi14"] <= 55
        and not long_fake_rebound
    )

    strong_short_ok = (
        bearish_trend
        and last["close"] < last["ema20"]
        and last["macd"] < last["macd_signal"]
        and 45 <= last["rsi14"] <= 62
        and not short_fake_break
    )

    if score_long >= 6.0 and score_long > score_short + 1.5 and strong_long_ok:
        signal_type = "LONG"
        signal_level = "STRONG"
        final_score = score_long
        reasons = long_reasons
        blockers = long_blockers
    elif score_long >= 4.5 and score_long > score_short + 1.0 and bullish_trend:
        signal_type = "LONG"
        signal_level = "WATCH"
        final_score = score_long
        reasons = long_reasons
        blockers = long_blockers
    elif score_short >= 6.0 and score_short > score_long + 1.5 and strong_short_ok:
        signal_type = "SHORT"
        signal_level = "STRONG"
        final_score = score_short
        reasons = short_reasons
        blockers = short_blockers
    elif score_short >= 4.5 and score_short > score_long + 1.0 and bearish_trend:
        signal_type = "SHORT"
        signal_level = "WATCH"
        final_score = score_short
        reasons = short_reasons
        blockers = short_blockers

    # 강한신호인데 거래량 부족하면 WATCH로 강등
    if signal_level == "STRONG" and pd.notna(volume_ratio) and volume_ratio < 0.8:
        signal_level = "WATCH"
        blockers.append(f"거래량 부족 ({volume_ratio:.2f}배)")

    levels = calc_trade_levels(signal_type, last)
    grade = get_grade(
        signal_level,
        final_score,
        float(volume_ratio) if pd.notna(volume_ratio) else None
    )
    comment = build_market_comment({
        "signal": signal_type,
        "level": signal_level,
        "volume_ratio": round(float(volume_ratio), 2) if pd.notna(volume_ratio) else None,
        "last": last
    })

    return {
        "signal": signal_type,
        "level": signal_level,
        "grade": grade,
        "score_long": round(score_long, 2),
        "score_short": round(score_short, 2),
        "final_score": round(final_score, 2),
        "reasons": reasons,
        "blockers": blockers,
        "levels": levels,
        "last": last,
        "volume_ratio": round(float(volume_ratio), 2) if pd.notna(volume_ratio) else None,
        "position_in_range": round(float(position_in_range), 2) if position_in_range is not None else None,
        "comment": comment,
    }


# =========================
# 중복 알림 키
# =========================
def make_alert_key(result: dict) -> str:
    last = result["last"]
    candle_time = last["timestamp"].strftime("%Y-%m-%d %H:%M")
    return f"{SYMBOL}|{TIMEFRAME}|{candle_time}|{result['signal']}|{result['level']}"


def is_duplicate_alert(state: dict, alert_key: str) -> bool:
    return state.get("last_alert_key") == alert_key


# =========================
# 메시지 포맷
# =========================
def build_signal_insight(result: dict) -> str:
    signal = result["signal"]
    level = result["level"]
    volume_ratio = result["volume_ratio"]

    lines = []

    if signal == "LONG":
        lines.append("상승 추세 눌림 반등 시도")
        if volume_ratio is not None:
            if volume_ratio >= 1.2:
                lines.append("→ 거래량 동반으로 신뢰도 양호")
            elif volume_ratio >= 0.8:
                lines.append("→ 거래량 보통, 추세 확인 필요")
            else:
                lines.append("→ 거래량 부족으로 확정 아님")

        if level == "WATCH":
            lines.append("→ 다음 캔들 확인 필요")
        elif level == "STRONG":
            lines.append("→ 손절 기준 지키며 진입 후보")

    elif signal == "SHORT":
        lines.append("하락 추세 반등 후 재차 약세 가능")
        if volume_ratio is not None:
            if volume_ratio >= 1.2:
                lines.append("→ 거래량 동반으로 신뢰도 양호")
            elif volume_ratio >= 0.8:
                lines.append("→ 거래량 보통, 추세 확인 필요")
            else:
                lines.append("→ 거래량 부족으로 확정 아님")

        if level == "WATCH":
            lines.append("→ 다음 캔들 확인 필요")
        elif level == "STRONG":
            lines.append("→ 손절 기준 지키며 진입 후보")

    else:
        lines.append("현재는 승률 높은 진입 구간으로 보기 어려움")
        if volume_ratio is not None and volume_ratio < 0.8:
            lines.append("→ 거래량 부족으로 관망 우위")
        else:
            lines.append("→ 방향 확인 전까지 대기 권장")

    return "\n".join(lines)


def build_message(result: dict) -> str:
    last = result["last"]
    levels = result["levels"]

    line = "━━━━━━━━━━━━━━━━━━"
    candle_time = last["timestamp"].strftime("%Y-%m-%d %H:%M UTC")

    entry = levels["entry"]
    stop = levels["stop"]
    target1 = levels["target1"]
    target2 = levels["target2"]

    volume_ratio_text = f"{result['volume_ratio']}배" if result["volume_ratio"] is not None else "계산불가"
    position_text = f"{result['position_in_range']}" if result["position_in_range"] is not None else "계산불가"

    if result["signal"] == "LONG":
        stop_move = f"-{levels['stop_pct']}%"
        t1_move = f"+{round(((target1 - entry) / entry) * 100, 2)}%"
        t2_move = f"+{round(((target2 - entry) / entry) * 100, 2)}%"
    else:
        stop_move = f"-{levels['stop_pct']}%"
        t1_move = f"+{round(((entry - target1) / entry) * 100, 2)}%"
        t2_move = f"+{round(((entry - target2) / entry) * 100, 2)}%"

    # 제목/배지는 level 기준으로 출력
    if result["signal"] == "LONG" and result["level"] == "STRONG":
        title = "🚀 BTC VIP LONG BRIEFING"
        badge = "🟢 강한 롱 신호"
    elif result["signal"] == "LONG" and result["level"] == "WATCH":
        title = "🟡 BTC VIP WATCH BRIEFING"
        badge = "🟡 롱 관심 신호"
    elif result["signal"] == "SHORT" and result["level"] == "STRONG":
        title = "🔻 BTC VIP SHORT BRIEFING"
        badge = "🔴 강한 숏 신호"
    elif result["signal"] == "SHORT" and result["level"] == "WATCH":
        title = "🟠 BTC VIP WATCH BRIEFING"
        badge = "🟠 숏 관심 신호"
    else:
        title = "ℹ️ BTC VIP MARKET BRIEFING"
        badge = "⚪️ 유효 신호 없음"

    risk_lines = []
    for b in result["blockers"]:
        if "거래량" in b or "가짜" in b:
            risk_lines.append(f"• {b}")

    if not risk_lines:
        risk_lines = ["• 특별한 리스크 경고 없음"]

    insight = build_signal_insight(result)

    return (
        f"{title}\n"
        f"{line}\n"
        f"{badge} | 등급: {result['grade']}\n\n"

        f"{line}\n"
        f"💠 Market Snapshot\n"
        f"{line}\n"
        f"• 종목: {SYMBOL}\n"
        f"• 시간봉: {TIMEFRAME}\n"
        f"• 캔들시간: {candle_time}\n"
        f"• 현재가: {last['close']:,.2f}\n\n"

        f"{line}\n"
        f"📊 Signal Score\n"
        f"{line}\n"
        f"• 최종 점수: {result['final_score']}\n"
        f"• 롱 점수: {result['score_long']}\n"
        f"• 숏 점수: {result['score_short']}\n"
        f"• 신호 단계: {result['level']}\n\n"

        f"{line}\n"
        f"📌 Trade Plan\n"
        f"{line}\n"
        f"• 진입: {entry:,.2f}\n"
        f"• 손절: {stop:,.2f} ({stop_move})\n"
        f"• 목표1: {target1:,.2f} ({t1_move})\n"
        f"• 목표2: {target2:,.2f} ({t2_move})\n"
        f"• RR: {levels['rr1']}\n\n"

        f"{line}\n"
        f"📈 Core Indicators\n"
        f"{line}\n"
        f"• RSI: {last['rsi14']:.2f}\n"
        f"• EMA20 / 50 / 200: {last['ema20']:,.2f} / {last['ema50']:,.2f} / {last['ema200']:,.2f}\n"
        f"• MACD: {last['macd']:.2f} / {last['macd_signal']:.2f}\n"
        f"• ATR: {levels['atr']:.2f}\n\n"

        f"{line}\n"
        f"🔍 Market Quality\n"
        f"{line}\n"
        f"• 거래량: {last['volume']:.2f}\n"
        f"• 평균: {last['vol_ma20']:.2f}\n"
        f"• 비율: {volume_ratio_text}{' ⚠️' if result['volume_ratio'] is not None and result['volume_ratio'] < 0.8 else ''}\n"
        f"• 위치: {position_text}\n\n"

        f"{line}\n"
        f"🧠 VIP Insight\n"
        f"{line}\n"
        f"{insight}\n\n"

        f"{line}\n"
        f"⚠️ Risk Factors\n"
        f"{line}\n"
        f"{chr(10).join(risk_lines)}\n\n"

        f"{line}\n"
        f"📝 VIP Note\n"
        f"{line}\n"
        f"• 자동매매 신호 아님\n"
        f"• 포지션 크기 반드시 관리\n"
        f"• 거래량 없으면 실패 확률 ↑"
    )


def build_no_signal_message(result: dict) -> str:
    last = result["last"]
    line = "━━━━━━━━━━━━━━━━━━"
    candle_time = last["timestamp"].strftime("%Y-%m-%d %H:%M UTC")
    volume_ratio_text = f"{result['volume_ratio']}배" if result["volume_ratio"] is not None else "계산불가"

    return (
        f"BTC VIP MARKET BRIEFING\n"
        f"{line}\n"
        f"⚪️ 유효 신호 없음\n\n"

        f"{line}\n"
        f"💠 Market Snapshot\n"
        f"{line}\n"
        f"• 종목: {SYMBOL}\n"
        f"• 시간봉: {TIMEFRAME}\n"
        f"• 캔들시간: {candle_time}\n"
        f"• 현재가: {last['close']:,.2f}\n\n"

        f"{line}\n"
        f"📊 Score Snapshot\n"
        f"{line}\n"
        f"• 롱 점수: {result['score_long']}\n"
        f"• 숏 점수: {result['score_short']}\n"
        f"• 거래량 비율: {volume_ratio_text}\n\n"

        f"{line}\n"
        f"🧠 VIP Market Comment\n"
        f"{line}\n"
        f"{result['comment']}"
    )


# =========================
# 메인
# =========================
def main() -> None:
    state = load_state()

    df = fetch_ohlcv()
    df = add_indicators(df)
    result = evaluate_signal(df)

    print("signal =", result["signal"])
    print("level =", result["level"])
    print("score_long =", result["score_long"])
    print("score_short =", result["score_short"])

    alert_key = make_alert_key(result)

    if is_duplicate_alert(state, alert_key):
        print("중복 알림이므로 전송하지 않음:", alert_key)
        return

    if result["signal"] in ["LONG", "SHORT"]:
        msg = build_message(result)
    else:
        msg = build_no_signal_message(result)

    send_telegram(msg)

    state["last_alert_key"] = alert_key
    state["last_signal"] = result["signal"]
    state["last_level"] = result["level"]
    state["last_grade"] = result["grade"]
    state["last_candle_time"] = result["last"]["timestamp"].strftime("%Y-%m-%d %H:%M:%S UTC")
    save_state(state)

    print("알림 전송 완료:", alert_key)


if __name__ == "__main__":
    main()

import os
import requests
import ccxt
import pandas as pd
import numpy as np


# =========================
# 환경변수
# =========================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

SYMBOL = "BTC/USDT"
TIMEFRAME = "15m"
LIMIT = 250


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
    response = requests.post(url, data=data, timeout=15)
    print("Telegram status:", response.status_code)
    print("Telegram response:", response.text)


# =========================
# 거래소 데이터 가져오기
# =========================
def fetch_ohlcv() -> pd.DataFrame:
    exchange = ccxt.binance()
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

    # RSI(14)
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

    # Bollinger Bands(20, 2)
    bb_mid = df["close"].rolling(window=20).mean()
    bb_std = df["close"].rolling(window=20).std()

    df["bb_mid"] = bb_mid
    df["bb_upper"] = bb_mid + (2 * bb_std)
    df["bb_lower"] = bb_mid - (2 * bb_std)

    # 거래량 평균
    df["vol_ma20"] = df["volume"].rolling(window=20).mean()

    # 최근 고점/저점 참고용
    df["recent_high_20"] = df["high"].rolling(window=20).max()
    df["recent_low_20"] = df["low"].rolling(window=20).min()

    return df


# =========================
# 시그널 평가
# =========================
def evaluate_signal(df: pd.DataFrame) -> dict:
    last = df.iloc[-1]
    prev = df.iloc[-2]

    score_long = 0
    score_short = 0

    long_reasons = []
    short_reasons = []

    # -------------------------
    # 1) 추세 판단
    # -------------------------
    if last["close"] > last["ema50"] > last["ema200"]:
        score_long += 2
        long_reasons.append("상승 추세(EMA50 > EMA200, 종가 상단)")
    elif last["close"] < last["ema50"] < last["ema200"]:
        score_short += 2
        short_reasons.append("하락 추세(EMA50 < EMA200, 종가 하단)")

    # -------------------------
    # 2) RSI 타이밍
    # -------------------------
    if 35 <= last["rsi14"] <= 50:
        score_long += 1
        long_reasons.append(f"RSI 눌림 후 반등 구간({last['rsi14']:.2f})")
    elif 50 <= last["rsi14"] <= 65:
        score_short += 0  # 중립
    elif last["rsi14"] < 35:
        # 너무 약한 경우도 있을 수 있으니 강한 롱 신호로 보지 않고 보조만
        score_long += 0.5
        long_reasons.append(f"RSI 낮음({last['rsi14']:.2f})")
    elif 50 <= last["rsi14"] <= 65:
        score_short += 0
    elif 50 <= last["rsi14"] <= 65:
        pass

    if 50 <= last["rsi14"] <= 65:
        score_short += 0

    if 50 <= last["rsi14"] <= 65:
        pass

    if 50 <= last["rsi14"] <= 65:
        pass

    if 50 <= last["rsi14"] <= 65:
        pass

    if 50 <= last["rsi14"] <= 65:
        pass

    if 50 <= last["rsi14"] <= 65:
        pass

    if 50 <= last["rsi14"] <= 65:
        pass

    if 50 <= last["rsi14"] <= 65:
        pass

    if 50 <= last["rsi14"] <= 65:
        pass

    if 50 <= last["rsi14"] <= 65:
        pass

    if 50 <= last["rsi14"] <= 65:
        pass

    if 50 <= last["rsi14"] <= 65:
        pass

    if 50 <= last["rsi14"] <= 65:
        pass

    if 50 <= last["rsi14"] <= 65:
        pass

    # 숏용 RSI
    if 50 <= last["rsi14"] <= 65 and last["close"] < last["ema50"]:
        score_short += 1
        short_reasons.append(f"RSI 반등 후 약세 구간({last['rsi14']:.2f})")
    elif last["rsi14"] > 65:
        score_short += 0.5
        short_reasons.append(f"RSI 높음({last['rsi14']:.2f})")

    # -------------------------
    # 3) MACD
    # -------------------------
    if prev["macd"] < prev["macd_signal"] and last["macd"] > last["macd_signal"]:
        score_long += 1.5
        long_reasons.append("MACD 골든크로스")
    elif last["macd"] > last["macd_signal"] and last["macd_hist"] > prev["macd_hist"]:
        score_long += 1
        long_reasons.append("MACD 상승 모멘텀 강화")

    if prev["macd"] > prev["macd_signal"] and last["macd"] < last["macd_signal"]:
        score_short += 1.5
        short_reasons.append("MACD 데드크로스")
    elif last["macd"] < last["macd_signal"] and last["macd_hist"] < prev["macd_hist"]:
        score_short += 1
        short_reasons.append("MACD 하락 모멘텀 강화")

    # -------------------------
    # 4) 볼린저 밴드
    # -------------------------
    if last["close"] <= last["bb_lower"] * 1.01:
        score_long += 1
        long_reasons.append("볼린저 하단 근처")
    elif last["close"] >= last["bb_upper"] * 0.99:
        score_short += 1
        short_reasons.append("볼린저 상단 근처")

    # -------------------------
    # 5) 거래량
    # -------------------------
    if last["volume"] > last["vol_ma20"] * 1.2:
        if score_long > score_short:
            score_long += 1
            long_reasons.append("거래량 증가 확인")
        elif score_short > score_long:
            score_short += 1
            short_reasons.append("거래량 증가 확인")

    # -------------------------
    # 6) 단기 방향성
    # -------------------------
    if last["close"] > last["ema20"]:
        score_long += 0.5
        long_reasons.append("단기 가격이 EMA20 위")
    elif last["close"] < last["ema20"]:
        score_short += 0.5
        short_reasons.append("단기 가격이 EMA20 아래")

    # -------------------------
    # 최종 판정
    # -------------------------
    signal_type = "NONE"
    final_score = 0
    reasons = []

    if score_long >= 5 and score_long > score_short + 1:
        signal_type = "LONG"
        final_score = score_long
        reasons = long_reasons
    elif score_short >= 5 and score_short > score_long + 1:
        signal_type = "SHORT"
        final_score = score_short
        reasons = short_reasons

    return {
        "signal": signal_type,
        "score_long": round(score_long, 2),
        "score_short": round(score_short, 2),
        "final_score": round(final_score, 2),
        "reasons": reasons,
        "last": last
    }


# =========================
# 메시지 생성
# =========================
def build_message(result: dict) -> str:
    last = result["last"]

    price = last["close"]
    rsi = last["rsi14"]
    ema20 = last["ema20"]
    ema50 = last["ema50"]
    ema200 = last["ema200"]
    macd = last["macd"]
    macd_signal = last["macd_signal"]
    volume = last["volume"]
    vol_ma20 = last["vol_ma20"]

    reasons_text = "\n".join([f"- {r}" for r in result["reasons"]]) if result["reasons"] else "- 없음"

    if result["signal"] == "LONG":
        title = "🚀 BTC 롱 후보 알림"
    elif result["signal"] == "SHORT":
        title = "🔻 BTC 숏 후보 알림"
    else:
        title = "ℹ️ BTC 중립"

    msg = (
        f"{title}\n"
        f"\n"
        f"종목: {SYMBOL}\n"
        f"시간봉: {TIMEFRAME}\n"
        f"가격: {price:.2f}\n"
        f"신호강도 점수: {result['final_score']}\n"
        f"롱 점수: {result['score_long']} | 숏 점수: {result['score_short']}\n"
        f"\n"
        f"[지표 상태]\n"
        f"- RSI14: {rsi:.2f}\n"
        f"- EMA20: {ema20:.2f}\n"
        f"- EMA50: {ema50:.2f}\n"
        f"- EMA200: {ema200:.2f}\n"
        f"- MACD: {macd:.4f}\n"
        f"- MACD Signal: {macd_signal:.4f}\n"
        f"- 거래량: {volume:.2f}\n"
        f"- 거래량평균20: {vol_ma20:.2f}\n"
        f"\n"
        f"[판단 근거]\n"
        f"{reasons_text}\n"
        f"\n"
        f"※ 참고: 이 알림은 보조 신호이며, 자동 매매 지시가 아닙니다."
    )
    return msg


# =========================
# 메인
# =========================
def main() -> None:
    df = fetch_ohlcv()
    df = add_indicators(df)

    result = evaluate_signal(df)

    # 유효 신호일 때만 알림
    if result["signal"] in ["LONG", "SHORT"]:
        msg = build_message(result)
        send_telegram(msg)
    else:
        # 원하면 중립 상태도 보고 가능
        print("유효 신호 없음")
        print(f"LONG={result['score_long']} / SHORT={result['score_short']}")


if __name__ == "__main__":
    main()

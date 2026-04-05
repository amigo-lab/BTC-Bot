import os
import requests
import ccxt
import pandas as pd
import numpy as np


TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

SYMBOL = "BTC/USD"
TIMEFRAME = "15m"
LIMIT = 250


def send_telegram(msg: str) -> None:
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {
        "chat_id": CHAT_ID,
        "text": msg
    }
    response = requests.post(url, data=data, timeout=15)
    print(response.status_code, response.text)


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


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi14"] = 100 - (100 / (1 + rs))

    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    bb_mid = df["close"].rolling(window=20).mean()
    bb_std = df["close"].rolling(window=20).std()
    df["bb_mid"] = bb_mid
    df["bb_upper"] = bb_mid + (2 * bb_std)
    df["bb_lower"] = bb_mid - (2 * bb_std)

    df["vol_ma20"] = df["volume"].rolling(window=20).mean()

    return df


def evaluate_signal(df: pd.DataFrame) -> dict:
    last = df.iloc[-1]
    prev = df.iloc[-2]

    score_long = 0
    score_short = 0
    long_reasons = []
    short_reasons = []

    if last["close"] > last["ema50"] > last["ema200"]:
        score_long += 2
        long_reasons.append("상승 추세")
    elif last["close"] < last["ema50"] < last["ema200"]:
        score_short += 2
        short_reasons.append("하락 추세")

    if 35 <= last["rsi14"] <= 50:
        score_long += 1
        long_reasons.append(f"RSI 눌림 구간 ({last['rsi14']:.2f})")
    elif last["rsi14"] < 35:
        score_long += 0.5
        long_reasons.append(f"RSI 낮음 ({last['rsi14']:.2f})")

    if 50 <= last["rsi14"] <= 65 and last["close"] < last["ema50"]:
        score_short += 1
        short_reasons.append(f"RSI 반등 후 약세 ({last['rsi14']:.2f})")
    elif last["rsi14"] > 65:
        score_short += 0.5
        short_reasons.append(f"RSI 높음 ({last['rsi14']:.2f})")

    if prev["macd"] < prev["macd_signal"] and last["macd"] > last["macd_signal"]:
        score_long += 1.5
        long_reasons.append("MACD 골든크로스")
    elif last["macd"] > last["macd_signal"] and last["macd_hist"] > prev["macd_hist"]:
        score_long += 1
        long_reasons.append("MACD 상승 모멘텀")

    if prev["macd"] > prev["macd_signal"] and last["macd"] < last["macd_signal"]:
        score_short += 1.5
        short_reasons.append("MACD 데드크로스")
    elif last["macd"] < last["macd_signal"] and last["macd_hist"] < prev["macd_hist"]:
        score_short += 1
        short_reasons.append("MACD 하락 모멘텀")

    if last["close"] <= last["bb_lower"] * 1.01:
        score_long += 1
        long_reasons.append("볼린저 하단 근처")
    elif last["close"] >= last["bb_upper"] * 0.99:
        score_short += 1
        short_reasons.append("볼린저 상단 근처")

    if last["volume"] > last["vol_ma20"] * 1.2:
        if score_long > score_short:
            score_long += 1
            long_reasons.append("거래량 증가")
        elif score_short > score_long:
            score_short += 1
            short_reasons.append("거래량 증가")

    if last["close"] > last["ema20"]:
        score_long += 0.5
        long_reasons.append("단기 EMA20 위")
    elif last["close"] < last["ema20"]:
        score_short += 0.5
        short_reasons.append("단기 EMA20 아래")

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


def build_message(result: dict) -> str:
    last = result["last"]
    reasons_text = "\n".join([f"- {r}" for r in result["reasons"]]) if result["reasons"] else "- 없음"

    if result["signal"] == "LONG":
        title = "🚀 BTC 롱 후보 알림"
    elif result["signal"] == "SHORT":
        title = "🔻 BTC 숏 후보 알림"
    else:
        title = "ℹ️ BTC 중립"

    return (
        f"{title}\n\n"
        f"종목: {SYMBOL}\n"
        f"시간봉: {TIMEFRAME}\n"
        f"가격: {last['close']:.2f}\n"
        f"신호강도: {result['final_score']}\n"
        f"롱 점수: {result['score_long']} | 숏 점수: {result['score_short']}\n\n"
        f"[지표]\n"
        f"- RSI14: {last['rsi14']:.2f}\n"
        f"- EMA20: {last['ema20']:.2f}\n"
        f"- EMA50: {last['ema50']:.2f}\n"
        f"- EMA200: {last['ema200']:.2f}\n"
        f"- MACD: {last['macd']:.4f}\n"
        f"- MACD Signal: {last['macd_signal']:.4f}\n"
        f"- 거래량: {last['volume']:.2f}\n"
        f"- 거래량평균20: {last['vol_ma20']:.2f}\n\n"
        f"[판단 근거]\n{reasons_text}\n\n"
        f"※ 참고용 보조 신호입니다."
    )


def main() -> None:
    try:
        df = fetch_ohlcv()
        df = add_indicators(df)
        result = evaluate_signal(df)

        if result["signal"] in ["LONG", "SHORT"]:
            send_telegram(build_message(result))
        else:
            send_telegram(
                f"ℹ️ BTC 신호 없음\n"
                f"종목: {SYMBOL}\n"
                f"시간봉: {TIMEFRAME}\n"
                f"롱 점수: {result['score_long']} | 숏 점수: {result['score_short']}"
            )
    except Exception as e:
        send_telegram(f"⚠️ 실행 오류 발생: {str(e)}")
        raise


if __name__ == "__main__":
    main()

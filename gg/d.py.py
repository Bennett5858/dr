import json
import time
import websocket
# from datetime import datetime  # Removed unused import
from collections import deque, Counter
from colorama import init, Fore  # Removed unused Style

init(autoreset=True)

API_TOKEN = "2G2HGBSga15fSlS"
SYMBOL = "R_100"
BASE_BET = 5
COOLDOWN_SECONDS = 13
MAX_LOSS_STREAK = 3
DAILY_PROFIT_GOAL = 20
MAX_CONSECUTIVE_LOSSES = 5
last_digits = deque(maxlen=1000)
price_history = deque(maxlen=10)
last_price = None
authorized = False
trade_in_progress = False
current_proposal_id = None
server_time = None
last_trade_time = 0
account_balance = 0.0
starting_balance = 0.0
current_bet = BASE_BET
loss_streak = 0
daily_profit = 0

# Trade statistics
total_trades = 0
wins = 0
losses = 0
consecutive_losses = 0
max_drawdown = 0
# current_price will be set dynamically in the on_message handler

def print_last_digit(price):
    price_str = str(price).replace('.', '')
    last_digit = price_str[-1]
    print(f"🎯 Last digit of tick: {last_digit}")


def extract_last_two_digits(price):
    price_str = str(price).replace('.', '')
    return price_str[-2:]
def get_current_price(tick_data):
    """
    Extracts the current price from tick data.

    Args:
        tick_data (dict): The latest tick data received from the data source.

    Returns:
        float: The current price extracted from the tick data.
    """
    try:
        # Adjust the key depending on your tick data structure
        price = float(tick_data.get('price') or tick_data.get('tick') or tick_data.get('quote'))
        return price
    except (TypeError, ValueError):
        print("⚠️ Failed to extract current price from tick data.")
        return None
    
def receive_tick(ws=None, symbol=SYMBOL):
    """
    Sends a tick subscription request. Tick data is handled in on_message.
    """
    if ws is not None:
        ws.send(json.dumps({"ticks": symbol}))

def detect_hot_digits(last_digits, window=100, min_freq=0.12, trend_weight=0.5):
    """
    Detects 'hot' digits that appear more frequently than expected in the recent history.
    - window: number of recent digits to analyze (default 100)
    - min_freq: minimum frequency (as a fraction) to consider a digit 'hot'
    - trend_weight: extra weight for digits that are trending upward in frequency
    Returns a sorted list of hot digits (most frequent first).
    """
    if not last_digits:
        return []

    # Use only the most recent 'window' digits
    recent = list(last_digits)[-window:] if len(last_digits) > window else list(last_digits)
    digit_counts = Counter(recent)
    total = len(recent)
    avg_freq = total / 10  # Expectation for uniform distribution (digits 0-9)

    # Calculate frequency for each digit
    digit_freqs = {d: digit_counts.get(d, 0) / total for d in range(10)}

    # Detect digits with frequency above threshold
    hot_digits = []
    for d, freq in digit_freqs.items():
        # Check if digit is trending up in the last half of the window
        half = total // 2
        first_half = recent[:half]
        second_half = recent[half:]
        freq_first = first_half.count(d) / half if half else 0
        freq_second = second_half.count(d) / (total - half) if (total - half) else 0
        trending_up = freq_second > freq_first + 0.03  # 3% increase

        # Apply trend weight if trending up
        effective_freq = freq + (trend_weight * (freq_second - freq_first) if trending_up else 0)

        if effective_freq > min_freq or digit_counts[d] > avg_freq * 1.25:
            hot_digits.append((d, effective_freq))

    # Sort hot digits by effective frequency descending
    hot_digits.sort(key=lambda x: x[1], reverse=True)
    return [d for d, _ in hot_digits]


def get_least_popular_digit_below_3(digits):
    """
    Dynamically finds the least frequent digit below 3 in the recent digit history.
    If all digits below 3 are equally frequent, returns the one that appeared least recently.
    If no digits below 3, returns 0 as a safe default.
    """
    if not digits:
        return 0

    # Only consider the most recent 100 digits for adaptiveness
    recent_digits = list(digits)[-100:] if len(digits) > 100 else list(digits)
    filtered = [d for d in recent_digits if d < 3]
    if not filtered:
        return 0

    counts = Counter(filtered)
    min_count = min(counts.values())
    least_popular = [d for d, c in counts.items() if c == min_count]

    # If tie, pick the one that appeared least recently
    for d in reversed(recent_digits):
        if d in least_popular:
            return d

    return 0


def price_trend(prices, window=5, min_strength=0.02):
    """
    Analyzes price history for trend direction and strength.
    Returns: "up", "down", or "flat"
    - window: number of recent prices to consider for trend calculation
    - min_strength: minimum average change per tick to consider a trend
    """
    if len(prices) < max(3, window):
        return "flat"

    # Use only the last 'window' prices
    recent = list(prices)[-window:]
    diffs = [recent[i+1] - recent[i] for i in range(len(recent)-1)]
    avg_change = sum(diffs) / len(diffs)

    # Calculate standard deviation to measure volatility
    mean = sum(recent) / len(recent)
    variance = sum((p - mean) ** 2 for p in recent) / len(recent)
    stddev = variance ** 0.5

    # Require trend to be stronger than a fraction of volatility
    min_trend = max(min_strength, stddev * 0.25)

    if avg_change > min_trend:
        return "up"
    elif avg_change < -min_trend:
        return "down"
    else:
        return "flat"


def confidence_score(current_tick, prev_tick, last_digits, loss_streak):
    """
    Calculates a confidence score for entering a trade based on:
    - Price momentum and volatility
    - Digit patterns and streaks
    - Loss streak (risk management)
    - Recent trend direction
    Returns an integer score (higher = more confident).
    """
    score = 0

    # Ensure last_digits is a list
    last_digits = list(last_digits)

    # 1. Price momentum: strong move in the last tick
    price_change = abs(current_tick - prev_tick)
    if price_change > 0.10:
        score += 2
    elif price_change > 0.05:
        score += 1

    # 2. Volatility: recent price swings
    if len(last_digits) >= 10:
        volatility = max(last_digits) - min(last_digits)
        if volatility >= 5:
            score += 1

    # 3. Digit pattern: avoid repeated digits or streaks
    if len(last_digits) >= 3:
        if last_digits[-1] != last_digits[-2] and last_digits[-2] != last_digits[-3]:
            score += 1
        if len(last_digits) >= 4 and last_digits[-1] not in last_digits[-4:-1]:
            score += 1

    # 4. Loss streak: penalize if on a losing streak
    if loss_streak == 0:
        score += 2
    elif loss_streak == 1:
        score += 1
    elif loss_streak >= 3:
        score -= 1

    # 5. Trend confirmation: check if last digit is moving in a favorable direction
    if len(last_digits) >= 5:
        recent = last_digits[-5:]
        if recent[-1] > min(recent):
            score += 1

    # 6. Bonus: if price change and digit pattern both strong
    if price_change > 0.10 and len(last_digits) >= 3 and last_digits[-1] != last_digits[-2]:
        score += 1

    return max(0, min(score, 6))



def adaptive_bet_size(base_bet, loss_streak, volatility, max_bet_multiplier=2.5, min_bet=1):
    """
    Advanced adaptive bet sizing:
    - Considers loss streak, volatility, and recent win/loss ratio.
    - Uses exponential scaling for higher loss streaks, but caps max bet.
    - Reduces bet if volatility is very low or profit goal is near.
    """
    global daily_profit, DAILY_PROFIT_GOAL, wins, losses, total_trades

    # Calculate win rate for dynamic risk adjustment
    win_rate = (wins / total_trades) if total_trades > 0 else 0.5

    # If close to profit goal, reduce risk
    profit_buffer = DAILY_PROFIT_GOAL - daily_profit
    if profit_buffer < base_bet * 2:
        return max(min_bet, round(base_bet * 0.5, 2))

    # Volatility adjustment (normalize to [0.8, 1.5])
    vol_factor = 1 + min(max(volatility, 0.05), 0.5)  # Clamp volatility
    if volatility < 0.08:
        vol_factor = 0.8
    elif volatility > 0.25:
        vol_factor = 1.5

    # Loss streak scaling (exponential, but capped)
    streak_factor = 1.0
    if loss_streak == 0:
        streak_factor = 1.0
    elif loss_streak == 1:
        streak_factor = 1.2
    elif loss_streak == 2:
        streak_factor = 1.5
    elif loss_streak == 3:
        streak_factor = 1.9
    elif loss_streak > 3:
        streak_factor = min(2.5, 1.9 + 0.2 * (loss_streak - 3))

    # Win rate adjustment: if win rate is low, be more conservative
    if win_rate < 0.45:
        streak_factor *= 0.85
    elif win_rate > 0.65:
        streak_factor *= 1.1

    # Final bet calculation
    bet = base_bet * streak_factor * vol_factor

    # Cap bet to avoid runaway risk
    bet = min(bet, base_bet * max_bet_multiplier)
    bet = max(min_bet, round(bet, 2))
    return bet
failed_trade_conditions = []

def record_failed_trade_condition(current_price, hot_digits, price_history, loss_streak, barrier=3):
    global failed_trade_conditions, last_price, last_digits

    price_str = str(current_price).replace('.', '')
    last_digit = int(price_str[-1])
    volatility = max(abs(price_history[i] - price_history[i - 1]) for i in range(1, len(price_history))) if len(price_history) > 1 else 0
    trend = price_trend(price_history)
    conf = confidence_score(current_price, last_price, last_digits, loss_streak)

    condition = {
        'last_digit': last_digit,
        'hot_digits': tuple(sorted(hot_digits)),
        'volatility': round(volatility, 3),
        'trend': trend,
        'confidence': round(conf, 2),
        'loss_streak': loss_streak,
        'barrier': barrier
    }

    # Avoid duplicates
    if condition not in failed_trade_conditions:
        failed_trade_conditions.append(condition)
        print(Fore.RED + f"⚠️ Recorded failed trade condition: {condition}")

def is_condition_failed(current_price, hot_digits, price_history, loss_streak, barrier=3):
    global failed_trade_conditions, last_price, last_digits

    price_str = str(current_price).replace('.', '')
    last_digit = int(price_str[-1])
    volatility = max(abs(price_history[i] - price_history[i - 1]) for i in range(1, len(price_history))) if len(price_history) > 1 else 0
    trend = price_trend(price_history)
    conf = confidence_score(current_price, last_price, last_digits, loss_streak)

    for cond in failed_trade_conditions:
        # Define matching criteria; you can adjust tolerance as needed
        if (cond['last_digit'] == last_digit and
            cond['hot_digits'] == tuple(sorted(hot_digits)) and
            abs(cond['volatility'] - round(volatility, 3)) < 0.01 and
            cond['trend'] == trend and
            abs(cond['confidence'] - round(conf, 2)) < 0.1 and
            cond['loss_streak'] == loss_streak and
            cond['barrier'] == barrier):
            print(Fore.YELLOW + "⏭ Current conditions match a previously failed trade, skipping trade.")
            return True
    return False

def should_trade(current_price, hot_digits, price_history, loss_streak, barrier=3):
    global last_price, consecutive_losses, daily_profit, wins, losses, total_trades, last_digits
    
    if is_condition_failed(current_price, hot_digits, price_history, loss_streak, barrier):
      return False
    if last_price is None:
        return False

    price_str = str(current_price).replace('.', '')
    last_digit = int(price_str[-1])

    # Calculate volatility
    volatility = max(abs(price_history[i] - price_history[i - 1]) for i in range(1, len(price_history))) if len(price_history) > 1 else 0
    high_volatility = volatility > 0.15

    # Dynamic forbidden digits based on context
    forbidden_digits = [0, 1, 2]
    if loss_streak >= 2 or not high_volatility:
        forbidden_digits = [0, 1, 2, 3]
    if high_volatility and loss_streak == 0:
        forbidden_digits = [0, 1, 3]
    if daily_profit > 0.8 * DAILY_PROFIT_GOAL:
        forbidden_digits = [0, 1, 2, 3, 4]

    # Skip if last digit forbidden
    if last_digit in forbidden_digits:
        print(Fore.YELLOW + f"⏭ Last digit {last_digit} is forbidden ({forbidden_digits}), skipping trade.")
        return False

    # Explicit check for 'digit over' trade: last digit must be strictly greater than barrier
    if last_digit <= barrier:
        print(Fore.YELLOW + f"⏭ Last digit {last_digit} not greater than barrier {barrier}, skipping 'digit over' trade.")
        return False

    # Avoid hot digits unless favorable conditions
    if last_digit in hot_digits and not (loss_streak == 0 or high_volatility):
        print(Fore.YELLOW + f"⏭ Last digit {last_digit} is a hot digit and conditions not favorable, skipping trade.")
        return False

    # Prefer digits just above barrier, allow higher if volatility or trend strong
    if last_digit == barrier + 1 and loss_streak == 0:
        print(Fore.GREEN + f"🟢 Last digit {last_digit} just above barrier {barrier} with no loss streak, considering trade.")
    elif last_digit > barrier + 1 or (high_volatility and last_digit > barrier):
        print(Fore.GREEN + f"🟢 Last digit {last_digit} well above barrier or high volatility, considering trade.")
    else:
        print(Fore.YELLOW + f"⏭ Last digit {last_digit} only slightly above barrier, skipping trade for caution.")
        return False

    # Trend analysis
    trend = price_trend(price_history)
    if trend == "flat":
        print(Fore.YELLOW + "🔄 Price trend is flat, skipping trade.")
        return False
    if (trend == "up" and current_price <= last_price) or (trend == "down" and current_price >= last_price):
        print(Fore.YELLOW + "⏭ Price momentum does not match trend, skipping trade.")
        return False

    # Confidence scoring with dynamic threshold
    conf = confidence_score(current_price, last_price, last_digits, loss_streak)
    dynamic_threshold = barrier + 1
    if loss_streak > 1 or not high_volatility:
        dynamic_threshold += 1

    # Adjust threshold for daily profit and win rate
    if daily_profit > 0.8 * DAILY_PROFIT_GOAL:
        dynamic_threshold += 1
    win_rate = (wins / total_trades) if total_trades > 0 else 0.5
    if win_rate < 0.45:
        dynamic_threshold += 1

    # Explicit confidence check for 'digit over' trade
    over_confidence_threshold = barrier + 2
    if conf < over_confidence_threshold:
        print(Fore.YELLOW + f"⏭ Confidence score {conf} below 'digit over' threshold {over_confidence_threshold}, skipping trade.")
        return False

    # Risk management: pause if too many consecutive losses
    if consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
        print(Fore.RED + f"🚫 Max consecutive losses ({MAX_CONSECUTIVE_LOSSES}) reached, pausing trading.")
        return False

    # Avoid trading if price change or volatility too low
    if abs(current_price - last_price) < 0.01 or volatility < 0.03:
        print(Fore.LIGHTBLACK_EX + "⏭ Price change or volatility too low, skipping trade.")
        return False

    print(Fore.GREEN + f"🟢 All conditions met. Confidence score {conf} meets threshold {dynamic_threshold}, proceeding with 'digit over' trade.")
    return True

def check_trade_result(contract, current_price, hot_digits, price_history, loss_streak, barrier=3):
    global account_balance, daily_profit, total_trades, wins, losses
    global current_bet, starting_balance, max_drawdown, consecutive_losses

    buy_price = contract.get("buy_price", 0)
    sell_price = contract.get("sell_price", 0)
    profit = sell_price - buy_price
    account_balance = contract.get("balance_after", account_balance)

    if starting_balance == 0.0:
        starting_balance = account_balance - profit

    total_trades += 1
    daily_profit = account_balance - starting_balance
    max_drawdown = min(max_drawdown, daily_profit)

    win_rate = (wins / total_trades) if total_trades > 0 else 0.5
    risk_factor = 1.0
    if win_rate < 0.45:
        risk_factor = 0.85
    elif win_rate > 0.65:
        risk_factor = 1.15

    if profit > 0:
        wins += 1
        consecutive_losses = 0
        print(Fore.GREEN + f"✅ Trade WON! Profit: {profit:.2f}")
        current_bet = max(BASE_BET, round(BASE_BET * risk_factor, 2))
        loss_streak = 0
    elif profit < 0:
        losses += 1
        consecutive_losses += 1
        print(Fore.RED + f"❌ Trade LOST. Loss: {-profit:.2f}")
        loss_streak += 1
        # Record failed trade conditions on loss
        record_failed_trade_condition(current_price, hot_digits, price_history, loss_streak, barrier)

        if loss_streak <= MAX_LOSS_STREAK and win_rate > 0.40:
            current_bet = min(current_bet * (1.05 + 0.02 * loss_streak) * risk_factor, BASE_BET * 2.5)
            print(Fore.YELLOW + f"🔺 Loss streak {loss_streak}, increasing bet to {current_bet:.2f}")
        else:
            print(Fore.RED + "⚠️ Max loss streak or low win rate, resetting bet.")
            current_bet = max(BASE_BET, round(BASE_BET * risk_factor, 2))
            loss_streak = 0
    else:
        print(Fore.YELLOW + "🟡🟡 Trade break-even or no profit/loss.")

    print(Fore.CYAN + f"📊 Updated Balance: {account_balance:.2f}")
    print_statistics()


def print_statistics():
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    print(f"📊 Trades: {total_trades} | ✅ Wins: {wins} | ❌ Losses: {losses} | 🎯 Win %: {win_rate:.2f}% | 📈 Profit: {daily_profit:.2f}\n")


def on_message(ws, message):
    global last_price, authorized, trade_in_progress, current_proposal_id
    global server_time, last_trade_time, account_balance
    global current_bet, loss_streak, daily_profit, price_history

    response = json.loads(message)
    msg_type = response.get('msg_type')

    if msg_type == 'authorize':
        print(Fore.GREEN + "✅ Authorized! Subscribing to ticks...")
        authorized = True
        ws.send(json.dumps({"balance": 1, "subscribe": 1}))
        ws.send(json.dumps({"ticks": SYMBOL}))

    elif msg_type == 'balance':
        account_balance = response['balance']['balance']
        print(Fore.YELLOW + f"💰 Account Balance: {account_balance}")

    elif msg_type == 'tick':
        tick = response['tick']
        current_price = tick['quote']
        server_time = tick['epoch']

        print(Fore.CYAN + f"📈 Tick price: {current_price}")
        print_last_digit(current_price)

        price_str = str(current_price).replace('.', '')
        last_digit = int(price_str[-1])
        last_two_digits = extract_last_two_digits(current_price)
        print(Fore.BLUE + f"🔢 Last 2 digits: {last_two_digits}")

        last_digits.append(last_digit)
        price_history.append(current_price)

        barrier = get_least_popular_digit_below_3(last_digits)

        if last_digit <= barrier:
            print(Fore.YELLOW + f"⏭ Last digit {last_digit} not greater than barrier {barrier}, skipping trade.")
            last_price = current_price
            return

        if last_price is None:
            last_price = current_price
            return

        if time.time() - last_trade_time < COOLDOWN_SECONDS:
            print(Fore.LIGHTBLACK_EX + "⏳ Cooldown active, skipping trade.")
            return

        hot_digits = detect_hot_digits(last_digits)
        volatility = max(abs(price_history[i] - price_history[i - 1]) for i in range(1, len(price_history))) if len(price_history) > 1 else 0

        if not should_trade(current_price, hot_digits=hot_digits, price_history=price_history, loss_streak=loss_streak):
            print(Fore.RED + f"❌ No trade signal. Last: {last_price}, Current: {current_price}")
            last_price = current_price
            return

        if last_digit in hot_digits:
            print(Fore.MAGENTA + f"🔥 Hot digit detected: {last_digit}, trading...")

        current_bet = adaptive_bet_size(BASE_BET, loss_streak, volatility)
        print(Fore.YELLOW + f"💵 Betting amount adjusted to: {current_bet:.2f}")

        proposal_request = {
            "proposal": 1,
            "amount": round(current_bet, 2),
            "basis": "stake",
            "contract_type": "DIGITOVER",
            "currency": "USD",
            "symbol": SYMBOL,
            "barrier": str(barrier),
            "duration": 1,
            "duration_unit": "t",
            "product_type": "basic"
        }
        ws.send(json.dumps(proposal_request))
        trade_in_progress = True
        last_trade_time = time.time()

        last_price = current_price

    elif msg_type == 'proposal':
        if 'error' in response:
            print(Fore.RED + "❌ Proposal error:", response['error']['message'])
            trade_in_progress = False
            return

        proposal = response['proposal']
        current_proposal_id = proposal['id']
        ask_price = proposal['ask_price']
        print(Fore.LIGHTBLUE_EX + f"📝 Received proposal id: {current_proposal_id}, ask price: {ask_price}")

        buy_request = {
            "buy": current_proposal_id,
            "price": ask_price,
            "subscribe": 1
        }
        ws.send(json.dumps(buy_request))

    elif msg_type == 'buy':
        if "error" in response:
            print(Fore.RED + "❌ Buy error:", response["error"]["message"])
        else:
            contract = response['buy']
            contract_id = contract['contract_id']
            print(Fore.GREEN + f"🛒 Trade executed. Contract ID: {contract_id}")

            ws.send(json.dumps({
                "proposal_open_contract": 1,
                "contract_id": contract_id,
                "subscribe": 1
            }))

        trade_in_progress = False

    elif msg_type == 'proposal_open_contract':
        contract = response.get('proposal_open_contract')
        if contract and contract.get('is_sold'):
            # Ensure current_price, hot_digits, and price_history are defined
            try:
                _current_price = current_price
            except NameError:
                _current_price = last_price if last_price is not None else 0
            try:
                _hot_digits = hot_digits
            except NameError:
                _hot_digits = detect_hot_digits(last_digits)
            try:
                _price_history = price_history
            except NameError:
                _price_history = deque()
            check_trade_result(contract, _current_price, _hot_digits, _price_history, loss_streak, barrier=3)

def on_error(_ws, error):
    print("❗ WebSocket error:", error)

def on_close(_ws, _close_status_code, _close_msg):
    print("🔌 WebSocket closed.")


def on_open(ws):
    print("🔑 Authorizing...")
    ws.send(json.dumps({"authorize": API_TOKEN}))


def on_error(ws, error):
    print("❗ WebSocket error:", error)


def on_close(ws, close_status_code, close_msg):
    print("🔌 WebSocket closed.")


if __name__ == "__main__":
    
    socket = "wss://ws.binaryws.com/websockets/v3?app_id=1089"
    ws = websocket.WebSocketApp(socket,
                                on_open=on_open,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)
    ws.run_forever()
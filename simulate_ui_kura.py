import json
import threading
import paho.mqtt.client as mqtt
from dataclasses import dataclass
from loguru import logger

BROKER     = "localhost"
PORT       = 1883
TOPIC_RECV = "handwash/process"
TOPIC_SEND = "handwash/system"
TOPIC_CTRL = "handwash/system"   # 訂閱來自系統的控制指令（如 Reset）
JSON_PATH  = "wash_steps.json"
TIME_SCALE = 0.1666

# 需求 1: 滿足 wash_steps.json 定義的步驟條件後, 延遲多久才發 NextStep
NEXT_STEP_DELAY = 0.2
# 需求 2: 收到 step12 trigger=True 後, 延遲多久才發 Logout
LOGOUT_DELAY = 0.5


def _need_lateral(items: list) -> tuple:
    """
    依 items 文字判斷該 step 是否需要分左右次數 / 左右時間
    (跟 utils/tool.py 的 parse_lateral_flags 邏輯保持一致)
    """
    joined = ''.join(items)
    need_count = ('左手回数' in joined) or ('右手回数' in joined)
    need_time = ('左手時間' in joined) or ('右手時間' in joined)

    # 有左右回數, 時間也一定要分左右
    if need_count:
        need_time = True

    return need_count, need_time


with open(JSON_PATH, encoding="utf-8") as f:
    _data = json.load(f)

STAGES = []
for s in _data["stages"]:
    need_lr_count, need_lr_time = _need_lateral(s.get("items", []))
    STAGES.append({
        "id":            s["id"],
        "washcountmax":  int(s["washcountmax"]),
        "washtimemax":   float(s["washtimemax"]) * TIME_SCALE,
        "need_lr_count": need_lr_count,
        "need_lr_time":  need_lr_time,
    })

logger.info(f'Config: {STAGES}')


def _publish(client, payload: str):
    """實際發送 MQTT 訊息, 在 Timer 的背景執行緒裡執行, 不會卡住主執行緒"""
    client.publish(TOPIC_SEND, payload)
    logger.info(f"[SEND] topic={TOPIC_SEND} payload={payload}")


@dataclass
class StepState:
    stage_idx: int = 0
    triggered: bool = False

    # 不分左右的 step 用這組
    washcount: int = 0
    washtime: float = 0.0

    # 需要分左右的 step, 左右各自獨立累計 (門檻共用, 但要各自達標)
    left_washcount: int = 0
    left_washtime: float = 0.0
    right_washcount: int = 0
    right_washtime: float = 0.0

    @property
    def stage(self) -> dict:
        return STAGES[self.stage_idx]

    def update(self, category, washcount: int, washtime: float, triggered: bool):
        self.triggered = triggered
        if category == 'left':
            self.left_washcount, self.left_washtime = washcount, washtime
        elif category == 'right':
            self.right_washcount, self.right_washtime = washcount, washtime
        else:
            self.washcount, self.washtime = washcount, washtime

    def is_satisfied(self) -> bool:
        if not self.triggered:
            return False

        s = self.stage
        if s["need_lr_count"]:
            count_ok = (s["washcountmax"] == 0) or (
                self.left_washcount >= s["washcountmax"] and
                self.right_washcount >= s["washcountmax"]
            )
        else:
            count_ok = (s["washcountmax"] == 0) or (self.washcount >= s["washcountmax"])

        if s["need_lr_time"]:
            time_ok = (s["washtimemax"] == 0) or (
                self.left_washtime >= s["washtimemax"] and
                self.right_washtime >= s["washtimemax"]
            )
        else:
            time_ok = (s["washtimemax"] == 0) or (self.washtime >= s["washtimemax"])

        return count_ok and time_ok


states: dict = {
    "left":  StepState(),
    "right": StepState(),
}


def on_connect(client, userdata, flags, rc):
    logger.info(f"MQTT connected (rc={rc}), subscribed → {TOPIC_RECV}, {TOPIC_CTRL}")
    client.subscribe(TOPIC_RECV)
    client.subscribe(TOPIC_CTRL)


def handle_reset(client, data: dict):
    """
    處理 Reset 指令。
    若 time == 0，視為逾時
    """
    side = data.get("side", "")
    if side not in states:
        logger.warning(f"[Reset] unknown side: {side}")
        return

    remain = float(data.get("time", 0))
    logger.info(f"[RECV] Reset side={side} remain={remain}")

    if remain == 0:
        states[side] = StepState()   # 重置狀態回第一步


def on_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload.decode())
    except json.JSONDecodeError:
        logger.warning(f"invalid JSON: {msg.payload}")
        return

    cmd = data.get("cmd")

    if cmd == "Reset":
        handle_reset(client, data)
        return

    # 自己發出的指令（Logout / NextStep）會被自己收到，直接忽略
    if cmd in ("Logout", "NextStep", "AILogin", "Reset", "ResetCancel", 'Login', 'Alarm', 'AlarmCancel', 'Trigger'):
        return

    side = data.get("side")
    if side not in states:
        logger.warning(f"unknown side: {side}")
        return

    state     = states[side]
    step_id   = int(data["step_id"].replace("Step", ""))
    washcount = int(data["washcount"])
    washtime  = float(data["washtime"])
    triggered = str(data["trigger"]).lower() in ("true", "1")
    category  = data.get("category")   # None / "left" / "right"

    current_id = state.stage["id"]
    if step_id != current_id:
        logger.debug(f"[{side}] skip Step{step_id}, waiting for Step{current_id}")
        return

    state.update(category, washcount, washtime, triggered)

    logger.info(
        f"[RECV] side={side} step={current_id:02d} category={category} "
        f"trigger={triggered} "
        f"count={washcount} (L={state.left_washcount}/R={state.right_washcount}) "
        f"/{state.stage['washcountmax']} "
        f"time={washtime:.2f} (L={state.left_washtime:.2f}/R={state.right_washtime:.2f}) "
        f"/{state.stage['washtimemax']:.2f}"
    )

    if state.is_satisfied():
        advance(client, side)


def advance(client, side: str):
    state      = states[side]
    current_id = state.stage["id"]
    next_idx   = (state.stage_idx + 1) % len(STAGES)
    next_id    = STAGES[next_idx]["id"]

    is_reset = next_idx == 0
    label    = "reset → Step 1" if is_reset else f"→ Step {next_id}"
    logger.success(f"[{side}] Step {current_id:02d} satisfied, {label}")

    if current_id == 12:
        cmd, delay = {"cmd": "Logout", "side": side}, LOGOUT_DELAY
    else:
        cmd, delay = {"cmd": "NextStep", "step_id": f"Step{next_id}", "side": side}, NEXT_STEP_DELAY

    payload = json.dumps(cmd)
    # 用 Timer 讓延遲發送在背景執行, 不要卡住 MQTT 的訊息處理迴圈
    # (否則另一側 / 其他訊息會在這段延遲期間被延後處理)
    timer = threading.Timer(delay, _publish, args=(client, payload))
    timer.daemon = True
    timer.start()

    # 立刻切到下一步的狀態, 避免同一個 step 的訊息在等待發送期間重複觸發 advance()
    states[side] = StepState(stage_idx=next_idx)


def main():
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(BROKER, PORT, keepalive=60)
    logger.info(f"starting, total {len(STAGES)} stages")
    client.loop_forever()


if __name__ == "__main__":
    main()
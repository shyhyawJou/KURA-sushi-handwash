import json
import paho.mqtt.client as mqtt
from dataclasses import dataclass
from loguru import logger

BROKER     = "localhost"
PORT       = 1883
TOPIC_RECV = "handwash/process"
TOPIC_SEND = "handwash/system"
JSON_PATH  = "wash_steps.json"
TIME_SCALE = 0.1666


with open(JSON_PATH, encoding="utf-8") as f:
    _data = json.load(f)

STAGES = [
    {
        "id":           s["id"],
        "washcountmax": int(s["washcountmax"]),
        "washtimemax":  float(s["washtimemax"]) * TIME_SCALE,
    }
    for s in _data["stages"]
]

logger.info(f'Config: {STAGES}')

def make_command(side: str) -> dict:
    return {"cmd": "switch_step", "side": side}

@dataclass
class StepState:
    stage_idx: int   = 0
    triggered: bool  = False
    washcount: int   = 0
    washtime:  float = 0.0

    @property
    def stage(self) -> dict:
        return STAGES[self.stage_idx]

    def is_satisfied(self) -> bool:
        s = self.stage
        count_ok = (s["washcountmax"] == 0) or (self.washcount >= s["washcountmax"])
        time_ok  = (s["washtimemax"]  == 0) or (self.washtime  >= s["washtimemax"])
        return self.triggered and count_ok and time_ok

states: dict[str, StepState] = {
    "left":  StepState(),
    "right": StepState(),
}

def on_connect(client, userdata, flags, rc):
    logger.info(f"MQTT connected (rc={rc}), subscribed → {TOPIC_RECV}")
    client.subscribe(TOPIC_RECV)

def on_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload.decode())
    except json.JSONDecodeError:
        logger.warning(f"invalid JSON: {msg.payload}")
        return

    side = data["side"]
    if side not in states:
        logger.warning(f"unknown side: {side}")
        return

    state     = states[side]
    step_id   = int(data["step_id"].replace("Step", ""))
    washcount = int(data["washcount"])
    washtime  = float(data["washtime"])
    triggered = str(data["trigger"]).lower() in ("true", "1")

    current_id = state.stage["id"]
    if step_id != current_id:
        logger.debug(f"[{side}] skip Step{step_id}, waiting for Step{current_id}")
        return

    state.triggered = triggered
    state.washcount = washcount
    state.washtime  = washtime

    logger.info(
        f"[RECV] side={side} step={current_id:02d} "
        f"trigger={triggered} "
        f"count={washcount}/{state.stage['washcountmax']} "
        f"time={washtime}/{state.stage['washtimemax']}"
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

    cmd = make_command(side)
    payload = json.dumps(cmd)
    client.publish(TOPIC_SEND, payload)
    logger.info(f"[SEND] topic={TOPIC_SEND} payload={payload}")

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
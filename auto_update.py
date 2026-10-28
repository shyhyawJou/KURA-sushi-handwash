"""
自動重試更新程式
- 連不上 / 更新失敗 的主機會一直重試，直到成功
- 成功的主機會記錄在 update_state.json，之後不會再更新
- 程式中途關掉再開，也會接續上次進度
- 本地更新資料夾內容有改變（hash 不同）時，會視為新版本，所有主機重新更新
"""
import hashlib
import json
import os
import stat
import time
import traceback

import paramiko

# ===================== 設定區 =====================
MACHINE_FILE = "批量更新/machine_demokit.txt"
LOCAL_DIR = "批量更新/kurasushi_handwash_demokit"   # 要上傳的資料夾
REMOTE_PARENT = "/mnt/reserved"            # 上傳到遠端的位置
STATE_FILE = "update_state.json"
RETRY_INTERVAL = 10                        # 一輪結束後等幾秒再重試失敗的主機
CONNECT_TIMEOUT = 10

# 上傳完成後要在遠端依序執行的指令: (指令, 是否預期要有輸出)
REMOTE_CMDS = [
    ("bash /mnt/reserved/kurasushi_handwash_demokit/restart_ui.sh", True),
    ("systemctl restart kurasushi_handwash_demokit.service", False),
]
# ==================================================


def load_machines(path):
    machines = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            user_host, port = line.split()
            user, host = user_host.split("@")
            machines.append((user, host, int(port)))
    return machines


def dir_hash(path):
    """計算資料夾內容 hash，用來判斷是不是新版本"""
    h = hashlib.md5()
    for root, dirs, files in os.walk(path):
        dirs.sort()
        for name in sorted(files):
            full = os.path.join(root, name)
            h.update(os.path.relpath(full, path).encode())
            with open(full, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
    return h.hexdigest()


# ---------- 進度紀錄 ----------
def load_state(version):
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            state = json.load(f)
        if state.get("version") == version:
            return state
        print("🆕 偵測到更新內容有變動，所有主機重新更新")
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return {"version": version, "done": {}}


def save_state(state):
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    os.replace(tmp, STATE_FILE)  # 原子替換，避免寫到一半當掉造成檔案壞掉


# ---------- 單台機器的動作 ----------
def upload_dir(sftp, local_dir, remote_dir):
    try:
        sftp.mkdir(remote_dir)
    except IOError:
        pass  # 已存在
    for item in os.listdir(local_dir):
        lp = os.path.join(local_dir, item)
        rp = remote_dir + "/" + item
        if os.path.isdir(lp):
            upload_dir(sftp, lp, rp)
        else:
            sftp.put(lp, rp, confirm=True)  # confirm=True 會檢查大小是否一致


def run_cmd(ssh, cmd, has_output):
    """成功回傳 True，失敗回傳 False（判斷邏輯與原本 task_execute 一致）"""
    print(f"  [執行] {cmd}")
    _, stdout, stderr = ssh.exec_command(cmd, timeout=120)
    out = stdout.read().decode("utf-8", errors="ignore").strip()
    err = stderr.read().decode("utf-8", errors="ignore").strip()
    if err:
        print(f"  ❌ stderr: {err}")
        return False
    if has_output and not out:
        print("  ❌ 預期有輸出但沒收到")
        return False
    if not has_output and out:
        print(f"  ❌ 不預期有輸出: {out}")
        return False
    return True


def update_one(user, host, port):
    """完整更新一台機器，全部步驟成功才回傳 True"""
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        ssh.connect(
            host, port=port, username=user, password="",
            timeout=CONNECT_TIMEOUT,
            banner_timeout=CONNECT_TIMEOUT,
            auth_timeout=CONNECT_TIMEOUT,
        )
        print("  ✅ 連線成功")

        # 1. 上傳
        sftp = ssh.open_sftp()
        try:
            base = os.path.basename(LOCAL_DIR.rstrip("/"))
            upload_dir(sftp, LOCAL_DIR, REMOTE_PARENT.rstrip("/") + "/" + base)
        finally:
            sftp.close()
        print("  ✅ 上傳完成")

        # 2. 重啟等指令
        for cmd, has_output in REMOTE_CMDS:
            if not run_cmd(ssh, cmd, has_output):
                return False
        return True

    except Exception as e:
        print(f"  ❌ 失敗: {type(e).__name__}: {e}")
        return False
    finally:
        ssh.close()


# ---------- 主流程 ----------
def main():
    machines = load_machines(MACHINE_FILE)
    version = dir_hash(LOCAL_DIR)
    state = load_state(version)

    round_no = 0
    while True:
        pending = [m for m in machines if f"{m[0]}@{m[1]}:{m[2]}" not in state["done"]]
        if not pending:
            print("\n🎉 所有主機都已更新完成")
            return

        round_no += 1
        print(f"\n===== 第 {round_no} 輪，尚未完成 {len(pending)}/{len(machines)} 台 =====")

        for user, host, port in pending:
            key = f"{user}@{host}:{port}"
            print(f"\n🚀 {key}")
            try:
                ok = update_one(user, host, port)
            except Exception:
                traceback.print_exc()
                ok = False

            if ok:
                state["done"][key] = time.strftime("%Y-%m-%d %H:%M:%S")
                save_state(state)  # 每成功一台就立刻存檔
                print(f"  ✅ {key} 更新成功，之後不會再更新")
            else:
                print(f"  ⏳ {key} 失敗，下一輪再試")

        if len(state["done"]) < len(machines):
            print(f"\n💤 等待 {RETRY_INTERVAL} 秒後重試失敗的主機...")
            time.sleep(RETRY_INTERVAL)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 已手動停止，進度已保存在", STATE_FILE)

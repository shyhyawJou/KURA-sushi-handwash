class HandPresenceGate:
    """
    手偵測的去彈跳 (debounce / hysteresis) 狀態機。

    規則：
    - 未進場時：手要「連續出現」滿 stable_delay 秒才算真正進場 (entered)。
      中途手一消失，計時就整個重來 (不是暫停)。
    - 已進場時：手一消失就開始計「消失多久」；滿 stable_delay 秒後進入
      「倒數離場」狀態 (exiting)，這期間手若又連續出現滿 stable_delay 秒，
      就取消倒數 (exit_cancelled)。
    - 倒數離場期間，若消失總時間達到 exit_timeout_fn(now) 回傳的秒數
      (預設 0，代表不額外倒數，滿 stable_delay 就直接算離場)，才算真正離場 (exited)。

    這個類別不含任何 MQTT / 登入邏輯，純狀態機，方便重複使用與獨立測試。
    使用方式：每個 frame 呼叫一次 update(now, has_hand)，再檢查
    entered / exit_started / exit_cancelled / exited 這幾個旗標（只在觸發的
    那一次 update() 呼叫內為 True，下一次呼叫前會被清空）決定要做什麼。
    """

    def __init__(self, stable_delay, exit_timeout_fn=None):
        """
        stable_delay: 手要連續出現/消失多久才算數 (秒)
        exit_timeout_fn: callable(now) -> 額外的離場倒數秒數。
                          預設為 None，等同於 lambda now: 0 (不額外倒數)。
        """
        self.stable_delay = stable_delay
        self.exit_timeout_fn = exit_timeout_fn or (lambda now: 0)
        self.reset()

    def reset(self):
        self.active = False        # 是否已經處於「進場」狀態
        self.exiting = False       # 是否正在倒數離場
        self.exit_elapsed = 0.
        self.remain = 0.
        self._has_hand_since = None
        self._no_hand_since = None
        self._clear_events()

    def force_active(self, active: bool):
        """
        由外部強制設定進場/離場狀態 (例如 scanner 登入/登出這種不經過手偵測的路徑)，
        同時清空所有計時器, 避免跟接下來的手偵測狀態打架。
        """
        self.active = active
        self.exiting = False
        self._has_hand_since = None
        self._no_hand_since = None
        self._clear_events()

    def _clear_events(self):
        self.entered = False
        self.exit_started = False
        self.exit_cancelled = False
        self.exited = False

    def _has_hand_start(self, now):
        if self._has_hand_since is None:
            self._has_hand_since = now
        return self._has_hand_since

    def _no_hand_start(self, now):
        if self._no_hand_since is None:
            self._no_hand_since = now
        return self._no_hand_since

    def update(self, now, has_hand):
        self._clear_events()

        if not self.active:
            if has_hand:
                if now - self._has_hand_start(now) >= self.stable_delay:
                    self.active = True
                    self._has_hand_since = None
                    self._no_hand_since = None
                    self.entered = True
            else:
                self._has_hand_since = None
            return

        # active == True: 偵測是否要開始/取消/完成離場倒數
        if self.exiting:
            if not has_hand:
                self._has_hand_since = None
            if now - self._has_hand_start(now) >= self.stable_delay:
                self.exiting = False
                self._has_hand_since = None
                self._no_hand_since = None
                self.exit_cancelled = True
        elif has_hand:
            self._no_hand_since = None

        self.exit_elapsed = now - self._no_hand_start(now)
        if self.exit_elapsed >= self.stable_delay:
            just_started = not self.exiting
            self.exiting = True
            self.exit_started = just_started

            timeout = self.exit_timeout_fn(now)
            self.remain = max(timeout - max(self.exit_elapsed - self.stable_delay, 0), 0)
            if self.remain == 0:
                self.active = False
                self.exiting = False
                self._has_hand_since = None
                self._no_hand_since = None
                self.exited = True

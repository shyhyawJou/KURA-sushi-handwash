# 先清殘留，確保 HID 可用
pkill -f handwash_qt.py; sleep 2

# 停用 Flet
#systemctl disable --now washgui.service

# 啟用 PyQt5
systemctl enable  --now washqt_stream.service
sleep 6

# 應為 active
systemctl is-active washqt_stream.service
journalctl -u washqt_stream.service -n 15 --no-pager | grep -iE "Scanner|Font|啟動"
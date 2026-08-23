set -euo pipefail

DATA_DIR='kurasushi_handwash'
SERVICES=(kurasushi_handwash.service disk_monitor.service)

unzip -o ${DATA_DIR}.zip -d /mnt/reserved

pip3 install \
    opencv-python \
    numpy \
    uvicorn \
    fastapi \
    loguru \
    ntplib

cd /mnt/reserved/${DATA_DIR}

cp -f "${SERVICES[@]}" /etc/systemd/system

systemctl daemon-reload
for s in "${SERVICES[@]}"; do
    systemctl enable "$s"
    systemctl restart "$s"
    systemctl status "$s" --no-pager
done

crontab cron_file
systemctl restart crond

#rm -f ~/${DATA_DIR}.zip

ps aux | grep python

sleep 2

# 檢查 log
printf "\n********************************* kura 主程式 log *********************************\n"
tail -n 20 /mnt/reserved/log/$(date +%Y%m%d)-*.log

printf "\n********************************* kura 主程式 ERROR *********************************\n"
grep ERROR -aE /mnt/reserved/log/$(date +%Y%m%d)-*.log || true

printf "\n********************************* disk_monitor log *********************************\n"
tail -n 20 /mnt/reserved/disk_monitor.log

#printf "\n********************************* usb_autocopy log *********************************\n"
#tail -n 20 /mnt/reserved/usb_autocopy.log
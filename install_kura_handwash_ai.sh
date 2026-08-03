set -euo pipefail

DATA_DIR='kurasushi_handwash'
SERVICES=(kurashushi_handwash.service disk_monitor.service usb_autocopy.service)

unzip -o ~/${DATA_DIR}.zip -d /mnt/reserved

pip3 install \
    opencv-python \
    numpy \
    uvicorn \
    fastapi \
    loguru

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

rm -f ~/${DATA_DIR}.zip

ps aux | grep python
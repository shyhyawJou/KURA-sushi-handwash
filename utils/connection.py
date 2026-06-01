import paho.mqtt.client as mqtt
import time
import traceback
from loguru import logger
import json



class MQTT:
    def __init__(self, ip, port, topic, qos, client_id=None, reconnect_interval=5, 
                 sub_cmds=None, callbacks=None, **kwargs):
        self.broker = ip
        self.port = port
        self.sub_topics = topic['subscribe']
        self.pub_topics = topic['publish']
        self.sub_cmds = set(sub_cmds) if sub_cmds is not None else None
        self.qos = qos
        self.reconnect_interval = reconnect_interval
        self.callbacks = callbacks

        #self.client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2, 
        #                          client_id=client_id)
        self.client = mqtt.Client(client_id=client_id)        
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message
        self.client.reconnect_delay_set(min_delay=reconnect_interval, max_delay=30)
        self.client.on_connect_fail = self._on_connect_fail

        self.is_running = True

        # 自動連線
        self.connect()

    def _on_connect(self, client, userdata, flags, rc, *args, **kwargs):
        if rc == 0:
            logger.success(f"Connected successfully to MQTT Broker [{self.broker}:{self.port}]")
            self.subscribe()
        else:
            logger.error(f"failed to connect [{self.broker}:{self.port}] with code {rc}")

    def _on_connect_fail(self, client, userdata, *args, **kwargs):
        # 因為只要斷線就會觸發, 所以控制只有第一次才會印
        logger.error(f'MQTT connection is failed! Try again after {self.reconnect_interval} (s)')

    def _on_disconnect(self, client, userdata, rc, *args, **kwargs):
        if rc == 0:
            logger.success('disconnected to MQTT broker !')
        else:
            logger.warning(f"Disconnected from Broker, code: {rc}, "
                           f"connect to MQTT Broker again after {self.reconnect_interval} (s)")

    def _on_message(self, client, userdata, msg):
        try:
            payload = msg.payload.decode("utf-8")
            cmd = json.loads(payload)
            if cmd['cmd'] not in self.sub_cmds:
                return
            logger.info(f"MQTT [{msg.topic}] received: {payload}")
            if self.callbacks is None:
                return

            side = cmd['side'].lower()
            if cmd['cmd'] == 'ResetFinish':
                self.callbacks['ResetFinish'][side](cmd)
            elif cmd['cmd'] == 'Login':
                self.callbacks['Login'](cmd)
            elif cmd['cmd'] == 'switch_step':
                self.callbacks['switch_step'][side](cmd)
        except:
            logger.error(f"{traceback.format_exc()}, "
                         f"MQTT [{msg.topic}] received: {payload}")

    def connect(self):
        """建立連線並啟動背景迴圈"""
        try:
            #self.client.connect_async(self.broker, self.port, 60)
            self.client.connect(self.broker, self.port, 60)
            self.client.loop_start()
        except:
            logger.error(f'Failed to connect to MQTT Broker [{self.broker}:{self.port}], '
                         f'connect to MQTT Broker again after {self.reconnect_interval} (s)')

    def disconnect(self):
        """中斷連線並停止背景迴圈"""
        logger.info('app ask to disconnect to MQTT Broker ...')
        self.is_running = False
        self.client.disconnect()
        self.client.loop_stop()

    def subscribe(self):
        """訂閱主題"""
        for topic in self.sub_topics.values():
            self.client.subscribe(topic, qos=self.qos)
            logger.success(f"Subscribed to topic: {topic}")

    def publish(self, topic, message, log_level):
        """發送訊息"""
        published_msg = None
        try:
            msg_json = json.dumps(message, ensure_ascii=False)
            result = self.client.publish(topic, msg_json, qos=self.qos)
            # 檢查是否發送成功
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                published_msg = msg_json
                level = log_level.upper()
                logger.log(level, f"publish {msg_json} to MQTT [{topic}]")
            elif self.is_running:
                logger.error(f"Failed to publish message to MQTT [{topic}]")
        except json.JSONDecodeError:
            logger.error(f'cannot decode message: {message}')

        return published_msg
    
    def add_callbacks(self, callbacks):
        self.callbacks = callbacks
        logger.success('MQTT add callbacks !')
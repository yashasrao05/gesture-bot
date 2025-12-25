import asyncio
import websockets
import json
import serial
import time

WS_URL = "ws://127.0.0.1:8082"
SERIAL_PORT = "COM11"  
BAUDRATE = 115200


GESTURE_MAP = {
    "UP": "F",
    "DOWN": "B",
    "LEFT": "L",
    "RIGHT": "R",
    "STOP": "S"
}


ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1)
time.sleep(2)
print("Connected to Arduino")

async def run():
    async with websockets.connect(WS_URL) as ws:
        print("Connected to Gesture Server")

        async for msg in ws:
            data = json.loads(msg)

            gesture = data.get("gesture", "").upper()
            cmd = GESTURE_MAP.get(gesture, "S")

            ser.write((cmd + "\n").encode())
            print(f"Gesture: {gesture} → Sent: {cmd}")

asyncio.run(run())

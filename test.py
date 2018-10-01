import time
import asyncio
import websockets

async def hello():
    async with websockets.connect(
            'ws://localhost:10309') as websocket:
        name = input("What's your name? ")

        await websocket.send(name)
        print(f"> {name}")

        greeting = await websocket.recv()
        print(f"< {greeting}")

while True:
    try:
        asyncio.get_event_loop().run_until_complete(hello())
    except:
        print("failed")
    time.sleep(5)
    
import websockets
import json
import asyncio
import time
import pandas as pd
from datetime import datetime

async def get_all_options_list(currency="any", kind="option", expired=False):
    async with websockets.connect(
        'wss://www.deribit.com/ws/api/v2',
        max_size=10 * 1024 * 1024
    ) as websocket:
        msg = {
            "jsonrpc": "2.0",
            "id": 833,
            "method": "public/get_instruments",
            "params": {
                "currency": currency,
                "kind": kind,
                "expired": expired
            }
        }
        
        await websocket.send(json.dumps(msg))
        response = await websocket.recv()
        result = json.loads(response)['result']
        
        instruments = {}
        for instrument in result:
            base_curr = instrument['base_currency']
            instr_name = instrument['instrument_name']
            expiry_datetime = datetime.fromtimestamp(instrument['expiration_timestamp']/1000).strftime('%Y-%m-%d %H:%M:%S')
            strike = instrument['strike']
            option_type = instrument['option_type']
            
            if base_curr not in instruments:
                instruments[base_curr] = {}
            if expiry_datetime not in instruments[base_curr]:
                instruments[base_curr][expiry_datetime] = {}
            if strike not in instruments[base_curr][expiry_datetime]:
                instruments[base_curr][expiry_datetime][strike] = {}
            if option_type not in instruments[base_curr][expiry_datetime][strike]:
                instruments[base_curr][expiry_datetime][strike][option_type] = []
            
            instruments[base_curr][expiry_datetime][strike][option_type].append(instr_name)
            
        return instruments

class WebSocketPool:
    def __init__(self, url, pool_size=5):
        self.url = url
        self.pool_size = pool_size
        self.connections = []
        self.locks = []  # Add locks for each connection
        self.current = 0
        
    async def __aenter__(self):
        for _ in range(self.pool_size):
            ws = await websockets.connect(self.url)
            lock = asyncio.Lock()  # Create lock for each connection
            self.connections.append(ws)
            self.locks.append(lock)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        for ws in self.connections:
            await ws.close()
            
    async def get_connection(self):
        # Round-robin connection distribution with lock
        idx = self.current
        self.current = (self.current + 1) % self.pool_size
        return self.connections[idx], self.locks[idx]

async def get_ticker(websocket, lock, instrument_name):
    msg = {
        "jsonrpc" : "2.0",
        "id" : 8106,
        "method" : "public/ticker",
        "params" : {
            "instrument_name" : instrument_name
        }
    }
    
    async with lock:  # Ensure send/recv pair stays together for each connection
        await websocket.send(json.dumps(msg))
        response = await websocket.recv()
        result = json.loads(response)['result']
    
    return {
        'best_bid_price': result.get('best_bid_price', 0),
        'best_ask_price': result.get('best_ask_price', 0),
        'index_price': result.get('index_price', 0),
        'volume_usd': result.get('stats', {}).get('volume_usd', 0),
    }

async def download_options_data(ticker, instruments, max_concurrent=20, progress_callback=None):
    sem = asyncio.Semaphore(max_concurrent)
    results = []
    ticker_data = instruments[ticker]
    tasks = []
    retry_tasks = []
    completed_tasks = 0
    total_tasks = 0
    
    async with WebSocketPool('wss://www.deribit.com/ws/api/v2', pool_size=5) as pool:
        async def download_single_option(expiry_date, strike, option_type, instrument_name):
            nonlocal completed_tasks
            async with sem:
                try:
                    # Get both connection and its lock
                    ws, lock = await pool.get_connection()
                    book_data = await get_ticker(ws, lock, instrument_name)
                    completed_tasks += 1
                    if progress_callback:
                        progress_callback(completed_tasks, total_tasks)
                    return {
                        'expirationDate': expiry_date,
                        'strike': float(strike),
                        'option_type': option_type,
                        'bid': book_data['best_bid_price'],
                        'ask': book_data['best_ask_price'],
                        'index_price': book_data['index_price'],
                        'volume_usd': book_data['volume_usd'],
                    }
                except Exception as e:
                    print(f"Error downloading {instrument_name}: {str(e)}")
                    if "429" in str(e):
                        retry_tasks.append(download_single_option(expiry_date, strike, option_type, instrument_name))
                    return None

        # Create tasks for all options
        for expiry_date in ticker_data:
            for strike in ticker_data[expiry_date]:
                for option_type in ticker_data[expiry_date][strike]:
                    instrument_name = ticker_data[expiry_date][strike][option_type][0]
                    tasks.append(asyncio.create_task(download_single_option(expiry_date, strike, option_type, instrument_name)))
                    total_tasks += 1

        print(f"Created {total_tasks} tasks")  # Debug print

        # Wait for all tasks to complete
        initial_results = await asyncio.gather(*tasks, return_exceptions=True)
        results.extend([r for r in initial_results if r is not None])

        # Handle retry tasks if any
        while retry_tasks:
            print(f"Retrying {len(retry_tasks)} tasks")  # Debug print
            current_tasks = retry_tasks
            retry_tasks = []
            retry_results = await asyncio.gather(*current_tasks, return_exceptions=True)
            results.extend([r for r in retry_results if r is not None])
        
        print(f"Completed {len(results)} out of {total_tasks} tasks")  # Debug print
        
        df = pd.DataFrame(results)
        
        if not df.empty:
            df['expirationDate'] = pd.to_datetime(df['expirationDate'])
            df = df.sort_values(['expirationDate', 'strike'])
        
        return df



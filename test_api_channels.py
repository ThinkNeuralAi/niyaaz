#!/usr/bin/env python
import requests
import json

try:
    response = requests.get('http://localhost:5000/api/get_active_channels?store_id=store_1', timeout=5)
    data = response.json()
    
    if data.get('success'):
        channels = data.get('active_channels', [])
        print(f"✅ SUCCESS: {len(channels)} channels loaded for store_1")
        if channels:
            print("\nFirst 3 channels:")
            for ch in channels[:3]:
                print(f"  - {ch['channel_id']}: {ch.get('modules', [])}")
    else:
        print(f"❌ API error: {data.get('error', 'Unknown')}")
except Exception as e:
    print(f"❌ Connection error: {e}")

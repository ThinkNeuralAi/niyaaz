#!/usr/bin/env python
import json
from pathlib import Path

# Load channels.json
config_path = Path('config/channels.json')
with open(config_path, 'r') as f:
    config = json.load(f)

channels = config.get('channels', [])
enabled_channels = [ch for ch in channels if ch.get('enabled', False)]
store_1_channels = [ch for ch in enabled_channels if ch.get('store_id', 'store_1') == 'store_1']
store_2_channels = [ch for ch in enabled_channels if ch.get('store_id', 'store_1') == 'store_2']

print(f'Total channels in config: {len(channels)}')
print(f'Total enabled channels: {len(enabled_channels)}')
print(f'Store 1 enabled channels ({len(store_1_channels)}):')
for ch in store_1_channels:
    print(f"  - {ch.get('channel_id')}: {ch.get('channel_name')}")
print(f'\nStore 2 enabled channels ({len(store_2_channels)}):')
for ch in store_2_channels:
    print(f"  - {ch.get('channel_id')}: {ch.get('channel_name')}")

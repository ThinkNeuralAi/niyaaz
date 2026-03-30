#!/usr/bin/env python3
"""Quick verification: print operation timing for all stores."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app, db_manager

with app.app_context():
    stores = db_manager.get_all_stores()
    if not stores:
        print("No stores found in database.")
    else:
        print(f"{'Store ID':<12} {'Name':<35} {'Open':<8} {'Close':<8}")
        print("-" * 68)
        for s in stores:
            print(f"{s['store_id']:<12} {s['name']:<35} {s.get('operation_start_time') or 'not set':<8} {s.get('operation_end_time') or 'not set':<8}")

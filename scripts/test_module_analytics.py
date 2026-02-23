"""
Test script to verify that the fixed API handlers return non-zero alert counts.
Run on the server: python scripts/test_module_analytics.py

This simulates what the dashboard does by querying alert_gifs directly
and comparing with what the API handlers would return.
"""
import sys
import os

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import psycopg2
from datetime import datetime, timedelta

def get_connection():
    return psycopg2.connect(
        host=os.getenv('DB_HOST', 'localhost'),
        port=os.getenv('DB_PORT', '5432'),
        dbname=os.getenv('DB_NAME', 'sakshiai'),
        user=os.getenv('DB_USER', 'postgres'),
        password=os.getenv('DB_PASSWORD', 'Postgres123')
    )

def main():
    conn = get_connection()
    cur = conn.cursor()

    print('=' * 70)
    print('  VERIFICATION: Alert Count Fix')
    print('=' * 70)

    modules = [
        {
            'name': 'DressCodeMonitoring',
            'alert_type': 'dresscode_alert',
            'dedicated_table': 'dresscode_alerts',
            'dedicated_col': 'created_at',
        },
        {
            'name': 'TableServiceMonitor',
            'alert_type': 'table_cleanliness_alert',
            'dedicated_table': 'table_cleanliness_violations',
            'dedicated_col': 'created_at',
        },
        {
            'name': 'ServiceDisciplineMonitor',
            'alert_type': 'service_discipline_alert',
            'dedicated_table': 'table_service_violations',
            'dedicated_col': 'created_at',
        },
        {
            'name': 'PPEMonitoring',
            'alert_type': 'ppe_alert',
            'dedicated_table': 'ppe_alerts',
            'dedicated_col': 'created_at',
        },
    ]

    date_7d = datetime.now() - timedelta(days=7)
    date_1d = datetime.now() - timedelta(days=1)

    for mod in modules:
        print(f'\n--- {mod["name"]} ---')

        # Check dedicated table
        try:
            cur.execute(f"SELECT COUNT(*) FROM {mod['dedicated_table']} WHERE {mod['dedicated_col']} >= %s", (date_7d,))
            dedicated_7d = cur.fetchone()[0]
            cur.execute(f"SELECT COUNT(*) FROM {mod['dedicated_table']} WHERE {mod['dedicated_col']} >= %s", (date_1d,))
            dedicated_1d = cur.fetchone()[0]
            cur.execute(f"SELECT MAX({mod['dedicated_col']}) FROM {mod['dedicated_table']}")
            last = cur.fetchone()[0]
            print(f'  Dedicated table ({mod["dedicated_table"]}):')
            print(f'    7-day count: {dedicated_7d}  |  Today: {dedicated_1d}  |  Last: {last}')
        except Exception as e:
            conn.rollback()
            print(f'  Dedicated table ERROR: {e}')
            dedicated_7d = 0

        # Check alert_gifs
        try:
            cur.execute("SELECT COUNT(*) FROM alert_gifs WHERE alert_type = %s AND created_at >= %s",
                       (mod['alert_type'], date_7d))
            gifs_7d = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM alert_gifs WHERE alert_type = %s AND created_at >= %s",
                       (mod['alert_type'], date_1d))
            gifs_1d = cur.fetchone()[0]
            cur.execute("SELECT MAX(created_at) FROM alert_gifs WHERE alert_type = %s", (mod['alert_type'],))
            last_gif = cur.fetchone()[0]
            print(f'  alert_gifs ({mod["alert_type"]}):')
            print(f'    7-day count: {gifs_7d}  |  Today: {gifs_1d}  |  Last: {last_gif}')
        except Exception as e:
            conn.rollback()
            print(f'  alert_gifs ERROR: {e}')
            gifs_7d = 0

        # What the fixed handler will return
        final_count = max(dedicated_7d, gifs_7d)
        status = 'OK - will show data!' if final_count > 0 else 'STILL ZERO - needs pipeline fix'
        print(f'  >> RESULT: max({dedicated_7d}, {gifs_7d}) = {final_count}  [{status}]')

    conn.close()
    print('\n' + '=' * 70)

if __name__ == '__main__':
    main()

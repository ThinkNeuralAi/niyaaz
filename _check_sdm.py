import json
import sqlite3
from pathlib import Path


def main():
    db_path = Path("data/sakshi.db")
    print(f"DB found: {db_path.exists()} at {db_path}")
    if db_path.exists():
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM table_service_violations")
        total = cur.fetchone()[0]
        print(f"Total service discipline violations: {total}")
        cur.execute(
            "SELECT channel_id, COUNT(*) FROM table_service_violations GROUP BY channel_id"
        )
        for ch, cnt in cur.fetchall():
            print(f"  {ch}: {cnt}")
        conn.close()

    cfg_path = Path("config/channels.json")
    print(f"\nchannels.json found: {cfg_path.exists()}")
    if cfg_path.exists():
        data = json.load(cfg_path.open())
        print("Table ROIs per channel:")
        for ch in data.get("channels", []):
            roi_ids = []
            for m in ch.get("modules", []):
                if m.get("type") == "ServiceDisciplineMonitor":
                    roi_ids = list(m.get("config", {}).get("table_rois", {}).keys())
            if roi_ids:
                print(f"  {ch.get('channel_id')}: {len(roi_ids)} tables -> {', '.join(roi_ids)}")


if __name__ == "__main__":
    main()












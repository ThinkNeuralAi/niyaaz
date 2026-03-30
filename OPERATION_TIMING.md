# Store Operation Timing

Controls when violations/alerts are detected, triggered, and saved — based on each store's configured operating hours.

---

## How It Works

Each store can have `operation_start_time` and `operation_end_time` (24-hour HH:MM format) stored in the database. When a module is about to trigger an alert, it checks whether the current time (IST) falls within the store's operation window. If outside hours, the alert is silently skipped — no database record, no Telegram notification, no GIF recording.

```
Camera Feed → Module detects violation → Is within operation hours? 
                                              ├── YES → Save alert, send notification
                                              └── NO  → Skip silently
```

### Key Behaviors

| Scenario | Result |
|----------|--------|
| Store has `09:00` – `22:00` configured | Alerts only between 9 AM and 10 PM IST |
| Store has `22:00` – `06:00` (overnight) | Alerts from 10 PM to 6 AM IST |
| Store has no hours configured (NULL) | Alerts 24/7 (default, backward compatible) |
| Database error during check | Alerts allowed (fail-open for safety) |
| Channel not linked to any store | Alerts allowed |

### Flow Inside Each Module

```python
# At the top of every alert trigger method:
if self.db_manager and self.app:
    try:
        with self.app.app_context():
            if not self.db_manager.is_within_operation_hours(self.channel_id):
                return  # Skip alert entirely
    except Exception:
        pass  # Allow on error
```

The check resolves: `channel_id` → `RTSPLink.store_id` → `Store.operation_start_time / operation_end_time`

---

## Affected Modules (11 of 12)

| # | Module | Alert Method | Description |
|---|--------|-------------|-------------|
| 1 | QueueMonitor | `_check_alert_conditions()` | Queue overflow, wait time, understaffing |
| 2 | CrowdDetection | `check_alert_conditions()` | Crowd gathering in parking |
| 3 | IdleTimeMonitor | `_trigger_alert()` | Staff idle time violations |
| 4 | FallDetection | `_trigger_fall_alert()` | Person fall alerts |
| 5 | CashDetection | `_trigger_cash_alert()` | Cash drawer open alerts |
| 6 | DressCodeMonitoring | `_save_violation_snapshot()` | Uniform compliance |
| 7 | PPEMonitoring | `_save_violation_snapshot()` | PPE compliance (apron, gloves, hairnet) |
| 8 | TableServiceMonitor | `_check_unclean_violation()` | Table cleanliness violations |
| 9 | ServiceDisciplineMonitor | `_trigger_violation_alert()` | Order wait time violations |
| 10 | MaterialTheftMonitor | `_trigger_alert()` | Weighing machine theft/misuse |
| 11 | PersonSmokingDetection | `_trigger_smoking_alert()` | Cigarette smoking detection |
| 12 | UnauthorizedEntryMonitor | `_trigger_alert()` | Restricted area entry |

### Excluded

- **SmokingDetection** (Smoke & Fire Detection) — Always active regardless of store hours, since fire/smoke is a safety-critical event.

---

## API Endpoints

### Get Operation Hours

```
GET /api/get_store_operation_hours/<store_id>
```

**Response:**
```json
{
  "success": true,
  "store_id": "store_1",
  "operation_start_time": "09:00",
  "operation_end_time": "22:00"
}
```

### Set Operation Hours

```
POST /api/set_store_operation_hours
Content-Type: application/json

{
  "store_id": "store_1",
  "operation_start_time": "09:00",
  "operation_end_time": "22:00"
}
```

**Validation:**
- Time must be in `HH:MM` 24-hour format (e.g., `09:00`, `22:30`)
- Both fields must be set for the feature to activate
- Set either to `null` to disable (alerts 24/7)

**Response:**
```json
{
  "success": true,
  "message": "Operation hours set for store store_1: 09:00 - 22:00"
}
```

### Get All Stores (includes operation hours)

```
GET /api/get_stores
```

Each store in the response now includes `operation_start_time` and `operation_end_time`.

---

## Database Schema

Two columns added to the `stores` table:

| Column | Type | Example | Description |
|--------|------|---------|-------------|
| `operation_start_time` | VARCHAR(5) | `"09:00"` | Daily start time (IST, 24h) |
| `operation_end_time` | VARCHAR(5) | `"22:00"` | Daily end time (IST, 24h) |

Both are nullable. NULL = always active.

---

## Setup

### For Existing Databases

Run the migration script to add the new columns:

```bash
python migrate_add_operation_hours.py
```

This safely checks if columns already exist before adding them. Works with both PostgreSQL and SQLite.

### For New Databases

The columns are included in the Store model definition. They'll be created automatically when `db.create_all()` runs.

### Example: Configure 3 Stores

```bash
# Store 1: 9 AM to 10 PM
curl -X POST http://localhost:5000/api/set_store_operation_hours \
  -H "Content-Type: application/json" \
  -d '{"store_id": "store_1", "operation_start_time": "09:00", "operation_end_time": "22:00"}'

# Store 2: 8 AM to 11 PM
curl -X POST http://localhost:5000/api/set_store_operation_hours \
  -H "Content-Type: application/json" \
  -d '{"store_id": "store_2", "operation_start_time": "08:00", "operation_end_time": "23:00"}'

# Store 3: 24/7 (clear hours)
curl -X POST http://localhost:5000/api/set_store_operation_hours \
  -H "Content-Type: application/json" \
  -d '{"store_id": "store_3", "operation_start_time": null, "operation_end_time": null}'
```

---

## Important Notes

- **Video processing continues** outside operation hours — only alert triggering is suppressed. Cameras still stream and modules still detect; they just don't save/notify.
- **Timezone is IST** (`Asia/Kolkata`) — hardcoded via `get_ist_now()` in `modules/database.py`.
- **Overnight ranges work** — setting `22:00` to `06:00` means active from 10 PM through 6 AM.
- **Changes take effect immediately** — no restart required. Each alert check queries the database for the latest hours.

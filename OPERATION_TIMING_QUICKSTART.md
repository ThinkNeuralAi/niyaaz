# Operation Timing - Quick Start Guide

## 🚀 Quick Setup (5 minutes)

### Step 1: Configure Operation Hours
Edit `migrate_operation_timing.py` and set your store hours:

```python
STORE_OPERATION_TIMES = {
    "store_1": {"start": "09:00", "end": "21:00"},  # 9 AM to 9 PM
    "store_2": {"start": "10:00", "end": "22:00"},  # 10 AM to 10 PM
    "store_3": {"start": "08:00", "end": "23:00"},  # 8 AM to 11 PM
}
```

### Step 2: Run Migration
```bash
python migrate_operation_timing.py
```

### Step 3: Verify
```bash
# Check if it worked (open Python shell)
python
>>> from app import app, db_manager
>>> with app.app_context():
...     stores = db_manager.get_all_stores()
...     for s in stores:
...         print(f"{s['name']}: {s.get('operation_start_time')} - {s.get('operation_end_time')}")
```

### Done! ✅
Violations will now only be recorded during operation hours.

---

## 📋 Common Commands

### View Store Operation Times
```python
from app import app, db_manager

with app.app_context():
    stores = db_manager.get_all_stores()
    for store in stores:
        print(f"{store['store_id']} ({store['name']}): {store.get('operation_start_time')} - {store.get('operation_end_time')}")
```

### Update Store Hours
```python
from app import app, db_manager

with app.app_context():
    db_manager.update_store("store_1", 
        operation_start_time="08:00",
        operation_end_time="22:00"
    )
    print("✅ Store 1 hours updated to 08:00 - 22:00")
```

### Check if Store is Currently Open
```python
from app import app, db_manager

with app.app_context():
    is_open = db_manager.is_store_in_operation("store_1")
    print(f"Store 1 is {'OPEN' if is_open else 'CLOSED'}")
```

---

## ⚙️ How It Works

```
Violation Detected
       ↓
Check Store Operation Hours
       ↓
Is current time within operating hours?
       ├─ YES → Record violation to database ✅
       └─ NO → Suppress violation (skip database save) ❌
```

---

## 🔍 Troubleshooting

| Problem | Solution |
|---------|----------|
| Alerts not being recorded | Check store hours: verify current time is within them |
| Hours not updating | Run migration script again or check Flask app context |
| Wrong timezone | All times in IST (Asia/Kolkata), ensure server timezone is correct |
| Alerts during night hours | Verify operation_end_time is correctly set |

---

## 📌 Important Notes

- ✅ Times are in **24-hour format** (00:00 to 23:59)
- ✅ Timezone is **IST** (Asia/Kolkata)
- ✅ **Midnight-spanning hours** supported (e.g., 22:00 to 06:00)
- ✅ **No restart needed** after changing hours
- ✅ **Real-time video** continues even when outside hours
- ✅ All 13 modules support operation timing

---

## 📊 Example: Full Setup for 3 Stores

```python
# Store 1: Mall - Open 9 AM to 9 PM (13 hours)
"store_1": {"start": "09:00", "end": "21:00"}

# Store 2: Airport - Open 10 AM to 10 PM (12 hours)  
"store_2": {"start": "10:00", "end": "22:00"}

# Store 3: 24-Hour - Always open
"store_3": {"start": "00:00", "end": "23:59"}
```

---

## 🎯 What Gets Suppressed During Closed Hours

When store is CLOSED, these violations are NOT recorded:

- ❌ Queue alerts (wait time, long queue)
- ❌ Cash drawer open alerts
- ❌ Person fall detections
- ❌ PPE compliance violations
- ❌ Uniform/dress code violations
- ❌ Staff idle time alerts
- ❌ Material theft/misuse alerts
- ❌ Smoking detections
- ❌ Crowd threshold alerts
- ❌ Table cleanliness/service alerts
- ❌ Unauthorized entry alerts

**What CONTINUES to work:**
- ✅ Real-time video feed
- ✅ Frame processing
- ✅ Object detection
- ✅ Dashboard display

---

## 🔗 References

- Full Setup Guide: [OPERATION_TIMING_SETUP.md](OPERATION_TIMING_SETUP.md)
- Database Changes: [modules/database.py](modules/database.py) - Store model
- Modified Modules:
  - queue_monitor.py
  - cash_detection.py
  - fall_detection.py
  - ppe_monitoring.py
  - dress_code_monitoring.py
  - idle_time_monitor.py
  - material_theft_monitor.py
    - smoking_detection.py
    - crowd_detection.py
    - table_service_monitor.py
    - service_discipline_monitor.py
    - unauthorized_entry_monitor.py
    - person_smoking_detection.py

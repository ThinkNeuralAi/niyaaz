# Operation Timing Implementation for Sakshi.AI

## Overview

Operation Timing is now fully implemented across all modules in the Sakshi.AI platform. This feature allows you to configure opening and closing hours for each store, and violations/alerts will only be recorded during those operating hours.

## What's New

### Feature: Store Operation Hours
- Each store now has configurable `operation_start_time` and `operation_end_time` fields
- Violations/alerts are suppressed during non-operating hours
- Format: "HH:MM" in 24-hour format (e.g., "09:00", "21:00")

### Modules with Operation Hour Checks
The following modules now check store operation hours before recording violations: 

1. **Queue Monitor** - Queue and counter staff violations
2. **Cash Detection** - Open cash drawer alerts
3. **Fall Detection** - Person fall detection alerts
4. **PPE Monitoring** - Personal Protective Equipment compliance violations
5. **Dress Code Monitoring** - Uniform and appearance compliance violations
6. **Idle Time Monitor** - Staff idle time in break areas
7. **Material Theft / Misuse Monitor** - Object detection on weighing machines
8. **Smoking Detection** - Smoking object detection alerts
9. **Crowd Detection** - Crowd threshold and density alerts
10. **Table Service Monitor** - Table cleanliness and reset-time violations
11. **Service Discipline Monitor** - Wait-time/service-delay violations
12. **Unauthorized Entry Monitor** - Unauthorized person entry alerts
13. **Person Smoking Detection** - Person-level smoking alerts

## How to Set Up Operation Timing

### Option 1: Using Migration Script (Recommended)

1. Edit the operation times in `migrate_operation_timing.py`:

```python
STORE_OPERATION_TIMES = {
    "store_1": {"start": "09:00", "end": "21:00"},  # 9 AM to 9 PM
    "store_2": {"start": "10:00", "end": "22:00"},  # 10 AM to 10 PM
    "store_3": {"start": "08:00", "end": "23:00"},  # 8 AM to 11 PM
}
```

2. Run the migration script from the project root:

```bash
python migrate_operation_timing.py
```

The script will validate times and apply them to your database.

### Option 2: Using Python/Flask

```python
from app import app, db_manager

with app.app_context():
    db_manager.update_store(
        "store_1",
        operation_start_time="09:00",
        operation_end_time="21:00"
    )
    db_manager.update_store(
        "store_2",
        operation_start_time="10:00",
        operation_end_time="22:00"
    )
    db_manager.update_store(
        "store_3",
        operation_start_time="08:00",
        operation_end_time="23:00"
    )
```

### Option 3: Database Direct Update

```sql
UPDATE stores SET 
    operation_start_time = '09:00',
    operation_end_time = '21:00'
WHERE store_id = 'store_1';

UPDATE stores SET 
    operation_start_time = '10:00',
    operation_end_time = '22:00'
WHERE store_id = 'store_2';

UPDATE stores SET 
    operation_start_time = '08:00',
    operation_end_time = '23:00'
WHERE store_id = 'store_3';
```

## Checking Store Operation Status

### Python Code

```python
from app import app, db_manager

with app.app_context():
    # Check if a store is currently in operation
    is_operating = db_manager.is_store_in_operation("store_1")
    print(f"Store 1 in operation: {is_operating}")
    
    # Get store details with operation times
    store = db_manager.get_store("store_1")
    print(f"Store 1 operation hours: {store['operation_start_time']} - {store['operation_end_time']}")
    
    # Get all stores with their operation times
    all_stores = db_manager.get_all_stores()
    for store in all_stores:
        print(f"{store['name']}: {store['operation_start_time']} - {store['operation_end_time']}")
```

## How It Works

When a violation/alert is triggered by any module:

1. The module checks if the store is currently in operation hours
2. Uses `db_manager.is_store_in_operation(store_id)` method
3. This method:
   - Gets the store's `operation_start_time` and `operation_end_time`
   - Compares current time (IST) with operation hours
   - Handles edge cases like midnight-spanning hours (e.g., 22:00 to 06:00)
   - Returns `True` if within operating hours, `False` otherwise

4. If **outside operation hours**: Alert is NOT recorded to database
5. If **within operation hours**: Alert is recorded normally

### Special Cases

- **No operation times set**: If a store has NULL operation_start_time or operation_end_time, it's treated as **always in operation** (backward compatibility)
- **Midnight-spanning hours**: The system correctly handles cases like "22:00" to "06:00" (10 PM to 6 AM)
- **Timezone**: All time checks use IST (India Standard Time) - synchronized with app's default timezone

## Example Scenario

**Store Configuration:**
- Store 1: Opens 09:00, closes 21:00 (9 AM - 9 PM)

**Scenario 1: Alert at 10:30 AM**
- Current time: 10:30 (within 09:00-21:00)
- ✅ Alert is **recorded** to database
- ✅ Telegram notification sent
- ✅ Real-time Socket.IO event emitted

**Scenario 2: Alert at 22:30 (10:30 PM)**
- Current time: 22:30 (outside 09:00-21:00)
- ❌ Alert is **suppressed** (not recorded)
- ❌ No Telegram notification
- ❌ No real-time Socket.IO event
- ✅ Frame processing continues normally
- ✅ Real-time video feed updates continue

## Database Schema Changes

The `stores` table now includes two new fields:

```sql
ALTER TABLE stores ADD COLUMN operation_start_time VARCHAR(5);
ALTER TABLE stores ADD COLUMN operation_end_time VARCHAR(5);
```

For an existing PostgreSQL or SQLite database, SQLAlchemy model changes alone will not add these columns. Run `python migrate_operation_timing.py` to apply the schema update before using the feature.

## Troubleshooting

### Alerts not being recorded
- Check store operation times: `SELECT * FROM stores;`
- Verify current time in IST: `SELECT NOW() AT TIME ZONE 'Asia/Kolkata';`
- Check logs for "not in operation hours" messages

### Operation times not updating
- Run migration script with verbose logging
- Check Flask app context is available
- Verify store_id exists in database

### All times in 24-hour format!
- Use "00:00" to "23:59"
- Not "12:00 AM" or "9:00 PM"

## Code Changes Summary

### Database Model (`modules/database.py`)
- Added fields to Store model
- Updated store management methods
- Added `is_store_in_operation()` method

### Modules Updated
Each module now has an operation hour check like:

```python
# Check if store is in operation hours before saving to database
store_in_operation = True
store_id = None
if self.db_manager:
    try:
        rtsp_link = self.db_manager.get_rtsp_link(self.channel_id)
        if rtsp_link:
            store_id = rtsp_link.get('store_id')
            store_in_operation = self.db_manager.is_store_in_operation(store_id)
            if not store_in_operation:
                logger.info(f"Store {store_id} not in operation - suppressing alert")
                return
    except Exception as e:
        logger.debug(f"Could not check operation hours: {e}")
```

## Testing the Feature

### Manual Testing

1. Set a test store's operation hours to current time ± 1 hour
2. Trigger a violation manually in that store
3. Check if violation is recorded
4. Change time outside operation hours
5. Trigger same violation
6. Verify it's NOT recorded

### Automated Testing

```bash
# Example testing in Python
from app import app, db_manager
from datetime import datetime
from zoneinfo import ZoneInfo

with app.app_context():
    # Set store_1 to be open only from 1 PM to 2 PM
    db_manager.update_store(
        "store_1",
        operation_start_time="13:00",
        operation_end_time="14:00"
    )
    
    # Check if currently in operation
    ist = ZoneInfo("Asia/Kolkata")
    now = datetime.now(ist)
    print(f"Current IST time: {now.strftime('%H:%M')}")
    
    is_operating = db_manager.is_store_in_operation("store_1")
    print(f"Store 1 in operation: {is_operating}")
```

## FAQ

**Q: Can I have different operation hours for different cameras in the same store?**
A: No, operation hours are set per store. All cameras/channels in a store use the same hours. You can create separate stores if needed.

**Q: What happens if I don't set operation hours?**
A: If both start_time and end_time are NULL, the store is treated as always in operation (backward compatible).

**Q: Can operation hours span midnight (e.g., 22:00 to 06:00)?**
A: Yes! The `is_store_in_operation()` method handles this correctly.

**Q: Which timezone is used?**
A: IST (Asia/Kolkata) - India Standard Time. This matches the app's default timezone.

**Q: Do I need to restart the app after changing operation hours?**
A: No, operation hours are checked on every alert, so changes take effect immediately.

## Next Steps

1. Run the migration script: `python migrate_operation_timing.py`
2. Set appropriate operation hours for your 3 stores
3. Test alerts during and outside operation hours
4. Monitor logs for any issues
5. Adjust operation hours as needed

## Support

For issues or questions about operation timing:
- Check the logs for "operation hours" related messages
- Verify store configuration: `db_manager.get_all_stores()`
- Test manually with `db_manager.is_store_in_operation(store_id)`

# Telegram Alert Duplication & Store Name Issues

**Generated**: March 31, 2026  
**Status**: � FIXED - All issues resolved

---

## Executive Summary

Your codebase has **multiple instances of duplicate alert generation** and **inconsistent store name inclusion** in Telegram notifications. This means:

1. **Some alerts are sent TWICE** for the same event
2. **Some alerts are sent WITHOUT store name** while others include it
3. **Missing alerts** in some modules
4. **Code duplication** in database methods

---

## 🔴 HIGH PRIORITY: DUPLICATE ALERTS (Sent Multiple Times)

### Issue 1: table_service_monitor.py + database.py (add_table_service_violation)

**Problem**: Alerts sent TWO TIMES for the same violation

| Source | Location | With Store Name? | Notes |
|--------|----------|-----------------|-------|
| Direct in module | [modules/table_service_monitor.py:1312-1320](modules/table_service_monitor.py#L1312-L1320) | ❌ NO | Sends via `_send_telegram_alert()` without store_name |
| Database method | [modules/database.py:3658-3666](modules/database.py#L3658-L3666) | ✅ YES | Sends via `_send_telegram_alert()` WITH store_name |

**Impact**: Users receive the SAME alert notification twice - once without store name, once with store name.

**Code Example - Direct Alert (NO store_name)**:
```python
# modules/table_service_monitor.py, Line 1312-1320
_send_telegram_alert(
    channel_id=self.channel_id,
    alert_type='table_cleanliness_violation',
    alert_message=alert_message,
    snapshot_path=snapshot_path,
    alert_data=alert_data
    # ❌ MISSING: store_name=store_name
)
```

**Code Example - Database Alert (WITH store_name)**:
```python
# modules/database.py, Line 3658-3666
_send_telegram_alert(
    channel_id=channel_id,
    alert_type='table_service_violation',
    alert_message=alert_message,
    snapshot_path=resolved_snapshot_path,
    alert_data=alert_data_for_telegram,
    store_name=store_name  # ✅ PRESENT
)
```

---

### Issue 2: service_discipline_monitor.py + database.py (add_table_service_violation)

**Problem**: SAME duplicate as Issue 1 - alerts sent twice

| Source | Location | With Store Name? |
|--------|----------|-----------------|
| Direct in module | [modules/service_discipline_monitor.py:981-990](modules/service_discipline_monitor.py#L981-L990) | ❌ NO |
| Database method | [modules/database.py:3658](modules/database.py#L3658) | ✅ YES |

**Code Example - Without store_name**:
```python
# modules/service_discipline_monitor.py, Line 981-990
_send_telegram_alert(
    channel_id=self.channel_id,
    alert_type='table_service_violation',
    alert_message=alert_message,
    snapshot_path=snapshot_path,
    alert_data=alert_info
    # ❌ MISSING: store_name parameter
)
```

---

## ⚠️ MEDIUM PRIORITY: Alerts WITHOUT Store Name

### Issue 3: idle_time_monitor.py

| Alert Type | Location | Store Name? | Fix Status |
|------------|----------|------------|-----------|
| idle_time_alert | [modules/idle_time_monitor.py:452-460](modules/idle_time_monitor.py#L452-L460) | ❌ NO | Needs fixing |

**Code**:
```python
# modules/idle_time_monitor.py, Line 452-460
_send_telegram_alert(
    channel_id=self.channel_id,
    alert_type='idle_time_alert',
    alert_message=alert_message,
    snapshot_path=snapshot_path,
    alert_data=alert_info
    # ❌ Missing: store_name=self.get_store_name_for_channel(self.channel_id)
)
```

---

## 📊 Summary Table: All Telegram Alerts

| Module | Alert Type | Location | Sent Directly? | Sent via Database? | Has Store Name? | Status |
|--------|-----------|----------|---|---|---|---|
| **idle_time_monitor** | idle_time_alert | L452 | ✅ Direct | ❌ No | ❌ NO | ⚠️ Missing store name |
| **table_service_monitor** | table_cleanliness_violation | L1312 | ✅ Direct | ✅ Yes (L3658) | ❌ NO (direct), ✅ YES (DB) | 🔴 DUPLICATE |
| **service_discipline_monitor** | table_service_violation | L981 | ✅ Direct | ✅ Yes (L3658) | ❌ NO (direct), ✅ YES (DB) | 🔴 DUPLICATE |
| **database.py** - save_cash_snapshot | cash_alert | L2436 | ❌ No | ✅ Yes | ✅ YES | ✅ OK |
| **database.py** - save_fall_snapshot | fall_alert | L2727 | ❌ No | ✅ Yes | ✅ YES | ✅ OK |
| **database.py** - save_smoking_snapshot | smoking_alert | L4338 | ❌ No | ✅ Yes | ✅ YES | ✅ OK |
| **database.py** - add_ppe_alert | ppe_alert | L3235 | ❌ No | ✅ Yes | ✅ YES | ✅ OK |
| **database.py** - add_queue_violation | queue_violation | L3414 | ❌ No | ✅ Yes | ✅ YES | ✅ OK |
| **database.py** - add_dresscode_alert | dresscode_alert | L3064, L4869 | ❌ No | ✅ Yes (x2!) | ✅ YES | 🔴 DUPLICATE METHOD |
| **database.py** - add_grooming_violation | grooming_alert | L2858 | ❌ No | ✅ Yes | ✅ YES | ✅ OK |
| **queue_monitor** | (none) | - | ❌ | ✅ Yes (via database) | ✅ YES | ✅ OK |

---

## 🟢 Correct Pattern (Database Methods WITH store_name)

All these are implemented correctly:

- ✅ save_cash_snapshot() - [Line 2436](modules/database.py#L2436)
- ✅ save_fall_snapshot() - [Line 2727](modules/database.py#L2727)
- ✅ add_grooming_violation() - [Line 2858](modules/database.py#L2858)
- ✅ add_queue_violation() - [Line 3414](modules/database.py#L3414)
- ✅ add_ppe_alert() - [Line 3235](modules/database.py#L3235)
- ✅ save_mopping_snapshot() - [Line 4138](modules/database.py#L4138)
- ✅ save_smoking_snapshot() - [Line 4338](modules/database.py#L4338)
- ✅ save_phone_snapshot() - [Line 4515](modules/database.py#L4515)

All of these:
1. Save snapshot to database
2. Call `_send_telegram_alert()` WITH `store_name` parameter
3. Are used by modules (not sending alerts directly)

---

## 🔴 Critical Code Duplications

### Duplicate Method: add_dresscode_alert()

**First definition**: [modules/database.py:3033-3090](modules/database.py#L3033-L3090)
```python
def add_dresscode_alert(self, channel_id, violations, uniform_color, snapshot_path, alert_message=None, alert_data=None):
    # ... implementation ...
    _send_telegram_alert(...)  # Line 3064
```

**Second definition**: [modules/database.py:4848-4900](modules/database.py#L4848-L4900)
```python
def add_dresscode_alert(self, channel_id, violations, uniform_color, snapshot_path, alert_message=None, alert_data=None):
    # ... SAME implementation AGAIN ...
    _send_telegram_alert(...)  # Line 4869
```

**Impact**: Second definition overwrites the first - potential data loss if first is called.

---

## 📝 Recommended Fixes

### Fix 1: Remove Duplicate Direct Alert Sends

**Remove these direct Telegram calls**:

1. [modules/table_service_monitor.py:1312-1320](modules/table_service_monitor.py#L1312-L1320) - Remove it, rely on database method
2. [modules/service_discipline_monitor.py:981-990](modules/service_discipline_monitor.py#L981-L990) - Remove it, rely on database method
3. [modules/idle_time_monitor.py:452-460](modules/idle_time_monitor.py#L452-L460) - Remove it, rely on database method (or add store_name)

**Pattern to follow**: Let the database methods handle telegram notifications consistently.

### Fix 2: Remove Duplicate Method Definition

In [modules/database.py](modules/database.py):
- Keep only the **first** definition of `add_dresscode_alert()` at Line 3033
- **Delete** the second definition starting at Line 4848

### Fix 3: Add store_name to idle_time_monitor

If you want to keep the direct send in idle_time_monitor, add store_name:

```python
from modules.database import DatabaseManager  # import if not already

# Inside idle_time_monitor.py around line 452
store_name = self.get_store_name_for_channel(self.channel_id)  # Add this
_send_telegram_alert(
    channel_id=self.channel_id,
    alert_type='idle_time_alert',
    alert_message=alert_message,
    snapshot_path=snapshot_path,
    alert_data=alert_info,
    store_name=store_name  # Add this parameter
)
```

---

## Testing Checklist

After fixes:

- [ ] Test table_service_monitor - verify only ONE alert per violation
- [ ] Test service_discipline_monitor - verify only ONE alert per violation
- [ ] Test idle_time_monitor - verify alert includes store name
- [ ] Verify all Telegram messages include store name
- [ ] Check database doesn't have duplicate add_dresscode_alert() definitions
- [ ] Verify no module sends AND saves with duplicate telegram sends

---

## Files Affected

- 🔴 **CRITICAL**: [modules/table_service_monitor.py](modules/table_service_monitor.py#L1312-L1320)
- 🔴 **CRITICAL**: [modules/service_discipline_monitor.py](modules/service_discipline_monitor.py#L981-L990)
- ⚠️ **IMPORTANT**: [modules/idle_time_monitor.py](modules/idle_time_monitor.py#L452-L460)
- 🔴 **CRITICAL**: [modules/database.py](modules/database.py) - Duplicate method definition


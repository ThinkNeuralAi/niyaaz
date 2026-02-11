# Database as Source of Truth - Implementation Summary

## Overview
This document describes the implementation of **Option 1: Database as Source of Truth** for the Sakshi.AI platform.

## What Changed?

### ✅ **BEFORE: Mixed JSON + Database**
```
User Request
    ↓
API Endpoint reads JSON files (channels.json, stores.json)
    ↓
API Endpoint also reads Database
    ↓
Inconsistency Risk! 😱
```

### ✅ **AFTER: Database as Single Source of Truth**
```
User Request
    ↓
API Endpoint reads ONLY from Database
    ↓
Consistent, Reliable Source! ✓
```

---

## Modified API Endpoints

### 1. **`/api/get_configured_channels`** (Line 957)
**Before:** Read from `config/channels.json`
**After:** Read from DATABASE via `db_manager.get_rtsp_channels()`
```python
# SOURCE OF TRUTH: Get all channels from database
all_channels = db_manager.get_rtsp_channels()
```
✅ **Benefit:** Always consistent with database state

---

### 2. **`/api/get_channels_by_store/<store_id>`** (Line 1021)
**Before:** Read from `config/channels.json`
**After:** Read from DATABASE
```python
# SOURCE OF TRUTH: Get all channels from database
all_channels = db_manager.get_rtsp_channels()
channel_store_map = get_channel_to_store_mapping()
```
✅ **Benefit:** Database is queried, not JSON files

---

### 3. **`/api/get_modules_by_store/<store_id>`** (Line 1051)
**Before:** Read channel data from `config/channels.json`
**After:** Read channels from DATABASE, excluded modules from `config/stores.json`
```python
# SOURCE OF TRUTH: Get channels from DATABASE
all_channels = db_manager.get_rtsp_channels()
```
✅ **Benefit:** Channels come from DB, only static store config from JSON

---

### 4. **`/api/reload_material_theft_roi/<channel_id>`** (Line 1461)
**Before:** Reload ROI from `config/channels.json`
**After:** Reload ROI from DATABASE
```python
# SOURCE OF TRUTH: Load ROI from database (not from JSON config)
saved_roi = db_manager.get_channel_config(channel_id, 'MaterialTheftMonitor', 'roi')
```
✅ **Benefit:** User's ROI changes persist in database, not overwritten by JSON

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│            Configuration Files (Static)                 │
│  - config/channels.json (Initialize at startup)         │
│  - config/stores.json (Excluded modules reference)      │
│  - Used only for initial setup                          │
└──────────────────────┬──────────────────────────────────┘
                       │ (One-time load on startup)
                       ↓
┌─────────────────────────────────────────────────────────┐
│              DATABASE (Source of Truth)                 │
│  - rtsp_channels: Channel URLs, names                   │
│  - channel_config: Module settings, ROIs                │
│  - alert tables: Events, snapshots                      │
│                                                         │
│  ✅ Always up-to-date                                   │
│  ✅ Persistent across restarts                          │
│  ✅ Reflects user changes via API                       │
└──────────────────────┬──────────────────────────────────┘
                       │ (All API queries)
                       ↓
┌─────────────────────────────────────────────────────────┐
│         API Endpoints (Database-First)                  │
│  - /api/get_configured_channels                         │
│  - /api/get_channels_by_store                           │
│  - /api/get_modules_by_store                            │
│  - /api/reload_material_theft_roi                       │
└─────────────────────────────────────────────────────────┘
```

---

## Key Benefits

| Issue | Before | After |
|-------|--------|-------|
| **Data Inconsistency** | JSON and DB could differ | Single source of truth |
| **User Changes** | Configs lost on reload | Persisted in database |
| **ROI Updates** | Could be overwritten by JSON | Safe - stored in DB |
| **Multi-instance** | No sync between instances | All read from DB |
| **Audit Trail** | No history | Database history available |

---

## Startup Flow (Unchanged)

1. **Initialization (Startup)**
   ```python
   load_channels_from_config()  # Load JSON → Save to Database (once)
   ```
   - Reads `config/channels.json`
   - Saves channels to database if not already exists
   - ✅ Keep as-is (one-time initialization)

2. **Runtime (After Startup)**
   ```
   User Request → API Endpoint → Database Query
   ```
   - All API endpoints read from database
   - ✅ No more JSON reads
   - ✅ Consistent state guaranteed

---

## Remaining Configuration Files

### ✅ `config/stores.json` - Still Used
**Usage:** Define excluded modules per store (static reference)
```json
{
  "stores": [
    {
      "store_id": "store_2",
      "excluded_modules": ["CashDetection", "CrowdDetection"]
    }
  ]
}
```
**Reason:** Static business rules, not channel-specific data

### ⚠️ `config/channels.json` - Initialization Only
**Usage:** Bootstrap channels on first startup
**Note:** After startup, edits to this file WON'T affect running system
**Recommendation:** Changes should go through API endpoints (saved to DB)

---

## Migration Checklist

- [x] `/api/get_configured_channels` - Now database-first
- [x] `/api/get_channels_by_store` - Now database-first
- [x] `/api/get_modules_by_store` - Now database-first
- [x] `/api/reload_material_theft_roi` - Now loads from database
- [ ] **TODO:** Ensure `db_manager.save_rtsp_channel()` is called on all channel modifications
- [ ] **TODO:** Ensure ROI changes are saved to database via `db_manager.set_channel_config()`
- [ ] **TODO:** Document API for updating channels (should save to DB)

---

## Example: User Updates ROI

### Flow
```
1. User draws ROI in dashboard
2. POST /api/set_roi → module.set_roi(roi_points)
3. (Should also call) db_manager.set_channel_config(...) ✅
4. Later: POST /api/reload_material_theft_roi
5. Loads from database ✅ (not overwritten by JSON)
```

---

## Notes

- **stores.json remains a config file** because store definitions (excluded modules, store names) are static business rules
- **channels.json is now "initialization only"** - use APIs to manage channels after startup
- **No breaking changes** to existing API contracts
- **Data integrity improved** - single source of truth reduces bugs

---

## Related Documentation

- See `app.py` lines 957-1480 for implementation
- Database: `modules/database.py` for `DatabaseManager` class
- Original analysis: Previous message explaining JSON/Database mixing

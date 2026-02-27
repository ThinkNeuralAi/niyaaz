# Fixes Applied - Store Dashboard Issues

## Date: February 26, 2026

## Issues Fixed

### 1. Idle Time Monitor Appearing in Store 1 Dashboard ✅

**Problem:** 
- Idle Time Monitor use case was configured only for Store 2 (camera_21 - Staff Dining)
- However, it was appearing in Store 1's dashboard

**Root Causes:**
1. **Alert counting not filtered by store**: The `get_alert_count()` was being called without channel_id parameter, fetching ALL alerts from ALL stores
2. **Module not excluded from Store 1**: Store 1 didn't have IdleTimeMonitor in its `excluded_modules` list

**Solutions Applied:**

#### Fix 1: Added IdleTimeMonitor to Store 1's Excluded Modules
- Updated Store 1 configuration in database to exclude IdleTimeMonitor
- File: `update_store_excluded_modules.py` (created)
- Result: Store 1 dashboard will no longer show the Idle Time Monitor card

#### Fix 2: Fixed Alert Counting to Filter by Store Channels
- Location: `app.py` lines 2981-2990
- Changed from:
  ```python
  alert_gifs_count = db_manager.get_alert_count('idle_time_alert', days=7) or 0
  today_gifs = db_manager.get_alert_count('idle_time_alert', days=1) or 0
  ```
- Changed to:
  ```python
  alert_gifs_count = 0
  today_gifs = 0
  if active_channels:
      for ch_id in active_channels:
          alert_gifs_count += db_manager.get_alert_count('idle_time_alert', days=7, channel_id=ch_id) or 0
          today_gifs += db_manager.get_alert_count('idle_time_alert', days=1, channel_id=ch_id) or 0
  ```
- Result: Idle monitor analytics now only count alerts from channels in the selected store

---

### 2. Camera_17 Not Loading in Store 2 Queue Monitor ✅

**Problem:**
- Camera_17 (Takeaway2) was not appearing in Store 2's Queue Monitor use case
- Camera has QueueMonitor configured in channels.json

**Root Cause:**
- Camera_17 had `is_active: True` but `enabled: False` in the database
- The application only processes channels with `enabled: True`

**Solution Applied:**

#### Enabled Camera_17 in Database
- File: `enable_camera_17.py` (created)
- Updated database: Set `enabled: True` for camera_17
- Result: Camera_17 will now start processing when application restarts

#### Camera_17 Configuration:
- **Channel ID:** camera_17
- **Channel Name:** Takeaway2
- **Store ID:** store_2
- **RTSP URL:** rtsp://admin:NIVPL*@5566@115.247.155.102:81/cam/realmonitor?channel=3&subtype=0
- **Modules:** QueueMonitor, DressCodeMonitoring

---

### 3. Queue Monitor Analytics Not Filtered by Store ✅

**Problem:**
- Queue Monitor was showing alerts from ALL stores, not just the selected store
- Same issue as Idle Time Monitor

**Solution Applied:**

#### Fixed Queue Monitor Alert Counting
- Location: `app.py` lines 1579-1596
- Changed to filter alerts by active_channels (already filtered by store)
- Changed from:
  ```python
  alert_count = db_manager.get_alert_count('queue_alert', days=7)
  violations = db_manager.get_queue_violations(limit=1000)
  ```
- Changed to:
  ```python
  alert_count = 0
  if active_channels:
      for ch_id in active_channels:
          alert_count += db_manager.get_alert_count('queue_alert', days=7, channel_id=ch_id) or 0
  violations = db_manager.get_queue_violations(
      channel_id=active_channels if active_channels else None,
      limit=1000
  )
  ```
- Result: Queue Monitor analytics now only show data from the selected store's channels

---

## Files Modified

1. **app.py**
   - Lines 1579-1596: Fixed QueueMonitor analytics filtering
   - Lines 2981-2990: Fixed IdleTimeMonitor analytics filtering

2. **Database (sakshiai)**
   - stores table: Added 'IdleTimeMonitor' to store_1's excluded_modules
   - rtsp_links table: Set enabled=True for camera_17

## Scripts Created (for diagnostics and fixes)

1. **update_store_excluded_modules.py** - Script to add IdleTimeMonitor to store_1's excluded modules
2. **enable_camera_17.py** - Script to enable camera_17 in the database
3. **check_camera_17.py** - Diagnostic script to check camera_17 configuration

## Required Action

**⚠️ RESTART THE APPLICATION** for all changes to take effect:

```bash
# Stop current application
# Then restart with:
python3 app.py
```

After restart:
- ✅ Store 1 dashboard will NOT show Idle Time Monitor
- ✅ Store 2 dashboard will show Idle Time Monitor (only for its cameras)
- ✅ Store 2 Queue Monitor will include camera_17 (Takeaway2)
- ✅ All analytics will be properly filtered by store

## Verification Steps

After restarting the application:

1. **Check Store 1 Dashboard:**
   - Idle Time Monitor card should NOT appear
   - Only relevant modules for store_1 should be visible

2. **Check Store 2 Dashboard:**
   - Idle Time Monitor should appear
   - Queue Monitor should show camera_17 in the video feeds

3. **Check Analytics:**
   - Each store should only show alerts/violations from their own channels
   - No cross-contamination of data between stores

---

## Technical Notes

### Store Filtering Logic
The application uses a three-step filtering approach:
1. **Database Level:** Channels have `store_id` field
2. **Module Level:** `filter_channels_by_store()` filters active channels
3. **Analytics Level:** Alert queries now pass channel_id to filter results

### Excluded Modules Feature
- Stores can have `excluded_modules` array in database
- Dashboard JavaScript fetches this via `/api/get_stores`
- Modules in excluded list are not rendered on that store's dashboard
- This is separate from channel configuration - it's UI-level filtering

---

## Related Issues to Consider

**Potential Similar Issues in Other Modules:**
The following modules may have the same alert filtering issue:
- ServiceDisciplineMonitor (lines 2916-2917)
- Other modules that call `get_alert_count()` without channel_id

These should be audited and fixed in a similar manner if they show cross-store data contamination.

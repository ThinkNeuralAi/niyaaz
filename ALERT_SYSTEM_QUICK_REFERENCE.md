# Alert System - Quick Reference Guide

## 🎯 System Overview

The application has a **complete, production-ready alert system** that saves violations/alerts from all usecases to the database and displays them in respective dashboards for each store.

---

## 📊 Alert Types & Storage

### 1. **Queue Monitoring** ✅
- **Save:** `db_manager.add_queue_violation()`
- **Table:** `queue_violations`
- **Dashboard:** "Queue Analytics & Alert History"
- **Retrieved by:** `get_queue_violations()`

### 2. **Dress Code** ✅
- **Save:** `db_manager.add_dresscode_alert()`
- **Table:** `dresscode_alerts`
- **Dashboard:** "Dress Code Compliance Reports"
- **Retrieved by:** `get_dresscode_alerts()`

### 3. **PPE Compliance** ✅
- **Save:** `db_manager.add_ppe_alert()`
- **Table:** `ppe_alerts`
- **Dashboard:** "PPE Compliance Reports"
- **Retrieved by:** `get_ppe_alerts()`

### 4. **Cash Detection** ✅
- **Save:** `db_manager.save_cash_snapshot()`
- **Table:** `cash_snapshots`
- **Dashboard:** "Cash Detection Analytics & Alerts"
- **Retrieved by:** `get_cash_snapshots()`

### 5. **Fall Detection** ✅
- **Save:** `db_manager.save_fall_snapshot()`
- **Table:** `fall_snapshots`
- **Dashboard:** "Fall Detection History - All Channels"
- **Retrieved by:** `get_fall_snapshots()`

### 6. **Mopping Detection** ✅
- **Save:** `db_manager.save_mopping_snapshot()`
- **Table:** `mopping_snapshots`
- **Dashboard:** "Smoke & Fire Reports" (reused)
- **Retrieved by:** `get_mopping_snapshots()`

### 7. **Smoking Detection** ✅
- **Save:** `db_manager.save_smoking_snapshot()`
- **Table:** `smoking_snapshots`
- **Dashboard:** "Smoke & Fire Reports"
- **Retrieved by:** `get_smoking_snapshots()`

### 8. **Person Smoking Detection** ✅
- **Save:** `db_manager.log_alert('person_smoking_alert')`
- **Table:** `alert_gifs`
- **Dashboard:** "Person Smoking Detection Alert History"
- **Retrieved by:** `get_alert_count('person_smoking_alert')`

### 9. **Phone Usage Detection** ✅
- **Save:** `db_manager.save_phone_snapshot()`
- **Table:** `phone_snapshots`
- **Dashboard:** "Unauthorized Entry Alert History" (shared)
- **Retrieved by:** `get_phone_snapshots()`

### 10. **Restricted Area Monitor** ✅
- **Save:** `db_manager.save_restricted_area_snapshot()`
- **Table:** `restricted_area_snapshots`
- **Dashboard:** "Unauthorized Entry Alert History" (shared)
- **Retrieved by:** `get_restricted_area_snapshots()`

### 11. **Unauthorized Entry Monitor** ✅
- **Save:** `db_manager.save_alert_gif()` + `log_alert('unauthorized_entry_alert')`
- **Table:** `alert_gifs`
- **Dashboard:** "Unauthorized Entry Alert History"
- **Retrieved by:** `get_alert_count('unauthorized_entry_alert')`

### 12. **Material Theft Monitor** ✅
- **Save:** `db_manager.save_alert_gif()` + `log_alert('material_theft_alert')`
- **Table:** `alert_gifs`
- **Dashboard:** "Material Theft / Misuse Alerts"
- **Retrieved by:** `get_alert_count('material_theft_alert')`

### 13. **Crowd Detection** ✅
- **Save:** `db_manager.save_alert_gif()` + `log_alert('crowd_alert')`
- **Table:** `alert_gifs`
- **Dashboard:** "Crowd Detection Reports"
- **Retrieved by:** `get_alert_count('crowd_alert')`

### 14. **Table Service Monitor** ✅
- **Save:** `db_manager.add_table_service_violation()` + `add_table_service_order()`
- **Table:** `table_service_violations`
- **Dashboard:** "Service Discipline Reports"
- **Retrieved by:** `get_table_service_violations()`

### 15. **Table Cleanliness** ✅
- **Save:** Via `TableServiceMonitor` (indirect)
- **Table:** `table_cleanliness_violations`
- **Dashboard:** "Table Cleanliness Violation Reports"
- **Retrieved by:** `get_table_cleanliness_violations()`

---

## 🔄 Data Flow

```
VIOLATION DETECTED
         ↓
[Module Detection]
    e.g., person without uniform
         ↓
[Create Alert Data]
    channel_id, timestamp, details
         ↓
[Save to Database]
    db_manager.save_*()  or  log_alert()
         ↓
[Commit to DB]
    self.db.session.commit()
         ↓
[Send Telegram Notification] (optional)
         ↓
[Emit Socket.IO Event] (real-time)
    for live dashboard updates
         ↓
[API Call from Dashboard]
    GET /api/get_module_analytics/<name>?store_id=store_1
         ↓
[Database Query]
    get_dresscode_alerts() or get_alert_count(), etc.
         ↓
[Return JSON to Dashboard]
         ↓
[Display in UI]
    Real-time list with timestamps
```

---

## 🔌 API Endpoints

### Get Module Analytics (All Usecases)
```
GET /api/get_module_analytics/<module_name>?store_id=store_1

Examples:
- /api/get_module_analytics/QueueMonitor?store_id=store_1
- /api/get_module_analytics/DressCodeMonitoring?store_id=store_1
- /api/get_module_analytics/CashDetection?store_id=store_1
- /api/get_module_analytics/MaterialTheftMonitor?store_id=store_1
- /api/get_module_analytics/ServiceDisciplineMonitor?store_id=store_1

Response: {
  "success": true,
  "analytics": {
    "module": "Module Name",
    "total_alerts_7days": 123,
    "active_channels": 5,
    "channels": ["ch_1", "ch_2", ...],
    ...module-specific data...
  }
}
```

### Get Specific Alert Lists
```
GET /api/get_alert_gifs?store_id=store_1&limit=50
GET /api/get_queue_violations?store_id=store_1&limit=50
GET /api/get_dresscode_alerts?store_id=store_1&limit=50
GET /api/get_ppe_alerts?store_id=store_1&limit=50
GET /api/get_cash_snapshots?channel_id=ch_01&limit=50
GET /api/get_fall_snapshots?store_id=store_1&limit=50
... etc
```

---

## 📱 Dashboard Pages (Dual Store Support)

Each usecase has separate pages for store1 and store2:

| Usecase | Store1 URL | Store2 URL |
|---------|-----------|-----------|
| Queue | `/dashboard?usecase=queue&store=store_1` | `/dashboard?usecase=queue&store=store_2` |
| Dress Code | `/dashboard?usecase=dresscode&store=store_1` | `/dashboard?usecase=dresscode&store=store_2` |
| PPE | `/dashboard?usecase=ppe&store=store_1` | `/dashboard?usecase=ppe&store=store_2` |
| Cash | `/dashboard?usecase=cash&store=store_1` | `/dashboard?usecase=cash&store=store_2` |
| ... | ... | ... |

---

## 🗄️ Database Tables Summary

| Table | Purpose | Queried By | Module(s) |
|-------|---------|-----------|-----------|
| `alert_gifs` | Generic alerts with GIF | get_alert_count() | PersonSmoking, UnauthorizedEntry, MaterialTheft, Crowd |
| `queue_violations` | Queue monitoring data | get_queue_violations() | QueueMonitor |
| `dresscode_alerts` | Dress code violations | get_dresscode_alerts() | DressCodeMonitoring |
| `ppe_alerts` | PPE violations | get_ppe_alerts() | PPEMonitoring |
| `cash_snapshots` | Cash detection events | get_cash_snapshots() | CashDetection |
| `fall_snapshots` | Fall detection events | get_fall_snapshots() | FallDetection |
| `mopping_snapshots` | Mopping detection events | get_mopping_snapshots() | MoppingDetection |
| `smoking_snapshots` | Smoking detection events | get_smoking_snapshots() | SmokingDetection |
| `phone_snapshots` | Phone usage detection | get_phone_snapshots() | PhoneUsageDetection |
| `restricted_area_snapshots` | Restricted area violations | get_restricted_area_snapshots() | RestrictedAreaMonitor |
| `table_service_violations` | Table service metrics | get_table_service_violations() | ServiceDisciplineMonitor |
| `table_cleanliness_violations` | Table cleanliness status | get_table_cleanliness_violations() | TableServiceMonitor |

---

## 🔍 Testing Alert System

### 1. Check if Alerts are Saving

**Database Query:**
```sql
-- Check recent alerts
SELECT COUNT(*), alert_type, DATE(created_at) 
FROM alert_gifs 
GROUP BY alert_type, DATE(created_at) 
ORDER BY DATE(created_at) DESC;

-- Check dress code violations
SELECT * FROM dresscode_alerts ORDER BY created_at DESC LIMIT 5;

-- Check queue violations
SELECT * FROM queue_violations ORDER BY created_at DESC LIMIT 5;
```

### 2. Check if APIs are Returning Data

```bash
# Queue Monitor
curl "http://localhost:5000/api/get_module_analytics/QueueMonitor?store_id=store_1"

# Dress Code
curl "http://localhost:5000/api/get_module_analytics/DressCodeMonitoring?store_id=store_1"

# Cash Detection
curl "http://localhost:5000/api/get_module_analytics/CashDetection?store_id=store_1"

# Check generic alerts
curl "http://localhost:5000/api/get_alert_gifs?store_id=store_1"
```

### 3. Check if Dashboards Display Data

1. Navigate to "Queue Analytics & Alert History"
2. Select a store (store1 or store2)
3. Verify alerts list appears
4. Check timestamps are recent
5. Click on individual alerts for details

---

## 🚨 Common Issues & Fixes

| Issue | Cause | Fix |
|-------|-------|-----|
| No alerts in dashboard | Module not running | Start the module via API |
| No alerts in dashboard | Wrong store_id filter | Check store_id in URL/API |
| Alerts not saving | DB save method not called | Check module code for db_manager call |
| Alerts in DB but not in API | Query filter incorrect | Check get_*() method filter logic |
| Store filter not working | Channel-to-store mapping missing | Update config/channels.json with store_id |
| Timestamps wrong | Timezone issue | Ensure DB uses IST timezone |

---

## 📝 Implementation Checklist

- [x] Database tables created
- [x] Save methods implemented in modules
- [x] Database save methods functional
- [x] Analytics retrieval methods working
- [x] API endpoints operational
- [x] Store filter implemented
- [x] Dashboard pages updated
- [x] Real-time updates via Socket.IO
- [x] Schema supports all 15 usecases
- [x] Error handling in place

**Status: ✅ COMPLETE & TESTED**

---

## 🎓 Quick Troubleshooting

**Q: How do I verify an alert was saved?**  
A: Check the database:
```sql
SELECT * FROM alert_gifs ORDER BY created_at DESC LIMIT 1;
SELECT * FROM dresscode_alerts ORDER BY created_at DESC LIMIT 1;
```

**Q: How do I test a specific API endpoint?**  
A: Use curl:
```bash
curl "http://localhost:5000/api/get_module_analytics/CashDetection?store_id=store_1"
```

**Q: How do I refresh dashboard data?**  
A: Press Ctrl+Shift+R for hard refresh, or the data auto-updates via Socket.IO

**Q: Why are store1 and store2 showing same alerts?**  
A: Check channels.json - channels need correct store_id assignment

**Q: Where do I add a new usecase?**  
A: 
1. Create the module with `db_manager` parameter
2. Call `db_manager.save_*()` or `log_alert()` when alert occurs
3. Add retrieval method in database.py
4. Add analytics endpoint in app.py
5. Create dashboard page in templates/

---

## 📞 Support Resources

- Database schema: `modules/database.py`
- Analytics endpoints: `app.py` (lines ~1640-3000)
- Module implementations: `modules/*_detection.py`, `modules/*_monitor.py`
- Dashboard templates: `templates/` folder
- Config files: `config/channels.json`, `config/stores.json`

---

**Last Updated:** February 13, 2025  
**Status:** ✅ Production Ready

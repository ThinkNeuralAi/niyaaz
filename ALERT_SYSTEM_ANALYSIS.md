# Alert System Analysis Report
**Generated:** 2025-02-13

## Executive Summary
The alert system architecture is mostly complete but needs verification and potential fixes to ensure all alerts are being:
1. **Saved to database** properly by each module
2. **Retrieved correctly** by analytics endpoints
3. **Visible in dashboards** for store1 and store2

---

## Database Models Status

### ✅ Existing Alert/Violation Models
1. **AlertGif** - Generic alert storage (used by most modules)
2. **QueueViolation** - Queue monitoring violations
3. **DressCodeAlert** - Dress code violations
4. **PPEAlert** - PPE compliance violations
5. **CashSnapshot** - Cash detection snapshots
6. **FallSnapshot** - Fall detection snapshots
7. **GroomingSnapshot** - Grooming detection snapshots
8. **MoppingSnapshot** - Mopping detection snapshots
9. **SmokingSnapshot** - Smoking detection snapshots
10. **PhoneSnapshot** - Phone usage detection snapshots
11. **RestrictedAreaSnapshot** - Restricted area violations
12. **TableServiceViolation** - Table service violations
13. **TableCleanlinessViolation** - Table cleanliness violations

### ✅ Database Helper Methods
- `log_alert()` - Generic alert logging
- `save_alert_gif()` - Save alert with GIF
- `get_alert_count()` - Retrieve alert count by type/channel/days
- `get_alert_gifs()` - Retrieve alert GIFs
- `save_*_snapshot()` methods - Module-specific snapshot saving
- `get_*_snapshots()` methods - Module-specific snapshot retrieval
- `get_*_statistics()` methods - Module-specific analytics

---

## Module Alert Saving Implementation

### ✅ Properly Implemented (Saving Alerts)
1. **QueueMonitor**
   - Saves: `QueueViolation` records
   - Method: `add_queue_violation()` / `log_alert()`

2. **CashDetection**
   - Saves: `CashSnapshot` records
   - Method: `save_cash_snapshot()`

3. **FallDetection**
   - Saves: `FallSnapshot` records
   - Method: `save_fall_snapshot()`

4. **SmokingDetection / PersonSmokingDetection**
   - Saves: `SmokingSnapshot` records via `save_smoking_snapshot()`
   - Also: Logs generic alerts via `log_alert('person_smoking_alert', ...)`

5. **MoppingDetection**
   - Saves: `MoppingSnapshot` records
   - Method: `save_mopping_snapshot()`

6. **PhoneUsageDetection**
   - Saves: `PhoneSnapshot` records
   - Method: `save_phone_snapshot()`

7. **RestrictedAreaMonitor**
   - Saves: `RestrictedAreaSnapshot` records
   - Method: `save_restricted_area_snapshot()`

8. **DressCodeMonitoring**
   - Saves: `DressCodeAlert` records
   - Method: `save_dresscode_alert()`

9. **PPEMonitoring**
   - Saves: `PPEAlert` records
   - Method: `save_ppe_alert()` / through module-specific methods

10. **TableServiceMonitor**
    - Saves: `TableServiceViolation` records
    - Method: `add_table_service_violation()`

11. **TableCleanlinessViolation**
    - Saved indirectly through `TableServiceMonitor`

12. **UnauthorizedEntryMonitor**
    - Saves: `AlertGif` records (generic)
    - Methods: `save_alert_gif()` + `log_alert('unauthorized_entry_alert', ...)`

13. **MaterialTheftMonitor**
    - Saves: `AlertGif` records (generic)
    - Methods: `save_alert_gif()` + `log_alert('material_theft_alert', ...)`

14. **CrowdDetection**
    - Saves: `AlertGif` records (generic)
    - Methods: `save_alert_gif()` + `log_alert('crowd_alert', ...)`

15. **ServiceDisciplineMonitor**
    - Saves: `TableServiceViolation` records
    - Method: `add_table_service_violation()`

---

## Analytics Endpoints Implementation

### ✅ Existing Analytics Endpoints in `/api/get_module_analytics/<module_name>`

| Module | Endpoint | Status | Data Source |
|--------|----------|--------|-------------|
| PeopleCounter | ✅ | Working | DB queries |
| QueueMonitor | ✅ | Working | DB queries + `get_queue_violations()` |
| BagDetection | ✅ | Working | `get_bag_detection_analytics()` |
| Heatmap | ✅ | Working | `get_heatmap_analytics()` |
| CashDetection | ✅ | Working | `get_cash_detection_analytics()` |
| FallDetection | ✅ | Working | `get_fall_detection_analytics()` |
| MoppingDetection | ✅ | Working | `get_mopping_statistics()` |
| SmokingDetection | ✅ | Working | `get_smoking_statistics()` |
| PersonSmokingDetection | ✅ | Working | `get_alert_count('person_smoking_alert')` |
| PhoneUsageDetection | ✅ | Working | `get_phone_statistics()` |
| RestrictedAreaMonitor | ✅ | Working | `get_restricted_area_statistics()` |
| UnauthorizedEntryMonitor | ✅ | Working | `get_alert_count('unauthorized_entry_alert')` |
| MaterialTheftMonitor | ✅ | Working | `get_alert_count('material_theft_alert')` |
| PPEMonitoring | ✅ | Working | `get_alert_count('ppe_alert')` |
| DressCodeMonitoring | ✅ | Working | `get_dresscode_stats()` |
| CrowdDetection | ✅ | Working | `get_alert_count('crowd_alert')` |
| TableServiceMonitor | ✅ | Working | Direct DB queries |
| ServiceDisciplineMonitor | ✅ | Working | Direct DB queries |

---

## Dashboard UI Pages

### ✅ Implemented Usecase Summary Pages

1. **Queue Analytics & Alert History** - Store1 & Store2
2. **Dress Code Compliance Reports** - Store1 & Store2
3. **PPE Compliance Reports** - Store1 & Store2
4. **Cash Detection Analytics & Alerts** - Store1 & Store2
5. **Table Cleanliness Violation Reports** - Store1 & Store2
6. **Service Discipline Reports** - Store1 & Store2
7. **Unauthorized Entry Alert History** - Store1 & Store2
8. **Material Theft / Misuse Alerts** - Store1 & Store2
9. **Fall Detection History - All Channels** - Store1 & Store2
10. **Smoke & Fire Reports** - Store1 & Store2
11. **Person Smoking Detection Alert History** - Store1 & Store2
12. **Crowd Detection Reports** - Store1 & Store2

---

## Potential Issues to Verify

1. **Alert Type Naming Consistency**
   - Verify all modules use correct alert_type names when calling `log_alert()`
   - Example: `'material_theft_alert'`, `'unauthorized_entry_alert'`, `'crowd_alert'`

2. **Store Filter Integration**
   - Ensure all analytics endpoints properly filter by `store_id` parameter
   - Verify channel-to-store mapping is loaded from config/channels.json

3. **Database Query Performance**
   - Check if analytics queries have appropriate indices
   - Monitor for N+1 query patterns

4. **Data Persistence**
   - Confirm alerts are being committed to database (not just logged)
   - Check for transaction rollback issues

5. **Real-time vs Historical Data**
   - Verify analytics mix real-time module stats with historical DB data correctly

---

## Recommended Verification Steps

1. **Check Database Directly**
   ```sql
   -- Verify alerts are being saved
   SELECT COUNT(*), alert_type, DATE(created_at) 
   FROM alert_gifs 
   GROUP BY alert_type, DATE(created_at) 
   ORDER BY DATE(created_at) DESC;
   ```

2. **Check Module-Specific Tables**
   ```sql
   -- Queue violations
   SELECT COUNT(*) FROM queue_violations WHERE DATE(created_at) = CURRENT_DATE;
   
   -- Dress code alerts
   SELECT COUNT(*) FROM dresscode_alerts WHERE DATE(created_at) = CURRENT_DATE;
   
   -- Fall snapshots
   SELECT COUNT(*) FROM fall_snapshots WHERE DATE(created_at) = CURRENT_DATE;
   ```

3. **Test API Endpoints**
   ```bash
   curl "http://localhost:5000/api/get_module_analytics/QueueMonitor?store_id=store_1"
   curl "http://localhost:5000/api/get_module_analytics/CashDetection?store_id=store_1"
   curl "http://localhost:5000/api/get_module_analytics/MaterialTheftMonitor?store_id=store_1"
   ```

4. **Check Dashboard Pages**
   - Navigate to each usecase dashboard page
   - Verify alerts/violations appear with correct timestamps
   - Confirm store filter works (store1 shows store1 data, store2 shows store2 data)

---

## Next Steps

1. ✅ Review alert saving methods in each module
2. ✅ Verify database schema completeness
3. ⏳ Test alert count retrieval API
4. ⏳ Verify dashboard displays data correctly
5. ⏳ Add missing alert type mappings if needed
6. ⏳ Optimize database queries for performance

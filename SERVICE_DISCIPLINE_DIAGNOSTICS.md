# Service Discipline Monitor - Diagnostics Guide

## Why No Violations Are Being Stored

If you're not seeing any violations stored in the database, check the following:

### 1. **Table ROIs Not Configured**
**Symptom**: No tables being tracked
**Check**: Look for this log message:
```
⚠️ No tables being tracked! Check table ROIs configuration.
```

**Solution**: 
- Configure table ROIs in `config/channels.json` under the `ServiceDisciplineMonitor` module
- Or use the dashboard to set table ROIs
- Verify tables are loaded: Look for log message like:
  ```
  ✅ Successfully loaded X table ROIs from channels.json
  ```

### 2. **No Customers Detected**
**Symptom**: Customers not being detected as seated
**Check**: Look for these log messages:
```
🪑 T_seated: Customer track X at table Y
```

**Possible Causes**:
- Person detection not working (check YOLO model is loaded)
- Customers being classified as waiters (uniform detection too close)
- Customers not inside table ROI polygons

**Solution**:
- Verify person detection is working (check other modules)
- Adjust `interaction_distance` setting if customers are misclassified
- Verify table ROI polygons are correctly drawn

### 3. **Thresholds Too High**
**Symptom**: Violations detected but not saved (wait times below threshold)
**Check**: Current thresholds:
- `order_wait_threshold`: Default 120 seconds (2 minutes)
- `service_wait_threshold`: Default 300 seconds (5 minutes)

**Solution**: 
- Lower thresholds in `config/channels.json` or via dashboard
- Check logs for: `Order wait X.Xs (threshold: Y.Ys) - OK` (not exceeding threshold)

### 4. **Database Save Failing**
**Symptom**: Violations detected but database save fails
**Check**: Look for error messages:
```
❌ Failed to save service discipline violation: [error message]
```

**Common Issues**:
- Database connection problems
- Missing database tables (run migrations)
- Permission issues with snapshot directory

**Solution**:
- Check database connection
- Verify `table_service_violations` table exists
- Check `static/service_discipline` directory exists and is writable

### 5. **Cooldown Period Active**
**Symptom**: Violation detected but not saved (in cooldown)
**Check**: Look for log message:
```
Violation detected but in cooldown: X.Xs / Y.Ys
```

**Solution**: 
- Default cooldown is 300 seconds (5 minutes)
- Adjust `alert_cooldown` setting if needed

## Diagnostic Logging

The code now includes enhanced logging to help diagnose issues:

### Status Logs (every 10 seconds):
```
📊 Violation check: X tables, Y customers tracked, thresholds: order=120s, service=300s
```

### Violation Detection:
```
⚠️ Order wait violation: Table X, customer Y waiting Z.Zs (threshold: W.Ws)
📊 Violation details: T_seated=..., T_order_start=..., order_wait=Z.Zs
💾 Saving violation to database: Table X, order_wait = Z.Zs
✅ Violation saved to table_service_violations: [result]
✅ Alert logged to general alerts table: [message]
```

### Errors:
```
❌ Failed to save service discipline violation: [error]
⚠️ db_manager is None - cannot save violation to database
```

## Quick Checklist

1. ✅ **Table ROIs configured?** Check logs for "Successfully loaded X table ROIs"
2. ✅ **Customers detected?** Look for "T_seated" log messages
3. ✅ **Wait times exceeding thresholds?** Check violation detection logs
4. ✅ **Database connection working?** Check for database errors
5. ✅ **Cooldown expired?** Wait 5 minutes between alerts per table

## Testing

To test if the system is working:

1. **Lower thresholds temporarily** (e.g., 10 seconds for order wait)
2. **Sit at a table** and wait without ordering
3. **Check logs** for violation detection and database save messages
4. **Check database** directly: `SELECT * FROM table_service_violations ORDER BY timestamp DESC LIMIT 10;`
5. **Check dashboard** for new violations

## Configuration

Edit `config/channels.json` or use the dashboard to configure:

```json
{
  "ServiceDisciplineMonitor": {
    "config": {
      "table_rois": {
        "table_1": {
          "points": [{"x": 0.2, "y": 0.3}, ...]
        }
      },
      "order_wait_threshold": 60.0,
      "service_wait_threshold": 180.0,
      "alert_cooldown": 300.0
    }
  }
}
```


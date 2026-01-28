# GIF Alert Implementation Guide

## Overview
All alert modules (except Dress Code Monitoring, PPE Compliance, Smoke/Fire Detection, and Smoking Detection) should save alerts as GIFs instead of static snapshots.

**Key Point:** GIFs are stored in the same `snapshot_path` database fields that snapshots used. The dashboard will automatically display GIFs - no changes needed to the dashboard code.

## Modules That Should Use GIFs

✅ **Already Implemented:**
- Queue Monitor - Uses `AlertGifRecorder`
- Bag Detection - Uses `AlertGifRecorder`
- Unauthorized Entry Monitor - Uses `AlertGifRecorder`

❌ **Need GIF Support:**
- Service Discipline Monitor
- Table Service Monitor (Table Cleanliness)
- Material Theft Monitor
- Cash Detection
- Fall Detection
- Person Smoking Detection
- Crowd Detection
- Any other alert modules

## Implementation Pattern

### Option 1: Using GIF Alert Helper (Recommended - Easiest)

The `GifAlertHelper` class provides a simple interface for adding GIF support:

```python
from .gif_alert_helper import GifAlertHelper

def __init__(self, channel_id, socketio, db_manager=None, app=None):
    # ... existing code ...
    
    # Initialize GIF alert helper
    self.gif_helper = GifAlertHelper(channel_id, db_manager, app, socketio)
    self.gif_helper.initialize_gif_recorder(
        buffer_size=90,  # 3 seconds at 30fps
        gif_duration=4.0,  # 4 second GIFs
        fps=30
    )

def process_frame(self, frame):
    # ... existing detection code ...
    
    # Add frame to buffer (always, for pre-alert context)
    self.gif_helper.add_frame_to_buffer(frame)
    
    # ... detection and alert logic ...
    
    # When alert triggers:
    if alert_condition:
        self.gif_helper.start_alert_recording(
            alert_type='your_alert_type',  # e.g., 'service_discipline_alert'
            alert_message='Alert message',
            frame=frame,
            alert_data={'key': 'value'}
        )
    
    # Continue recording if in progress
    if self.gif_helper.is_recording():
        self.gif_helper.add_alert_frame(frame)
        
        # Check if recording completed
        if self.gif_helper.is_recording_complete():
            gif_info = self.gif_helper.get_completed_gif()
            self.gif_helper.save_alert_to_database(
                gif_info=gif_info,
                alert_type='your_alert_type',
                alert_message='Alert message',
                alert_data={'key': 'value'}
            )
```

### Option 2: Direct GIF Recorder Usage (More Control)

If you need more control, use `AlertGifRecorder` directly:

```python
from .gif_recorder import AlertGifRecorder

def __init__(self, channel_id, socketio, db_manager=None, app=None):
    # ... existing code ...
    
    # Initialize GIF recorder for alerts
    self.gif_recorder = AlertGifRecorder(
        buffer_size=90,  # 3 seconds at 30fps
        gif_duration=4.0,  # 4 second GIFs
        fps=30
    )

def process_frame(self, frame):
    # ... existing detection code ...
    
    # Add frame to GIF recorder buffer (always, for pre-alert context)
    self.gif_recorder.add_frame(frame)
    
    # ... rest of processing ...
    
    # When alert triggers:
    if alert_condition:
        alert_info = {
            'message': 'Alert message',
            'channel_id': self.channel_id,
            'alert_type': 'your_alert_type'
        }
        self.gif_recorder.start_alert_recording(alert_info)
        self.gif_recorder.add_alert_frame(frame)
    
    # Continue recording if in progress
    if self.gif_recorder.is_recording_alert:
        self.gif_recorder.add_alert_frame(frame)
        
        # Check if recording completed automatically
        if not self.gif_recorder.is_recording_alert:
            gif_info = self.gif_recorder.get_last_gif_info()
            if gif_info:
                self._save_alert_with_gif(gif_info, alert_data)

def _save_alert_with_gif(self, gif_info, alert_data):
    """Save alert with GIF to database"""
    if not self.db_manager:
        return
    
    try:
        if self.app:
            with self.app.app_context():
                gif_id = self.db_manager.save_alert_gif(
                    channel_id=self.channel_id,
                    alert_type='your_alert_type',
                    gif_info=gif_info,
                    alert_message=alert_data.get('message'),
                    alert_data=alert_data
                )
        else:
            gif_id = self.db_manager.save_alert_gif(
                channel_id=self.channel_id,
                alert_type='your_alert_type',
                gif_info=gif_info,
                alert_message=alert_data.get('message'),
                alert_data=alert_data
            )
        
        logger.info(f"Alert GIF saved: ID {gif_id}")
        
    except Exception as e:
        logger.error(f"Error saving alert GIF: {e}")
```

## Database Integration

**Important:** Save GIF paths in the same `snapshot_path` field that snapshots used. The dashboard will automatically display GIFs (browsers support GIFs).

```python
# Get GIF path for violation table
gif_path = self.gif_helper.get_snapshot_path_for_violation(gif_info)
# Returns: "static/alerts/alert_20260127_103000.gif"

# Save to database - use GIF path in snapshot_path field
# The dashboard will display it correctly (GIFs work just like images)
self.db_manager.add_table_service_violation(
    channel_id=self.channel_id,
    table_id=table_id,
    waiting_time=wait_time,
    snapshot_path=gif_path,  # GIF path goes here - same field as snapshots
    snapshot_filename=os.path.basename(gif_path),  # Just the filename
    # ... other parameters
)
```

**No dashboard changes needed** - Browsers automatically display GIFs when you use them in `<img>` tags, so your existing dashboard code will work.

## Simple Example: Replacing Snapshot with GIF

**Before (saving snapshot):**
```python
# Old way - saving static snapshot
snapshot_path = f"static/violations/alert_{timestamp}.jpg"
cv2.imwrite(snapshot_path, frame)

# Save to database
db_manager.add_violation(
    channel_id=self.channel_id,
    snapshot_path=snapshot_path,
    snapshot_filename=os.path.basename(snapshot_path),
    # ... other fields
)
```

**After (saving GIF):**
```python
# New way - saving GIF
from .gif_alert_helper import GifAlertHelper

# In __init__:
self.gif_helper = GifAlertHelper(channel_id, db_manager, app, socketio)
self.gif_helper.initialize_gif_recorder()

# In process_frame:
self.gif_helper.add_frame_to_buffer(frame)

# When alert triggers:
self.gif_helper.start_alert_recording('alert_type', 'Alert message', frame)

# Continue in process_frame:
if self.gif_helper.is_recording():
    self.gif_helper.add_alert_frame(frame)
    if self.gif_helper.is_recording_complete():
        gif_info = self.gif_helper.get_completed_gif()
        gif_path = self.gif_helper.get_snapshot_path_for_violation(gif_info)
        
        # Save to database - same fields, just GIF instead of snapshot
        db_manager.add_violation(
            channel_id=self.channel_id,
            snapshot_path=gif_path,  # GIF path goes here
            snapshot_filename=os.path.basename(gif_path),
            # ... other fields
        )
```

## Modules to Update

Based on the user's requirement, update these modules to use GIFs:

1. **Service Discipline Monitor** - Replace `_save_snapshot_new` with GIF recording
2. **Table Service Monitor** - Replace snapshot saving with GIF recording
3. **Material Theft Monitor** - Add GIF recording
4. **Cash Detection** - Add GIF recording
5. **Fall Detection** - Add GIF recording
6. **Person Smoking Detection** - Add GIF recording
7. **Crowd Detection** - Add GIF recording

## Notes

- **No dashboard changes needed** - GIFs work in `<img>` tags, so your existing dashboard will display them automatically
- GIFs are stored in the same `snapshot_path` database fields that snapshots used
- GIF files are stored in `static/alerts/` directory (or module-specific directories)
- GIF recorder automatically includes pre-alert frames (last 1 second)
- GIF duration is configurable (default: 4 seconds)
- Database `alert_gifs` table stores GIF metadata (optional - you can also just use violation tables)
- Violation tables store GIF paths in `snapshot_path` field - same as snapshots, just with `.gif` extension

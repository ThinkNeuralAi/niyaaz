# Database Design - Sakshi.AI Video Analytics Platform

## Overview
This document describes the complete database schema for the Sakshi.AI video analytics platform. The database uses SQLAlchemy ORM and supports both PostgreSQL (via environment variables) and SQLite (fallback).

    ## Database Tables

    ### 1. **users**
    User authentication and authorization table.

    | Column | Type | Constraints | Description |
    |--------|------|-------------|-------------|
    | `id` | Integer | Primary Key | Auto-incrementing user ID |
    | `username` | String(50) | Unique, Not Null | Username for login |
    | `password_hash` | String(255) | Not Null | Hashed password |
    | `role` | String(20) | Not Null | User role: 'admin' or 'user' |
    | `created_at` | DateTime | Default: IST Now | Account creation timestamp |
    | `last_login` | DateTime | Nullable | Last login timestamp |

    **Indexes:**
    - Unique index on `username`

    ---

    ### 2. **rtsp_channels**
    RTSP camera channel configuration.

    | Column | Type | Constraints | Description |
    |--------|------|-------------|-------------|
    | `id` | Integer | Primary Key | Auto-incrementing channel ID |
    | `channel_id` | String(50) | Unique, Not Null | Unique channel identifier |
    | `name` | String(100) | Not Null | Human-readable channel name |
    | `rtsp_url` | String(500) | Not Null | RTSP stream URL |
    | `description` | Text | Nullable | Channel description |
    | `is_active` | Boolean | Default: True | Whether channel is active |
    | `created_at` | DateTime | Default: IST Now | Creation timestamp |
    | `updated_at` | DateTime | Default: IST Now, On Update | Last update timestamp |

    **Indexes:**
    - Unique index on `channel_id`

    ---

    ### 3. **channel_config**
    Configuration storage for modules (ROI, counting lines, settings).

    | Column | Type | Constraints | Description |
    |--------|------|-------------|-------------|
    | `id` | Integer | Primary Key | Auto-incrementing config ID |
    | `channel_id` | String(50) | Not Null | Channel identifier |
    | `app_name` | String(50) | Not Null | Module name (e.g., 'QueueMonitor', 'CrowdDetection') |
    | `config_type` | String(50) | Not Null | Type: 'roi', 'counting_line', 'settings' |
    | `config_data` | Text | Nullable | JSON configuration data |
    | `created_at` | DateTime | Default: IST Now | Creation timestamp |
    | `updated_at` | DateTime | Default: IST Now, On Update | Last update timestamp |

    **Indexes:**
    - Unique constraint on (`channel_id`, `app_name`, `config_type`)

    ---

    ### 4. **detection_events**
    Audit trail for detection events.

    | Column | Type | Constraints | Description |
    |--------|------|-------------|-------------|
    | `id` | Integer | Primary Key | Auto-incrementing event ID |
    | `channel_id` | String(50) | Not Null | Channel identifier |
    | `app_name` | String(50) | Not Null | Module name |
    | `event_type` | String(50) | Not Null | Event type: 'person_in', 'person_out', 'queue_alert', etc. |
    | `event_data` | Text | Nullable | JSON event data |
    | `confidence` | Float | Nullable | Detection confidence score |
    | `timestamp` | DateTime | Default: IST Now | Event timestamp |

    ---

    ### 5. **daily_footfall**
    Daily footfall statistics for People Counter.

    | Column | Type | Constraints | Description |
    |--------|------|-------------|-------------|
    | `id` | Integer | Primary Key | Auto-incrementing record ID |
    | `channel_id` | String(50) | Not Null | Channel identifier |
    | `report_date` | Date | Not Null | Date of the report |
    | `in_count` | Integer | Default: 0 | Number of people entering |
    | `out_count` | Integer | Default: 0 | Number of people exiting |

    **Indexes:**
    - Unique constraint on (`channel_id`, `report_date`)

    ---

    ### 6. **hourly_footfall**
    Hourly footfall statistics for People Counter.

    | Column | Type | Constraints | Description |
    |--------|------|-------------|-------------|
    | `id` | Integer | Primary Key | Auto-incrementing record ID |
    | `channel_id` | String(50) | Not Null | Channel identifier |
    | `report_date` | Date | Not Null | Date of the report |
    | `hour` | Integer | Not Null | Hour of day (0-23) |
    | `in_count` | Integer | Default: 0 | Number of people entering |
    | `out_count` | Integer | Default: 0 | Number of people exiting |

    **Indexes:**
    - Unique constraint on (`channel_id`, `report_date`, `hour`)

    ---

    ### 7. **queue_analytics**
    Queue monitoring analytics data.

    | Column | Type | Constraints | Description |
    |--------|------|-------------|-------------|
    | `id` | Integer | Primary Key | Auto-incrementing record ID |
    | `channel_id` | String(50) | Not Null | Channel identifier |
    | `timestamp` | DateTime | Default: IST Now | Record timestamp |
    | `queue_count` | Integer | Default: 0 | Number of people in queue |
    | `counter_count` | Integer | Default: 0 | Number of people at counter |
    | `alert_triggered` | Boolean | Default: False | Whether alert was triggered |
    | `alert_message` | Text | Nullable | Alert message |

    ---

    ### 8. **alert_gifs**
    Alert GIF recordings for various alert types.

    | Column | Type | Constraints | Description |
    |--------|------|-------------|-------------|
    | `id` | Integer | Primary Key | Auto-incrementing alert ID |
    | `channel_id` | String(50) | Not Null | Channel identifier |
    | `alert_type` | String(50) | Not Null | Alert type: 'queue_alert', 'people_alert', 'unauthorized_entry_alert', etc. |
    | `gif_filename` | String(255) | Not Null | GIF filename |
    | `gif_path` | String(500) | Not Null | Relative path to GIF file |
    | `alert_message` | Text | Nullable | Alert message |
    | `alert_data` | Text | Nullable | JSON data with alert details |
    | `frame_count` | Integer | Nullable | Number of frames in GIF |
    | `file_size` | Integer | Nullable | File size in bytes |
    | `duration_seconds` | Float | Nullable | Duration of alert in seconds |
    | `created_at` | DateTime | Default: IST Now | Alert timestamp |

    **Common Alert Types:**
    - `queue_alert`
    - `people_alert`
    - `unauthorized_entry_alert`
    - `person_smoking_alert`
    - `fire_smoke_alert`
    - `fall_alert`
    - `material_theft_alert`

    ---

    ### 9. **heatmap_snapshots**
    Heatmap visualization snapshots.

    | Column | Type | Constraints | Description |
    |--------|------|-------------|-------------|
    | `id` | Integer | Primary Key | Auto-incrementing snapshot ID |
    | `channel_id` | String(50) | Not Null | Channel identifier |
    | `snapshot_filename` | String(255) | Not Null | Snapshot filename |
    | `snapshot_path` | String(500) | Not Null | Relative path to snapshot |
    | `hotspot_count` | Integer | Default: 0 | Number of hotspots detected |
    | `hotspots_data` | Text | Nullable | JSON data with hotspot locations |
    | `created_at` | DateTime | Default: IST Now | Snapshot timestamp |

    ---

    ### 10. **cash_snapshots**
    Cash detection snapshots.

    | Column | Type | Constraints | Description |
    |--------|------|-------------|-------------|
    | `id` | Integer | Primary Key | Auto-incrementing snapshot ID |
    | `channel_id` | String(50) | Not Null | Channel identifier |
    | `snapshot_filename` | String(255) | Not Null | Snapshot filename |
    | `snapshot_path` | String(500) | Not Null | Relative path to snapshot |
    | `alert_message` | Text | Nullable | Alert message |
    | `alert_data` | Text | Nullable | JSON data with detection details |
    | `detection_count` | Integer | Default: 0 | Number of cash detections |
    | `file_size` | Integer | Nullable | File size in bytes |
    | `created_at` | DateTime | Default: IST Now | Snapshot timestamp |

    ---

    ### 11. **fall_snapshots**
    Fall detection snapshots.

    | Column | Type | Constraints | Description |
    |--------|------|-------------|-------------|
    | `id` | Integer | Primary Key | Auto-incrementing snapshot ID |
    | `channel_id` | String(50) | Not Null | Channel identifier |
    | `snapshot_filename` | String(255) | Not Null | Snapshot filename |
    | `snapshot_path` | String(500) | Not Null | Relative path to snapshot |
    | `alert_message` | Text | Nullable | Alert message |
    | `alert_data` | Text | Nullable | JSON data with fall detection details |
    | `fall_duration` | Float | Nullable | Duration person was fallen (seconds) |
    | `file_size` | Integer | Nullable | File size in bytes |
    | `created_at` | DateTime | Default: IST Now | Snapshot timestamp |

    ---

    ### 12. **grooming_snapshots**
    Grooming standards violation snapshots.

    | Column | Type | Constraints | Description |
    |--------|------|-------------|-------------|
    | `id` | Integer | Primary Key | Auto-incrementing snapshot ID |
    | `channel_id` | String(50) | Not Null | Channel identifier |
    | `snapshot_filename` | String(255) | Not Null | Snapshot filename |
    | `snapshot_path` | String(500) | Not Null | Relative path to snapshot |
    | `alert_message` | Text | Nullable | Alert message |
    | `alert_data` | Text | Nullable | JSON data with violation details |
    | `violation_type` | String(100) | Nullable | Type: 'missing_required', 'prohibited_item' |
    | `violation_item` | String(100) | Nullable | Item: 'uniform', 'name_tag', 'long_hair', etc. |
    | `file_size` | Integer | Nullable | File size in bytes |
    | `created_at` | DateTime | Default: IST Now | Snapshot timestamp |

    ---

    ### 13. **dresscode_alerts**
    Dress code monitoring violation alerts.

    | Column | Type | Constraints | Description |
    |--------|------|-------------|-------------|
    | `id` | Integer | Primary Key | Auto-incrementing alert ID |
    | `channel_id` | String(50) | Not Null | Channel identifier |
    | `employee_id` | String(50) | Nullable | Employee identifier |
    | `snapshot_filename` | String(255) | Not Null | Snapshot filename |
    | `snapshot_path` | String(500) | Not Null | Relative path to snapshot |
    | `violations` | Text | Nullable | Comma-separated list of violations |
    | `uniform_color` | String(50) | Nullable | Detected uniform color: 'grey', 'black', 'beige', 'blue', 'red' |
    | `alert_data` | Text | Nullable | JSON data with detailed violation info |
    | `is_compliant` | Boolean | Default: False | Whether employee is compliant |
    | `file_size` | Integer | Nullable | File size in bytes |
    | `created_at` | DateTime | Default: IST Now | Alert timestamp |

    ---

    ### 14. **ppe_alerts**
    PPE (Personal Protective Equipment) monitoring violation alerts.

    | Column | Type | Constraints | Description |
    |--------|------|-------------|-------------|
    | `id` | Integer | Primary Key | Auto-incrementing alert ID |
    | `channel_id` | String(50) | Not Null | Channel identifier |
    | `employee_id` | String(50) | Nullable | Employee identifier |
    | `snapshot_filename` | String(255) | Nullable | Snapshot filename |
    | `snapshot_path` | String(500) | Nullable | Relative path to snapshot |
    | `violations` | Text | Not Null | Comma-separated list: "No apron", "No gloves", "No hairnet" |
    | `violation_types` | Text | Nullable | JSON array of violation types |
    | `alert_data` | Text | Nullable | JSON data with detailed violation info |
    | `is_compliant` | Boolean | Default: False | Whether employee is compliant |
    | `file_size` | Integer | Nullable | File size in bytes (if snapshot exists) |
    | `created_at` | DateTime | Default: IST Now | Alert timestamp |

    ---

    ### 15. **queue_violations**
    Queue monitoring violation records.

    | Column | Type | Constraints | Description |
    |--------|------|-------------|-------------|
    | `id` | Integer | Primary Key | Auto-incrementing violation ID |
    | `channel_id` | String(50) | Not Null | Channel identifier |
    | `snapshot_filename` | String(255) | Nullable | Snapshot filename |
    | `snapshot_path` | String(500) | Nullable | Relative path to snapshot |
    | `violation_type` | String(100) | Not Null | Type: 'queue_too_long', 'wait_time_exceeded', 'no_counter_staff' |
    | `violation_message` | Text | Not Null | Violation message |
    | `queue_count` | Integer | Default: 0 | Number of people in queue |
    | `counter_count` | Integer | Default: 0 | Number of people at counter |
    | `wait_time_seconds` | Float | Default: 0.0 | Wait time in seconds |
    | `alert_data` | Text | Nullable | JSON data with detailed violation info |
    | `file_size` | Integer | Nullable | File size in bytes (if snapshot exists) |
    | `created_at` | DateTime | Default: IST Now | Violation timestamp |

    ---

    ### 16. **mopping_snapshots**
    Mopping detection snapshots.

    | Column | Type | Constraints | Description |
    |--------|------|-------------|-------------|
    | `id` | Integer | Primary Key | Auto-incrementing snapshot ID |
    | `channel_id` | String(50) | Not Null | Channel identifier |
    | `snapshot_filename` | String(255) | Not Null | Snapshot filename |
    | `snapshot_path` | String(500) | Not Null | Relative path to snapshot |
    | `alert_message` | Text | Nullable | Alert message |
    | `alert_data` | Text | Nullable | JSON data with detection details |
    | `detection_count` | Integer | Default: 0 | Number of mopping detections |
    | `detection_time` | DateTime | Nullable | When mopping was detected |
    | `file_size` | Integer | Nullable | File size in bytes |
    | `created_at` | DateTime | Default: IST Now | Snapshot timestamp |

    ---

    ### 17. **smoking_snapshots**
    Smoking detection snapshots.

    | Column | Type | Constraints | Description |
    |--------|------|-------------|-------------|
    | `id` | Integer | Primary Key | Auto-incrementing snapshot ID |
    | `channel_id` | String(50) | Not Null | Channel identifier |
    | `snapshot_filename` | String(255) | Not Null | Snapshot filename |
    | `snapshot_path` | String(500) | Not Null | Relative path to snapshot |
    | `alert_message` | Text | Nullable | Alert message |
    | `alert_data` | Text | Nullable | JSON data with detection details |
    | `detection_count` | Integer | Default: 0 | Number of smoking detections |
    | `detection_time` | DateTime | Nullable | When smoking was detected |
    | `file_size` | Integer | Nullable | File size in bytes |
    | `created_at` | DateTime | Default: IST Now | Snapshot timestamp |

    ---

    ### 18. **phone_snapshots**
    Phone usage detection snapshots.

    | Column | Type | Constraints | Description |
    |--------|------|-------------|-------------|
    | `id` | Integer | Primary Key | Auto-incrementing snapshot ID |
    | `channel_id` | String(50) | Not Null | Channel identifier |
    | `snapshot_filename` | String(255) | Not Null | Snapshot filename |
    | `snapshot_path` | String(500) | Not Null | Relative path to snapshot |
    | `alert_message` | Text | Nullable | Alert message |
    | `alert_data` | Text | Nullable | JSON data with detection details |
    | `detection_count` | Integer | Default: 0 | Number of phone detections |
    | `detection_time` | DateTime | Nullable | When phone usage was detected |
    | `file_size` | Integer | Nullable | File size in bytes |
    | `created_at` | DateTime | Default: IST Now | Snapshot timestamp |

    ---

    ### 19. **restricted_area_snapshots**
    Restricted area monitoring violation snapshots.

    | Column | Type | Constraints | Description |
    |--------|------|-------------|-------------|
    | `id` | Integer | Primary Key | Auto-incrementing snapshot ID |
    | `channel_id` | String(50) | Not Null | Channel identifier |
    | `snapshot_filename` | String(255) | Not Null | Snapshot filename |
    | `snapshot_path` | String(500) | Not Null | Relative path to snapshot |
    | `alert_message` | Text | Nullable | Alert message |
    | `alert_data` | Text | Nullable | JSON data with violation details |
    | `violation_count` | Integer | Default: 0 | Number of violations |
    | `detection_time` | DateTime | Nullable | When violation was detected |
    | `file_size` | Integer | Nullable | File size in bytes |
    | `created_at` | DateTime | Default: IST Now | Snapshot timestamp |

    ---

    ### 20. **table_service_violations**
    Table service discipline violation records.

    | Column | Type | Constraints | Description |
    |--------|------|-------------|-------------|
    | `id` | Integer | Primary Key | Auto-incrementing violation ID |
    | `channel_id` | String(50) | Not Null | Channel identifier |
    | `table_id` | String(50) | Not Null | Table identifier |
    | `waiting_time` | Float | Not Null | Waiting time in seconds |
    | `snapshot_filename` | String(255) | Nullable | Snapshot filename |
    | `snapshot_path` | String(500) | Nullable | Relative path to snapshot |
    | `alert_data` | Text | Nullable | JSON data with detailed violation info |
    | `file_size` | Integer | Nullable | File size in bytes (if snapshot exists) |
    | `created_at` | DateTime | Default: IST Now | Violation timestamp |

    ---

    ### 21. **table_cleanliness_violations**
    Table cleanliness violation records (unclean tables / slow reset).

    | Column | Type | Constraints | Description |
    |--------|------|-------------|-------------|
    | `id` | Integer | Primary Key | Auto-incrementing violation ID |
    | `channel_id` | String(50) | Not Null | Channel identifier |
    | `table_id` | String(50) | Not Null | Table identifier |
    | `violation_type` | String(50) | Not Null | Type: 'unclean_table' or 'slow_reset' |
    | `snapshot_filename` | String(255) | Nullable | Snapshot filename |
    | `snapshot_path` | String(500) | Nullable | Relative path to snapshot |
    | `alert_data` | Text | Nullable | JSON data with detailed violation info |
    | `file_size` | Integer | Nullable | File size in bytes (if snapshot exists) |
    | `created_at` | DateTime | Default: IST Now | Violation timestamp |

    ---

## Relationships

### Logical Relationships (No Foreign Keys Defined)

1. **Channel-based relationships:**
   - All tables with `channel_id` reference `rtsp_channels.channel_id` (logically)
   - Multiple modules can share the same channel

2. **Configuration relationships:**
   - `channel_config` stores configuration for each `channel_id` + `app_name` combination

3. **Analytics relationships:**
   - `daily_footfall` and `hourly_footfall` aggregate data by `channel_id` and date
   - `queue_analytics` tracks real-time queue metrics by `channel_id`

4. **Alert/Snapshot relationships:**
   - All alert and snapshot tables are linked to `channel_id`
   - `alert_gifs` is a general-purpose table for various alert types
   - Specific snapshot tables (e.g., `cash_snapshots`, `fall_snapshots`) store module-specific data

---

## Common Patterns

### JSON Data Fields
Many tables include `alert_data` or `event_data` columns that store JSON data. Common JSON structures:

- **alert_data** (in snapshots/alerts):
  ```json
  {
    "detection_count": 2,
    "confidence": 0.85,
    "bounding_boxes": [...],
    "custom_fields": {...}
  }
  ```

- **config_data** (in channel_config):
  ```json
  {
    "roi_points": [[x1, y1], [x2, y2], ...],
    "thresholds": {...},
    "settings": {...}
  }
  ```

### Timestamp Fields
- All tables use `created_at` with default `get_ist_now()` (IST timezone)
- Some tables have `updated_at` for tracking modifications
- `detection_time` fields store specific event timestamps

### File Storage
- All snapshot/alert tables store:
  - `snapshot_filename` or `gif_filename`: Just the filename
  - `snapshot_path` or `gif_path`: Relative path from static directory
  - `file_size`: Size in bytes

---

## Indexes and Constraints

### Primary Keys
- All tables have an auto-incrementing `id` as primary key

### Unique Constraints
- `users.username`: Unique
- `rtsp_channels.channel_id`: Unique
- `channel_config(channel_id, app_name, config_type)`: Unique
- `daily_footfall(channel_id, report_date)`: Unique
- `hourly_footfall(channel_id, report_date, hour)`: Unique

### Common Query Patterns
- Most queries filter by `channel_id`
- Time-based queries use `created_at` with date ranges
- Alert queries filter by `alert_type` (in `alert_gifs`)

---

## Database Initialization

The database is initialized using Flask-SQLAlchemy. Models are defined dynamically in `DatabaseManager.define_models()` method.

**Supported Databases:**
- PostgreSQL (via environment variables: `DATABASE_URL`)
- SQLite (fallback, default: `instance/sakshi_ai.db`)

**Timezone:**
- All timestamps use IST (Indian Standard Time) via `get_ist_now()` function

---

## Notes

1. **No Foreign Key Constraints**: The schema uses logical relationships rather than database-enforced foreign keys for flexibility.

2. **Flexible JSON Storage**: Many fields use JSON (`Text` type) to allow schema evolution without migrations.

3. **File Paths**: All file paths are stored as relative paths from the `static/` directory.

4. **Alert Types**: The `alert_gifs.alert_type` field is used to categorize different types of alerts across the system.

5. **Channel Configuration**: The `channel_config` table allows storing multiple configuration types per channel per module, enabling flexible ROI and settings management.





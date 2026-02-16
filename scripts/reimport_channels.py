
import os
import json
import psycopg2
from dotenv import load_dotenv

load_dotenv()

host = os.getenv('DB_HOST', 'localhost')
port = os.getenv('DB_PORT', '5432')
dbname = os.getenv('DB_NAME', 'sakshiai')
user = os.getenv('DB_USER', 'postgres')
password = os.getenv('DB_PASSWORD')

def get_db_connection():
    conn = psycopg2.connect(
        host=host,
        port=port,
        dbname=dbname,
        user=user,
        password=password
    )
    return conn

def reimport_channels():
    conn = get_db_connection()
    cur = conn.cursor()

    try:
        with open('config/channels.json', 'r') as f:
            data = json.load(f)

        print(f"Loaded {len(data['channels'])} channels from config/channels.json")

        for channel in data['channels']:
            channel_id = channel['channel_id']
            store_id = channel['store_id']
            channel_name = channel['channel_name']
            rtsp_url = channel['rtsp_url']
            enabled = channel.get('enabled', True)

            # 1. Update rtsp_links
            # Check if exists
            cur.execute("SELECT id FROM rtsp_links WHERE channel_id = %s", (channel_id,))
            res = cur.fetchone()

            if res:
                print(f"Updating rtsp_links for {channel_id}...")
                cur.execute("""
                    UPDATE rtsp_links 
                    SET channel_name = %s, rtsp_url = %s, store_id = %s, is_active = %s, updated_at = NOW()
                    WHERE channel_id = %s
                """, (channel_name, rtsp_url, store_id, enabled, channel_id))
            else:
                print(f"Inserting into rtsp_links for {channel_id}...")
                cur.execute("""
                    INSERT INTO rtsp_links (channel_id, store_id, channel_name, rtsp_url, is_active, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, NOW(), NOW())
                """, (channel_id, store_id, channel_name, rtsp_url, enabled))

            # 2. Update channel_config
            modules = channel.get('modules', [])
            for module in modules:
                module_type = module['type']
                config = module.get('config', {})
                
                # --- QueueMonitor ---
                if module_type == 'QueueMonitor':
                    # ROI Config
                    # Expects: "points" list inside "queue_roi" or "counter_roi" in JSON
                    # DB Expects: "main" and "secondary" lists of points in ONE "roi" config
                    
                    roi_data = {
                        "main": config.get('queue_roi', {}).get('points', []),
                        "secondary": config.get('counter_roi', {}).get('points', [])
                    }
                    
                    save_config(cur, channel_id, module_type, 'roi', roi_data)
                    
                    # Settings
                    settings = config.get('settings', {})
                    save_config(cur, channel_id, module_type, 'settings', settings)

                # --- DressCodeMonitoring ---
                elif module_type == 'DressCodeMonitoring':
                    # Counter ROI
                    counter_roi = config.get('counter_roi', {}).get('points', [])
                    if counter_roi:
                        save_config(cur, channel_id, module_type, 'counter_roi', counter_roi)
                    
                    # Queue ROI (if any)
                    queue_roi = config.get('queue_roi', {}).get('points', [])
                    if queue_roi:
                        save_config(cur, channel_id, module_type, 'queue_roi', queue_roi)
                        
                    # Allowed Uniforms
                    allowed_uniforms = config.get('allowed_uniforms', {})
                    if allowed_uniforms:
                        save_config(cur, channel_id, module_type, 'allowed_uniforms', allowed_uniforms)
                        
                    # Settings
                    settings = config.get('settings', {})
                    if settings:
                        save_config(cur, channel_id, module_type, 'settings', settings)

                # --- ServiceDisciplineMonitor & TableServiceMonitor ---
                elif module_type in ['ServiceDisciplineMonitor', 'TableServiceMonitor']:
                    # Table ROIs
                    table_rois = config.get('table_rois', {})
                    # The JSON has "table_rois": { "table_1": { "points": [...] } }
                    # The module likely expects this structure directly or a list of tables.
                    # Looking at JSON, it seems consistent.
                    if table_rois:
                        save_config(cur, channel_id, module_type, 'table_rois', table_rois)

                    # Settings
                    settings = config.get('settings', {})
                    if settings:
                        save_config(cur, channel_id, module_type, 'settings', settings)

                # --- Generic / Other Modules (FallDetection, SmokingDetection, etc.) ---
                else:
                    # Settings
                    settings = config.get('settings', {})
                    if settings:
                        save_config(cur, channel_id, module_type, 'settings', settings)
                        
                    # Config (some modules might have direct config keys)
                    # For now, we assume 'settings' is the main one. 
                    # If there are specific keys like 'conf_threshold' outside settings (e.g. CashDetection), handle them.
                    
                    # Check for direct keys that are not 'settings' or ROIs
                    direct_config = {k: v for k, v in config.items() if k not in ['settings', 'queue_roi', 'counter_roi', 'table_rois', 'allowed_uniforms']}
                    if direct_config:
                        # Some modules might expect these as "settings" or just merged. 
                        # Looking at channels.json for CashDetection: "config": { "conf_threshold": 0.7 ... }
                        # It doesn't have a "settings" sub-key.
                        # So for these, we might need to save 'settings' as the whole config object if 'settings' key is missing.
                        
                        if not settings:
                             save_config(cur, channel_id, module_type, 'settings', direct_config)

                # 3. Update channel_modules (User request: "channel modules also")
                # This table seems to store the RAW config from json in 'config_data'
                # Schema: channel_id, store_id, module_name, module_type, enabled, config_data, is_default
                
                # Check if exists
                cur.execute("""
                    SELECT id FROM channel_modules
                    WHERE channel_id = %s AND module_name = %s
                """, (channel_id, module_type))
                res_mod = cur.fetchone()
                
                raw_config_json = json.dumps(config)
                
                if res_mod:
                    cur.execute("""
                        UPDATE channel_modules
                        SET config_data = %s, updated_at = NOW(), enabled = %s, store_id = %s
                        WHERE id = %s
                    """, (raw_config_json, enabled, store_id, res_mod[0]))
                    print(f"  Updated channel_modules for {channel_id} - {module_type}")
                else:
                    cur.execute("""
                        INSERT INTO channel_modules (channel_id, store_id, module_name, module_type, enabled, config_data, is_default, created_at, updated_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
                    """, (channel_id, store_id, module_type, module_type, enabled, raw_config_json, False))
                    print(f"  Inserted into channel_modules for {channel_id} - {module_type}")

        conn.commit()
        print("Re-import successful.")

    except Exception as e:
        conn.rollback()
        print(f"Error during re-import: {e}")
    finally:
        cur.close()
        conn.close()

def save_config(cur, channel_id, app_name, config_type, config_data):
    # Upsert into channel_config
    # Table: channel_config (channel_id, app_name, config_type, config_data)
    
    # Check if exists
    cur.execute("""
        SELECT id FROM channel_config 
        WHERE channel_id = %s AND app_name = %s AND config_type = %s
    """, (channel_id, app_name, config_type))
    res = cur.fetchone()
    
    json_data = json.dumps(config_data)
    
    if res:
        cur.execute("""
            UPDATE channel_config 
            SET config_data = %s, updated_at = NOW()
            WHERE id = %s
        """, (json_data, res[0]))
    else:
        cur.execute("""
            INSERT INTO channel_config (channel_id, app_name, config_type, config_data, created_at, updated_at)
            VALUES (%s, %s, %s, %s, NOW(), NOW())
        """, (channel_id, app_name, config_type, json_data))
    
    print(f"  Saved {app_name}.{config_type} for {channel_id}")


if __name__ == "__main__":
    reimport_channels()

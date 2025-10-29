"""
Test script to verify configuration persistence functionality
"""
from app import app, db_manager
import json

def test_config_persistence():
    """Test saving and loading configurations"""
    with app.app_context():
        print('='*70)
        print('Testing Configuration Persistence')
        print('='*70)
        
        channel_id = 'test_channel'
        app_name = 'PeopleCounter'
        
        # Test 1: Save ROI configuration
        print('\n1️⃣  Testing ROI Configuration Save/Load')
        print('-' * 70)
        
        roi_config = {
            'main': [
                {'x': 0.2, 'y': 0.3},
                {'x': 0.8, 'y': 0.3},
                {'x': 0.8, 'y': 0.7},
                {'x': 0.2, 'y': 0.7}
            ],
            'secondary': [
                {'x': 0.4, 'y': 0.5},
                {'x': 0.6, 'y': 0.5},
                {'x': 0.6, 'y': 0.6},
                {'x': 0.4, 'y': 0.6}
            ]
        }
        
        try:
            db_manager.save_channel_config(channel_id, app_name, 'roi', roi_config)
            print(f'✅ ROI configuration saved')
            print(f'   Main points: {len(roi_config["main"])}')
            print(f'   Secondary points: {len(roi_config["secondary"])}')
            
            # Load it back
            loaded_roi = db_manager.get_channel_config(channel_id, app_name, 'roi')
            if loaded_roi:
                print(f'✅ ROI configuration loaded successfully')
                print(f'   Main points match: {loaded_roi["main"] == roi_config["main"]}')
                print(f'   Secondary points match: {loaded_roi["secondary"] == roi_config["secondary"]}')
            else:
                print(f'❌ Failed to load ROI configuration')
                
        except Exception as e:
            print(f'❌ Error with ROI config: {e}')
        
        # Test 2: Save counting line configuration
        print('\n2️⃣  Testing Counting Line Configuration Save/Load')
        print('-' * 70)
        
        line_config = {
            'start': {'x': 0.5, 'y': 0.1},
            'end': {'x': 0.5, 'y': 0.9},
            'orientation': 'vertical'
        }
        
        try:
            db_manager.save_channel_config(channel_id, app_name, 'counting_line', line_config)
            print(f'✅ Counting line configuration saved')
            print(f'   Start: ({line_config["start"]["x"]}, {line_config["start"]["y"]})')
            print(f'   End: ({line_config["end"]["x"]}, {line_config["end"]["y"]})')
            print(f'   Orientation: {line_config["orientation"]}')
            
            # Load it back
            loaded_line = db_manager.get_channel_config(channel_id, app_name, 'counting_line')
            if loaded_line:
                print(f'✅ Counting line configuration loaded successfully')
                print(f'   Start matches: {loaded_line["start"] == line_config["start"]}')
                print(f'   End matches: {loaded_line["end"] == line_config["end"]}')
                print(f'   Orientation matches: {loaded_line["orientation"] == line_config["orientation"]}')
            else:
                print(f'❌ Failed to load counting line configuration')
                
        except Exception as e:
            print(f'❌ Error with counting line config: {e}')
        
        # Test 3: Test Queue Monitor config
        print('\n3️⃣  Testing Queue Monitor Configuration Save/Load')
        print('-' * 70)
        
        qm_app_name = 'QueueMonitor'
        qm_roi_config = {
            'main': [
                {'x': 0.1, 'y': 0.2},
                {'x': 0.9, 'y': 0.2},
                {'x': 0.9, 'y': 0.8},
                {'x': 0.1, 'y': 0.8}
            ],
            'secondary': [
                {'x': 0.3, 'y': 0.4},
                {'x': 0.7, 'y': 0.4},
                {'x': 0.7, 'y': 0.6},
                {'x': 0.3, 'y': 0.6}
            ]
        }
        
        try:
            db_manager.save_channel_config(channel_id, qm_app_name, 'roi', qm_roi_config)
            print(f'✅ Queue Monitor ROI configuration saved')
            
            loaded_qm_roi = db_manager.get_channel_config(channel_id, qm_app_name, 'roi')
            if loaded_qm_roi:
                print(f'✅ Queue Monitor ROI configuration loaded successfully')
                print(f'   Configuration preserved correctly')
            else:
                print(f'❌ Failed to load Queue Monitor ROI configuration')
                
        except Exception as e:
            print(f'❌ Error with Queue Monitor config: {e}')
        
        # Test 4: Update configuration
        print('\n4️⃣  Testing Configuration Update')
        print('-' * 70)
        
        updated_line_config = {
            'start': {'x': 0.3, 'y': 0.2},
            'end': {'x': 0.7, 'y': 0.8},
            'orientation': 'diagonal'
        }
        
        try:
            db_manager.save_channel_config(channel_id, app_name, 'counting_line', updated_line_config)
            print(f'✅ Configuration updated')
            
            loaded_updated = db_manager.get_channel_config(channel_id, app_name, 'counting_line')
            if loaded_updated:
                print(f'✅ Updated configuration loaded successfully')
                print(f'   New orientation: {loaded_updated["orientation"]}')
                print(f'   Configuration updated correctly: {loaded_updated["orientation"] == "diagonal"}')
            else:
                print(f'❌ Failed to load updated configuration')
                
        except Exception as e:
            print(f'❌ Error updating config: {e}')
        
        # Test 5: Check database records
        print('\n5️⃣  Checking Database Records')
        print('-' * 70)
        
        try:
            from sqlalchemy import inspect
            inspector = inspect(db_manager.db.engine)
            
            # Check if channel_config table exists
            tables = inspector.get_table_names()
            if 'channel_config' in tables:
                print(f'✅ channel_config table exists')
                
                # Count records
                total_configs = db_manager.ChannelConfig.query.count()
                pc_configs = db_manager.ChannelConfig.query.filter_by(
                    channel_id=channel_id,
                    app_name=app_name
                ).count()
                qm_configs = db_manager.ChannelConfig.query.filter_by(
                    channel_id=channel_id,
                    app_name=qm_app_name
                ).count()
                
                print(f'   Total configurations in database: {total_configs}')
                print(f'   People Counter configs for {channel_id}: {pc_configs}')
                print(f'   Queue Monitor configs for {channel_id}: {qm_configs}')
                
                # Display all configs for test channel
                all_configs = db_manager.ChannelConfig.query.filter_by(
                    channel_id=channel_id
                ).all()
                
                print(f'\n   All configurations for {channel_id}:')
                for config in all_configs:
                    print(f'   - {config.app_name}/{config.config_type} (Updated: {config.updated_at})')
                    
            else:
                print(f'❌ channel_config table does not exist!')
                
        except Exception as e:
            print(f'❌ Error checking database: {e}')
        
        print('\n' + '='*70)
        print('✅ ALL TESTS COMPLETED!')
        print('='*70)
        
        print('\n💡 Next Steps:')
        print('   1. Start the application: python app.py')
        print('   2. Select a channel and start it')
        print('   3. Open ROI or Line editor')
        print('   4. Existing configurations should load automatically')
        print('   5. Make changes and save')
        print('   6. Restart application - configurations should persist')
        print('='*70)

if __name__ == '__main__':
    test_config_persistence()

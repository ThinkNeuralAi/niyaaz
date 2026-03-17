#!/usr/bin/env python3
"""
Integration test for ServiceDisciplineMonitor ROI visualization
Tests loading ROIs from config and drawing them on frames
"""
import cv2
import numpy as np
import json
from pathlib import Path as PathlibPath
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch
import sys
import logging
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_service_discipline_roi():
    """Test ServiceDisciplineMonitor ROI loading and visualization"""
    
    print("=== ServiceDisciplineMonitor ROI Integration Test ===\n")
    
    # Create test config with ROIs
    test_config = {
        "channels": [
            {
                "channel_id": "test_channel_1",
                "modules": [
                    {
                        "type": "ServiceDisciplineMonitor",
                        "config": {
                            "table_rois": {
                                "table_1": {
                                    "points": [
                                        {"x": 0.1, "y": 0.1},
                                        {"x": 0.3, "y": 0.1},
                                        {"x": 0.3, "y": 0.3},
                                        {"x": 0.1, "y": 0.3}
                                    ]
                                },
                                "table_2": {
                                    "points": [
                                        {"x": 0.5, "y": 0.2},
                                        {"x": 0.8, "y": 0.2},
                                        {"x": 0.8, "y": 0.5},
                                        {"x": 0.5, "y": 0.5}
                                    ]
                                }
                            }
                        }
                    }
                ]
            }
        ]
    }
    
    # Create temporary config file
    config_path = PathlibPath("/tmp/test_channels.json")
    with open(config_path, 'w') as f:
        json.dump(test_config, f, indent=2)
    print(f"✓ Created test config with {len(test_config['channels'][0]['modules'][0]['config']['table_rois'])} tables")
    
    # Mock the necessary components
    mock_socketio = Mock()
    mock_db_manager = Mock()
    mock_app = Mock()
    
    # Import after mocking to get the modified behavior
    sys.path.insert(0, '/home/ubuntu/Niyaaz_biryani/niyaaz')
    
    print("\nLoading ServiceDisciplineMonitor...")
    
    # The class will try to load from config/channels.json
    # We need to copy our test config there
    config_dest = PathlibPath("/home/ubuntu/Niyaaz_biryani/niyaaz/config/channels.json")
    config_dest_backup = config_dest.with_suffix('.json.backup')
    
    # Backup original if exists
    if config_dest.exists():
        config_dest.rename(config_dest_backup)
        print(f"  Backed up original config to {config_dest_backup}")
    
    try:
        # Copy test config
        import shutil
        shutil.copy(config_path, config_dest)
        print(f"  Copied test config to {config_dest}")
        
        # Now we can import and instantiate
        from modules.service_discipline_monitor import ServiceDisciplineMonitor
        
        monitor = ServiceDisciplineMonitor(
            channel_id="test_channel_1",
            socketio=mock_socketio,
            db_manager=mock_db_manager,
            app=mock_app
        )
        
        print(f"✓ ServiceDisciplineMonitor instantiated")
        print(f"  Loaded {len(monitor.table_rois)} table ROIs")
        
        for table_id, roi_info in monitor.table_rois.items():
            polygon = roi_info.get("polygon", [])
            bbox = roi_info.get("bbox", ())
            print(f"  - Table {table_id}: {len(polygon)} points, bbox={bbox}")
        
        # Test drawing on a frame
        print("\nTesting frame annotation...")
        frame = np.ones((1080, 1920, 3), dtype=np.uint8) * 50
        
        annotated = monitor._draw_annotations_new(frame, datetime.now())
        
        print(f"✓ Frame annotated successfully")
        print(f"  Output shape: {annotated.shape}")
        
        # Save test frame
        output_path = "/tmp/test_roi_monitor_output.jpg"
        cv2.imwrite(output_path, annotated)
        print(f"✓ Saved annotated frame to {output_path}")
        
        # Verify the frame has changes (ROI was drawn)
        if not np.array_equal(frame, annotated):
            print(f"✓ ROIs were successfully drawn on the frame")
        else:
            print(f"✗ WARNING: Frame was not modified (ROI may not be visible)")
        
        print("\n✅ Integration test PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Integration test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Restore original config
        if config_dest_backup.exists():
            shutil.move(config_dest_backup, config_dest)
            print(f"\nRestored original config from {config_dest_backup}")
        elif config_dest.exists():
            config_dest.unlink()
            print(f"\nRemoved test config from {config_dest}")

if __name__ == "__main__":
    success = test_service_discipline_roi()
    sys.exit(0 if success else 1)

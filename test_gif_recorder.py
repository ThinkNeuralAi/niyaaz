#!/usr/bin/env python3
"""
Test script for GIF recorder functionality
"""
import sys
import os
sys.path.append('/home/ajmal_tnai/sakshiai')

import cv2
import numpy as np
from modules.gif_recorder import AlertGifRecorder
from datetime import datetime
import time

def create_test_frames(count=60, width=480, height=360):
    """Create test frames for GIF testing"""
    frames = []
    for i in range(count):
        # Create a simple test frame with moving circle
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add gradient background
        for y in range(height):
            frame[y, :] = [int(255 * y / height), 50, 100]
        
        # Add moving circle
        center_x = int(width * 0.1 + (width * 0.8) * (i / count))
        center_y = int(height / 2)
        cv2.circle(frame, (center_x, center_y), 30, (0, 255, 255), -1)
        
        # Add frame number
        cv2.putText(frame, f"Frame {i+1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Alert Test", (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        frames.append(frame)
    
    return frames

def test_gif_recorder():
    """Test the GIF recorder functionality"""
    print("🧪 Testing GIF Recorder...")
    
    # Initialize recorder
    recorder = AlertGifRecorder(
        buffer_size=30,  # Small buffer for testing
        gif_duration=2.0,  # 2 second GIFs
        fps=15  # Lower FPS for testing
    )
    
    # Create test frames
    print("📹 Creating test frames...")
    test_frames = create_test_frames(60, 480, 360)
    
    # Add frames to buffer
    print("📦 Adding frames to buffer...")
    for i, frame in enumerate(test_frames[:40]):  # Add first 40 frames to buffer
        recorder.add_frame(frame)
        time.sleep(0.02)  # Simulate real-time
    
    # Simulate alert trigger
    print("🚨 Simulating alert trigger...")
    alert_info = {
        'type': 'test_alert',
        'message': 'Test alert for GIF recording',
        'timestamp': datetime.now().isoformat()
    }
    
    recorder.start_alert_recording(alert_info)
    
    # Add more frames during alert
    print("🎬 Adding alert frames...")
    for frame in test_frames[40:]:  # Add remaining frames during alert
        recorder.add_alert_frame(frame)
        time.sleep(0.02)
    
    # Create manual GIF
    print("💾 Creating test GIF...")
    gif_result = recorder.create_manual_gif(
        test_frames[30:60],  # Use frames 30-60 for GIF
        alert_info,
        'test_alert.gif'
    )
    
    if gif_result:
        print(f"✅ GIF created successfully!")
        print(f"   📁 File: {gif_result['gif_filename']}")
        print(f"   📏 Frames: {gif_result['frame_count']}")
        print(f"   📍 Path: {gif_result['gif_path']}")
        
        # Check if file exists
        if os.path.exists(gif_result['gif_path']):
            file_size = os.path.getsize(gif_result['gif_path'])
            print(f"   💾 Size: {file_size} bytes ({file_size / 1024:.1f} KB)")
        else:
            print("   ❌ File not found!")
            return False
    else:
        print("❌ Failed to create GIF!")
        return False
    
    # Test stats
    stats = recorder.get_stats()
    print("\n📊 Recorder Stats:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Test recent GIFs
    recent_gifs = recorder.get_recent_gifs(5)
    print(f"\n📋 Recent GIFs: {len(recent_gifs)} found")
    for gif in recent_gifs:
        print(f"   📄 {gif['filename']} - {gif['size']} bytes")
    
    print("\n✅ GIF Recorder test completed successfully!")
    return True

if __name__ == "__main__":
    try:
        success = test_gif_recorder()
        if success:
            print("\n🎉 All tests passed!")
            sys.exit(0)
        else:
            print("\n❌ Tests failed!")
            sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
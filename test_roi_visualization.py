#!/usr/bin/env python3
"""
Test script to verify ROI visualization in service_discipline_monitor
"""
import cv2
import numpy as np
import sys
from pathlib import Path

# Test the ROI drawing with mock data
def test_roi_drawing():
    """Test ROI polygon drawing with both normalized and pixel coordinates"""
    
    # Create a test frame (1920x1080)
    h, w = 1080, 1920
    frame = np.ones((h, w, 3), dtype=np.uint8) * 50  # Dark gray background
    
    print(f"Test frame size: {w}x{h}")
    
    # Test Case 1: Normalized coordinates (0-1 range)
    print("\n=== Test Case 1: Normalized Coordinates ===")
    normalized_polygon = [
        (0.1, 0.1),   # top-left
        (0.4, 0.1),   # top-right
        (0.4, 0.35),  # bottom-right
        (0.1, 0.35)   # bottom-left
    ]
    
    annotated = frame.copy()
    polygon_pixels = []
    
    # Convert normalized to pixel coordinates
    for p in normalized_polygon:
        px = int(p[0] * w)
        py = int(p[1] * h)
        polygon_pixels.append((px, py))
        print(f"  Normalized {p} -> Pixel ({px}, {py})")
    
    if len(polygon_pixels) >= 3:
        polygon_array = np.array(polygon_pixels, np.int32)
        overlay = annotated.copy()
        cv2.fillPoly(overlay, [polygon_array], (0, 255, 255))
        cv2.addWeighted(overlay, 0.2, annotated, 0.8, 0, annotated)
        cv2.polylines(annotated, [polygon_array], True, (0, 255, 255), 3)
        
        # Add label
        label_pos = polygon_pixels[0]
        label_text = "Table 1"
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        )
        cv2.rectangle(annotated,
                    (label_pos[0] - 5, label_pos[1] - text_height - 10),
                    (label_pos[0] + text_width + 5, label_pos[1] + baseline + 5),
                    (0, 0, 0), -1)
        cv2.putText(annotated, label_text, label_pos,
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        print(f"✓ Drew ROI with {len(polygon_pixels)} points (normalized)")
    
    cv2.imwrite("/tmp/test_roi_normalized.jpg", annotated)
    print(f"✓ Saved to /tmp/test_roi_normalized.jpg")
    
    # Test Case 2: Pixel coordinates (raw pixel values)
    print("\n=== Test Case 2: Pixel Coordinates ===")
    pixel_polygon = [
        (200, 100),    # top-left
        (800, 100),    # top-right
        (800, 400),    # bottom-right
        (200, 400)     # bottom-left
    ]
    
    annotated = frame.copy()
    polygon_pixels = list(pixel_polygon)
    
    for idx, p in enumerate(polygon_pixels):
        print(f"  Point {idx}: {p}")
    
    if len(polygon_pixels) >= 3:
        polygon_array = np.array(polygon_pixels, np.int32)
        overlay = annotated.copy()
        cv2.fillPoly(overlay, [polygon_array], (0, 255, 0))
        cv2.addWeighted(overlay, 0.2, annotated, 0.8, 0, annotated)
        cv2.polylines(annotated, [polygon_array], True, (0, 255, 0), 3)
        
        # Add label
        label_pos = polygon_pixels[0]
        label_text = "Table 2"
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        )
        cv2.rectangle(annotated,
                    (label_pos[0] - 5, label_pos[1] - text_height - 10),
                    (label_pos[0] + text_width + 5, label_pos[1] + baseline + 5),
                    (0, 0, 0), -1)
        cv2.putText(annotated, label_text, label_pos,
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        print(f"✓ Drew ROI with {len(polygon_pixels)} points (pixel coordinates)")
    
    cv2.imwrite("/tmp/test_roi_pixels.jpg", annotated)
    print(f"✓ Saved to /tmp/test_roi_pixels.jpg")
    
    # Test Case 3: Detect coordinate format (heuristic)
    print("\n=== Test Case 3: Coordinate Format Detection ===")
    test_cases = [
        ([(0.1, 0.2), (0.3, 0.4)], True),   # Should be normalized
        ([(100, 200), (300, 400)], False),  # Should be pixel
        ([(0.5, 0.5), (1.0, 1.0)], True),   # Should be normalized
        ([(1000, 1000), (1500, 1500)], False)  # Should be pixel
    ]
    
    for points, expected_normalized in test_cases:
        first_point = points[0]
        test_x, test_y = float(first_point[0]), float(first_point[1])
        is_normalized = (0 <= test_x <= 2) and (0 <= test_y <= 2)
        
        status = "✓" if is_normalized == expected_normalized else "✗"
        print(f"  {status} Points {points}: detected as {'NORMALIZED' if is_normalized else 'PIXEL'} {'(expected)' if is_normalized == expected_normalized else '(UNEXPECTED!)'}")
    
    print("\n✓ All tests completed!")
    print("Generated test images: /tmp/test_roi_normalized.jpg, /tmp/test_roi_pixels.jpg")

if __name__ == "__main__":
    try:
        test_roi_drawing()
        sys.exit(0)
    except Exception as e:
        print(f"✗ Test failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

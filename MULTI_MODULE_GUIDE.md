# 🎯 Multi-Module Video Analysis Guide

## 📋 Overview

Sakshi.AI now supports **multiple analysis modules on the same video feed**, allowing you to run People Counter and Queue Monitor simultaneously on a single camera stream. This provides comprehensive analytics while optimizing system resources.

## 🌟 Key Benefits

### ✅ **Resource Efficiency**
- **Single video processing pipeline** for multiple analyses
- **Reduced CPU/GPU usage** compared to separate streams
- **Shared YOLO detection** across all modules
- **Optimized memory management**

### ✅ **Comprehensive Analytics** 
- **People counting** + **Queue monitoring** on same feed
- **Unified dashboard** showing all metrics
- **Correlated data** from same video source
- **Real-time updates** for all modules

### ✅ **Easy Management**
- **Start/stop modules independently** 
- **Shared configuration** per channel
- **Visual indicators** for shared channels
- **Combined status reporting**

## 🚀 How It Works

### **Architecture Overview**
```
Video Feed → Multi-Module Processor → [People Counter] + [Queue Monitor]
                     ↓                        ↓              ↓
              Combined Frame Display    Footfall Data   Queue Analytics
```

### **Shared Processing Pipeline**
1. **Single video capture** from source
2. **One YOLO detection** run per frame
3. **Multiple modules** process the same detections
4. **Combined visual annotations** in dashboard
5. **Independent data storage** per module

## 📱 Using Multi-Module Analysis

### **Step 1: Start First Module**
1. Go to **People Counter** tab
2. Select a video file (e.g., `shop_entrance.mp4`)
3. Click **"Start People Counting"**
4. System creates shared video processor

### **Step 2: Add Second Module**
1. Go to **Queue Monitor** tab  
2. Select the **same video file** (`shop_entrance.mp4`)
3. Click **"Start Queue Monitoring"**
4. System adds Queue Monitor to existing processor

### **Step 3: View Combined Results**
- Both modules now analyze the same video
- **Shared indicator** shows active modules
- **"Show All Analytics"** button displays combined status
- **Independent configuration** for each module

## 🎛️ Dashboard Features

### **Visual Indicators**
```
Channel shop_entrance - People Counter [Live] [Shared: PeopleCounter, QueueMonitor]
```

### **Combined Stream Display**
- **Header shows** active modules
- **People Counter metrics**: IN/OUT counts with green annotations
- **Queue Monitor metrics**: Queue/Counter counts with colored areas
- **Unified video feed** with all annotations

### **Independent Controls**
- **Edit Counting Line** (People Counter specific)
- **Configure Areas** (Queue Monitor specific)  
- **Show All Analytics** (Combined status)
- **Stop modules independently**

## ⚙️ Configuration Examples

### **Example 1: Retail Store Entrance**
**Use Case**: Monitor both footfall and queue at checkout
```javascript
// Start People Counter for entrance monitoring
startAnalysisModule('PeopleCounter');

// Add Queue Monitor for checkout area
startAnalysisModule('QueueMonitor');

// Configure counting line for entrance
// Configure ROI areas for checkout queue
```

### **Example 2: Office Reception**
**Use Case**: Track visitor counts and reception desk queue
```javascript
// Same video feed analyzes:
// - Visitor IN/OUT counts
// - Queue at reception desk
// - Alert when reception is understaffed
```

### **Example 3: Restaurant**
**Use Case**: Monitor customer flow and waiting area
```javascript
// Combined analysis provides:
// - Customer entry/exit trends
// - Waiting area congestion
// - Host station coverage alerts
```

## 🔧 Technical Implementation

### **Shared Video Processor**
```python
# Single processor handles multiple modules
processor = MultiModuleVideoProcessor(video_path, channel_id)

# Add modules dynamically
processor.add_module('PeopleCounter', people_counter_instance)
processor.add_module('QueueMonitor', queue_monitor_instance)

# Combined frame processing
combined_frame = processor.process_frame(frame)
```

### **Module Independence**
- Each module maintains its own configuration
- Independent database storage
- Separate WebSocket events
- Module-specific API endpoints

### **Resource Optimization**
- Single YOLO inference per frame
- Shared person detection results
- Combined visual annotations
- Optimized memory usage

## 📊 API Endpoints

### **Start Shared Analysis**
```http
POST /api/start_channel
{
  "app_name": "PeopleCounter",
  "channel_id": "shop_entrance", 
  "video_path": "videos/shop_entrance.mp4"
}

Response: {
  "success": true,
  "shared": false,
  "active_modules": ["PeopleCounter"]
}
```

### **Add Module to Shared Channel**
```http
POST /api/start_channel  
{
  "app_name": "QueueMonitor",
  "channel_id": "shop_entrance",
  "video_path": "videos/shop_entrance.mp4"
}

Response: {
  "success": true,
  "shared": true,
  "active_modules": ["PeopleCounter", "QueueMonitor"]
}
```

### **Get Combined Status**
```http
GET /api/get_channel_status/shop_entrance

Response: {
  "channel_id": "shop_entrance",
  "active_modules": ["PeopleCounter", "QueueMonitor"],
  "frames_processed": 1500,
  "average_fps": 28.5,
  "module_info": {
    "PeopleCounter": {
      "in_count": 45,
      "out_count": 38,
      "net_count": 7
    },
    "QueueMonitor": {
      "queue_count": 3,
      "counter_count": 2,
      "roi_configured": true
    }
  }
}
```

## 🎯 Best Practices

### **Optimal Use Cases**
✅ **Same camera angle** needs multiple analyses  
✅ **Correlated metrics** (footfall + queue)  
✅ **Resource optimization** requirements  
✅ **Unified monitoring** dashboard  

### **When to Use Separate Channels**  
❌ **Different camera locations**  
❌ **Completely unrelated analyses**  
❌ **Different video quality requirements**  
❌ **Independent processing needs**

### **Performance Tips**
- **Start with one module**, add others incrementally
- **Configure both modules** before heavy analysis periods  
- **Monitor system resources** when running multiple modules
- **Use appropriate video resolution** for processing capability

## 🔮 Advanced Features

### **Module Priority**
- Configure which module's annotations show prominently
- Adjust processing order for optimal performance

### **Conditional Analysis**
- Enable/disable modules based on time of day
- Automatic module switching based on detection patterns

### **Enhanced Reporting**
- Cross-module correlation reports
- Combined analytics dashboard
- Unified alert system

---

**🎉 Result**: One video feed now provides comprehensive analytics with People Counter tracking footfall while Queue Monitor manages service efficiency - all optimized for maximum performance and minimum resource usage!

## 🛠️ Quick Test

1. **Add a video file**: `cp your_video.mp4 videos/test_video.mp4`
2. **Start People Counter** on `test_video`
3. **Start Queue Monitor** on same `test_video` 
4. **See shared processing** in action with combined analytics!

The system automatically detects when you're using the same video for multiple modules and optimizes accordingly. 🚀
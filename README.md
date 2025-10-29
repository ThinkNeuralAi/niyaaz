# Sakshi.AI - Intelligent Video Analytics Platform

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-2.3.3-green.svg)](https://flask.palletsprojects.com/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange.svg)](https://github.com/ultralytics/ultralytics)

A comprehensive video analytics platform that transforms any CCTV camera into an intelligent monitoring system. Sakshi.AI provides real-time people counting, queue monitoring, and advanced analytics with a modern web dashboard.

## 🎯 Features

### People Counter
- **Real-time Footfall Tracking**: Count people entering and exiting with high accuracy
- **Direction Detection**: Distinguish between IN and OUT traffic
- **Configurable Counting Line**: Adjustable vertical/horizontal detection line
- **Historical Analytics**: Daily, weekly, and monthly footfall reports
- **Peak Hours Analysis**: Identify busiest periods for resource planning

### Queue Monitor
- **Smart Queue Detection**: Monitor queue length in designated areas
- **ROI-based Analysis**: Define custom queue and counter areas
- **Dwell Time Filtering**: Count only people waiting for minimum time
- **Staffing Alerts**: Get notified when queues are long but counters understaffed
- **Real-time Monitoring**: Live queue status and historical trends

### Technical Features
- **YOLOv8 Person Detection**: State-of-the-art AI for accurate person detection
- **Real-time Streaming**: Live video feeds with AI annotations
- **WebSocket Communication**: Instant updates without page refresh
- **Multi-channel Support**: Monitor multiple camera feeds simultaneously
- **Database Analytics**: Comprehensive data storage and reporting
- **Responsive Dashboard**: Modern web interface for all devices

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- Git
- 4GB+ RAM recommended
- CUDA-compatible GPU (optional, for better performance)

### Quick Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd sakshiai
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Create video directory and add your video files**
   ```bash
   mkdir -p videos
   # Copy your .mp4 video files to the videos/ directory
   ```

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Access the dashboard**
   - Open your browser and go to: `http://localhost:5000`
   - Click "Launch Dashboard" to access the analytics interface

## 📁 Project Structure

```
sakshiai/
├── app.py                  # Main Flask application
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── videos/                # Place your video files here
├── data/                  # Database storage
├── templates/             # HTML templates
│   ├── landing.html       # Landing page
│   └── dashboard.html     # Analytics dashboard
├── modules/               # Core AI modules
│   ├── database.py        # Database models and management
│   ├── yolo_detector.py   # YOLO person detection
│   ├── people_counter.py  # People counting logic
│   ├── queue_monitor.py   # Queue monitoring logic
│   └── video_processor.py # Video processing pipeline
└── config/               # Configuration files
```

## 🚀 Usage

### Setting Up People Counter

1. **Start the application** and navigate to the dashboard
2. **Select a video file** from the People Counter dropdown
3. **Click "Start Monitoring"** to begin real-time analysis
4. **Configure counting line** by clicking "Edit Counting Line"
   - Choose vertical or horizontal orientation
   - Drag to adjust position
   - Save configuration

### Setting Up Queue Monitor

1. **Select a video file** from the Queue Monitor dropdown
2. **Click "Start Monitoring"** to begin analysis
3. **Configure areas** by clicking "Configure Areas"
   - Draw yellow polygon for queue area
   - Draw cyan polygon for counter area
   - Save configuration

### Viewing Analytics

- **Real-time counts** are displayed below each video feed
- **Historical reports** available in the Analytics & Reports tabs
- **Alerts** appear automatically when queue thresholds are exceeded

## ⚙️ Configuration

### Video Sources
- Place your video files in the `videos/` directory
- Supported formats: MP4, AVI, MOV
- For live cameras, modify the video source in the channel configuration

### Detection Settings
- **Confidence threshold**: Adjust in `yolo_detector.py` (default: 0.5-0.6)
- **Tracking distance**: Modify in person tracker settings
- **Dwell time**: Configure minimum wait time for queue counting

### Database
- Default: SQLite database in `data/sakshi.db`
- For production, configure PostgreSQL or MySQL in `app.py`

## 🔧 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Landing page |
| `/dashboard` | GET | Analytics dashboard |
| `/video_feed/<app>/<channel>` | GET | Live video stream |
| `/api/start_channel` | POST | Start monitoring a channel |
| `/api/stop_channel` | POST | Stop monitoring a channel |
| `/api/set_roi` | POST | Configure queue areas |
| `/api/set_counting_line` | POST | Configure counting line |
| `/api/get_footfall_report/<channel>` | GET | Footfall analytics |
| `/api/get_queue_report/<channel>` | GET | Queue analytics |

## 📊 Database Schema

### Tables
- **daily_footfall**: Daily in/out counts per channel
- **hourly_footfall**: Hourly breakdown of footfall data
- **queue_analytics**: Queue monitoring data and alerts
- **channel_config**: ROI and line configurations
- **detection_events**: Audit trail of all detections

## 🚨 Troubleshooting

### Common Issues

1. **"Cannot open video source" error**
   - Ensure video files are in the `videos/` directory
   - Check file permissions and format compatibility

2. **High CPU usage**
   - Reduce video resolution in `video_processor.py`
   - Lower the processing FPS limit
   - Consider using GPU acceleration

3. **Socket.IO connection issues**
   - Check firewall settings
   - Ensure port 5000 is not blocked
   - Try running on different port in `app.py`

4. **YOLO model download fails**
   - Check internet connection
   - Manually download YOLOv8 model to local directory
   - Update model path in `yolo_detector.py`

### Performance Optimization

1. **For better performance:**
   - Use GPU acceleration (install CUDA + PyTorch GPU version)
   - Reduce video resolution before processing
   - Adjust confidence thresholds to reduce false positives

2. **For production deployment:**
   - Use PostgreSQL or MySQL instead of SQLite
   - Set up Redis for session management
   - Use nginx for serving static files
   - Consider Docker deployment

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🏢 About

**Sakshi.AI** is developed by **ThinkNeural.AI** - a cutting-edge AI research and development company focused on computer vision and intelligent systems.

- Website: [ThinkNeural.AI](https://thinkneural.ai)
- Email: contact@thinkneural.ai

---

**Sakshi** (Sanskrit: साक्षी) means "witness" - representing our mission to create an intelligent digital witness for your physical spaces.

## 🔮 Roadmap

- [ ] Real-time heatmap analysis
- [ ] Facial recognition integration
- [ ] Mobile app for alerts
- [ ] Cloud deployment options
- [ ] Advanced behavioral analytics
- [ ] Integration with existing security systems

---

For support and questions, please open an issue or contact our team.
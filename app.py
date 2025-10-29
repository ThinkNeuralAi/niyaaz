"""
Sakshi.AI - Intelligent Video Analytics Platform
Main Flask Application
"""
import os
import json
import logging
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, Response, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_socketio import SocketIO, emit
import cv2
import threading
import time
from pathlib import Path

# Import custom modules
from modules.people_counter import PeopleCounter
from modules.queue_monitor import QueueMonitor
from modules.bag_detection import BagDetection
from modules.heatmap_processor import HeatmapProcessor
from modules.video_processor import VideoProcessor
from modules.multi_module_processor import MultiModuleVideoProcessor
from modules.database import DatabaseManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'sakshi-ai-secret-key-2025'

# Database configuration
DATABASE_URL = os.environ.get('DATABASE_URL', "postgresql+psycopg2://postgres:ajmal123@localhost:5432/sakshiai")
app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db = SQLAlchemy(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global variables - Updated for multi-module support
shared_video_processors = {}  # {channel_id: MultiModuleVideoProcessor}
channel_modules = {}  # {channel_id: {module_name: module_instance}}
app_configs = {
    'PeopleCounter': {
        'name': 'People Counter',
        'description': 'Count people entering and exiting',
        'channels': {},
        'status': 'online'
    },
    'QueueMonitor': {
        'name': 'Queue Monitor', 
        'description': 'Monitor queue length and wait times',
        'channels': {},
        'status': 'online'
    },
    'BagDetection': {
        'name': 'Bag Detection',
        'description': 'Detect unattended baggage with alerts',
        'channels': {},
        'status': 'online'
    },
    'Heatmap': {
        'name': 'Heatmap Analytics',
        'description': 'Visualize crowd density and hotspots',
        'channels': {},
        'status': 'online'
    }
}

# Database manager
db_manager = DatabaseManager(db)

@app.route('/')
def landing():
    """Landing page"""
    return render_template('landing.html')

@app.route('/dashboard')
def dashboard():
    """Main dashboard"""
    return render_template('dashboard.html', app_configs=app_configs)

@app.route('/static/alerts/<filename>')
def serve_alert_gif(filename):
    """Serve alert GIF files"""
    return send_from_directory('static/alerts', filename)

@app.route('/video_feed/<app_name>/<channel_id>')
def video_feed(app_name, channel_id):
    """Video streaming endpoint - supports both shared and module-specific feeds"""
    def generate():
        if channel_id in shared_video_processors:
            processor = shared_video_processors[channel_id]
            while processor.is_running:
                # Get frame - either combined or module-specific
                frame = processor.get_latest_frame(module_name=app_name if app_name in processor.get_active_modules() else None)
                if frame is not None:
                    ret, buffer = cv2.imencode('.jpg', frame)
                    if ret:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                time.sleep(0.1)
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/get_channels/<app_name>')
def get_channels(app_name):
    """Get available channels for an app"""
    channels = []
    
    # Get saved RTSP channels from database
    try:
        saved_channels = db_manager.get_rtsp_channels()
        for channel in saved_channels:
            channels.append({
                'id': channel['channel_id'],
                'name': channel['name'],
                'rtsp_url': channel['rtsp_url'],
                'type': 'rtsp'
            })
    except Exception as e:
        logger.error(f"Error loading RTSP channels: {e}")
    
    # Also check for video files (backward compatibility)
    videos_dir = Path('videos')
    if videos_dir.exists():
        for video_file in videos_dir.glob('*.mp4'):
            channel_id = video_file.stem
            channels.append({
                'id': channel_id,
                'name': f"Video {channel_id}",
                'path': str(video_file),
                'type': 'video'
            })
    
    return jsonify(channels)

@app.route('/api/add_rtsp_channel', methods=['POST'])
def add_rtsp_channel():
    """Add a new RTSP channel"""
    data = request.json
    channel_name = data.get('name')
    rtsp_url = data.get('rtsp_url')
    
    if not channel_name or not rtsp_url:
        return jsonify({'success': False, 'error': 'Name and RTSP URL are required'})
    
    try:
        # Generate channel ID from name
        channel_id = channel_name.lower().replace(' ', '_').replace('-', '_')
        
        # Save to database
        success = db_manager.save_rtsp_channel(channel_id, channel_name, rtsp_url)
        
        if success:
            return jsonify({
                'success': True, 
                'channel_id': channel_id,
                'message': f'RTSP channel "{channel_name}" added successfully'
            })
        else:
            return jsonify({'success': False, 'error': 'Failed to save RTSP channel'})
    
    except Exception as e:
        logger.error(f"Error adding RTSP channel: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/remove_rtsp_channel', methods=['POST'])
def remove_rtsp_channel():
    """Remove an RTSP channel"""
    data = request.json
    channel_id = data.get('channel_id')
    
    if not channel_id:
        return jsonify({'success': False, 'error': 'Channel ID is required'})
    
    try:
        # Stop channel if running
        if channel_id in shared_video_processors:
            processor = shared_video_processors[channel_id]
            processor.stop()
            del shared_video_processors[channel_id]
            if channel_id in channel_modules:
                del channel_modules[channel_id]
        
        # Remove from database
        success = db_manager.remove_rtsp_channel(channel_id)
        
        if success:
            return jsonify({'success': True, 'message': 'RTSP channel removed successfully'})
        else:
            return jsonify({'success': False, 'error': 'Failed to remove RTSP channel'})
    
    except Exception as e:
        logger.error(f"Error removing RTSP channel: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/start_channel', methods=['POST'])
def start_channel():
    """Start processing a video channel - supports multiple modules on same video"""
    data = request.json
    app_name = data.get('app_name')
    channel_id = data.get('channel_id')
    
    # Determine video source (RTSP URL or video file path)
    rtsp_url = data.get('rtsp_url')
    video_path = data.get('video_path')
    
    if rtsp_url:
        video_source = rtsp_url
        source_type = 'rtsp'
    elif video_path:
        video_source = video_path
        source_type = 'video'
    else:
        # Try to get from database or fallback to video file
        try:
            rtsp_channel = db_manager.get_rtsp_channel(channel_id)
            if rtsp_channel:
                video_source = rtsp_channel['rtsp_url']
                source_type = 'rtsp'
            else:
                video_source = f'videos/{channel_id}.mp4'
                source_type = 'video'
        except:
            video_source = f'videos/{channel_id}.mp4'
            source_type = 'video'
    
    try:
        # Check if video processor already exists for this channel
        if channel_id not in shared_video_processors:
            # Create new multi-module processor
            processor = MultiModuleVideoProcessor(video_source, channel_id)
            shared_video_processors[channel_id] = processor
            channel_modules[channel_id] = {}
            
            # Start the processor
            if not processor.start():
                del shared_video_processors[channel_id]
                del channel_modules[channel_id]
                return jsonify({'success': False, 'error': f'Failed to start video processor for {source_type} source'})
        
        processor = shared_video_processors[channel_id]
        
        # Check if this module is already active for this channel
        if app_name in channel_modules[channel_id]:
            return jsonify({'success': True, 'message': f'{app_name} already active on channel {channel_id}'})
        
        # Create and add the analysis module
        if app_name == 'PeopleCounter':
            module = PeopleCounter(channel_id, socketio, db_manager, app)
        elif app_name == 'QueueMonitor':
            module = QueueMonitor(channel_id, socketio, db_manager, app)
        elif app_name == 'BagDetection':
            module = BagDetection(channel_id, socketio, db_manager, app)
        elif app_name == 'Heatmap':
            module = HeatmapProcessor(channel_id, socketio, db_manager, app)
        else:
            return jsonify({'success': False, 'error': 'Unknown app type'})
        
        # Add module to processor
        processor.add_module(app_name, module)
        channel_modules[channel_id][app_name] = module
        
        # Update config
        if channel_id not in app_configs[app_name]['channels']:
            app_configs[app_name]['channels'][channel_id] = {
                'name': f"Channel {channel_id}",
                'status': 'online',
                'video_source': video_source,
                'source_type': source_type,
                'shared': True,
                'active_modules': []
            }
        
        # Add this module to active modules list
        if app_name not in app_configs[app_name]['channels'][channel_id].get('active_modules', []):
            app_configs[app_name]['channels'][channel_id].setdefault('active_modules', []).append(app_name)
        
        # Update other app configs to show shared status
        for other_app in app_configs:
            if other_app != app_name and channel_id in channel_modules[channel_id]:
                if channel_id not in app_configs[other_app]['channels']:
                    app_configs[other_app]['channels'][channel_id] = {
                        'name': f"Channel {channel_id}",
                        'status': 'online',
                        'video_source': video_source,
                        'source_type': source_type,
                        'shared': True,
                        'active_modules': []
                    }
                # Update active modules for other apps
                current_modules = list(channel_modules[channel_id].keys())
                app_configs[other_app]['channels'][channel_id]['active_modules'] = current_modules
        
        logger.info(f"Started {app_name} on channel {channel_id} ({source_type}: {video_source})")
        return jsonify({
            'success': True, 
            'shared': True, 
            'active_modules': list(channel_modules[channel_id].keys()),
            'source_type': source_type,
            'video_source': video_source
        })
        
    except Exception as e:
        logger.error(f"Error starting channel: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/stop_channel', methods=['POST'])
def stop_channel():
    """Stop processing a video channel or remove a module from shared channel"""
    data = request.json
    app_name = data.get('app_name')
    channel_id = data.get('channel_id')
    
    try:
        if channel_id in shared_video_processors:
            processor = shared_video_processors[channel_id]
            
            # Remove the specific module
            if app_name in channel_modules.get(channel_id, {}):
                processor.remove_module(app_name)
                del channel_modules[channel_id][app_name]
                
                # Update app config
                if channel_id in app_configs[app_name]['channels']:
                    app_configs[app_name]['channels'][channel_id]['status'] = 'offline'
                
                # If no modules left, stop the entire processor
                if not channel_modules[channel_id]:
                    processor.stop()
                    del shared_video_processors[channel_id]
                    del channel_modules[channel_id]
                    
                    # Update all app configs
                    for app_config in app_configs.values():
                        if channel_id in app_config['channels']:
                            app_config['channels'][channel_id]['status'] = 'offline'
                else:
                    # Update active modules list for all apps
                    remaining_modules = list(channel_modules[channel_id].keys())
                    for app_config in app_configs.values():
                        if channel_id in app_config['channels']:
                            app_config['channels'][channel_id]['active_modules'] = remaining_modules
        
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Error stopping channel: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/set_roi', methods=['POST'])
def set_roi():
    """Set ROI for queue monitoring on shared channel"""
    data = request.json
    app_name = data.get('app_name')
    channel_id = data.get('channel_id')
    roi_points = data.get('roi_points')
    
    try:
        if channel_id in channel_modules and app_name in channel_modules[channel_id]:
            module = channel_modules[channel_id][app_name]
            if hasattr(module, 'set_roi'):
                module.set_roi(roi_points)
                return jsonify({'success': True})
        
        return jsonify({'success': False, 'error': 'Module not found on this channel'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/get_roi/<app_name>/<channel_id>')
def get_roi(app_name, channel_id):
    """Get current ROI configuration for queue monitoring"""
    try:
        if channel_id in channel_modules and app_name in channel_modules[channel_id]:
            module = channel_modules[channel_id][app_name]
            if hasattr(module, 'get_roi'):
                roi_points = module.get_roi()
                return jsonify({'success': True, 'roi_points': roi_points})
        
        return jsonify({'success': False, 'error': 'Module not found on this channel'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/set_counting_line', methods=['POST'])
def set_counting_line():
    """Set counting line for people counter on shared channel"""
    data = request.json
    app_name = data.get('app_name')
    channel_id = data.get('channel_id')
    line_config = data.get('line_config')
    
    try:
        if channel_id in channel_modules and app_name in channel_modules[channel_id]:
            module = channel_modules[channel_id][app_name]
            if hasattr(module, 'set_counting_line'):
                module.set_counting_line(line_config)
                return jsonify({'success': True})
        
        return jsonify({'success': False, 'error': 'Module not found on this channel'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/get_counting_line/<app_name>/<channel_id>')
def get_counting_line(app_name, channel_id):
    """Get current counting line configuration for people counter"""
    try:
        if channel_id in channel_modules and app_name in channel_modules[channel_id]:
            module = channel_modules[channel_id][app_name]
            if hasattr(module, 'get_counting_line'):
                line_config = module.get_counting_line()
                return jsonify({'success': True, 'line_config': line_config})
        
        return jsonify({'success': False, 'error': 'Module not found on this channel'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/set_bag_settings', methods=['POST'])
def set_bag_settings():
    """Set bag detection settings (time threshold, proximity threshold, confidence)"""
    data = request.json
    app_name = 'BagDetection'
    channel_id = data.get('channel_id')
    time_threshold = data.get('time_threshold')
    proximity_threshold = data.get('proximity_threshold')
    confidence = data.get('confidence')
    alert_cooldown = data.get('alert_cooldown')
    
    try:
        if channel_id in channel_modules and app_name in channel_modules[channel_id]:
            module = channel_modules[channel_id][app_name]
            
            # Update module configuration
            if time_threshold is not None:
                module.time_threshold = float(time_threshold)
            if proximity_threshold is not None:
                module.proximity_threshold = float(proximity_threshold)
            if confidence is not None:
                module.confidence = float(confidence)
            if alert_cooldown is not None:
                module.alert_cooldown = float(alert_cooldown)
            
            return jsonify({
                'success': True,
                'message': 'Bag detection settings updated successfully',
                'settings': {
                    'time_threshold': module.time_threshold,
                    'proximity_threshold': module.proximity_threshold,
                    'confidence': module.confidence,
                    'alert_cooldown': module.alert_cooldown
                }
            })
        
        return jsonify({'success': False, 'error': 'Bag detection module not running on this channel'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/get_channel_status/<channel_id>')
def get_channel_status(channel_id):
    """Get status of all modules running on a channel"""
    try:
        if channel_id in shared_video_processors:
            processor = shared_video_processors[channel_id]
            status = processor.get_status()
            
            # Add module-specific information
            module_info = {}
            for module_name, module in channel_modules.get(channel_id, {}).items():
                if hasattr(module, 'get_current_counts'):
                    module_info[module_name] = module.get_current_counts()
                elif hasattr(module, 'get_current_status'):
                    module_info[module_name] = module.get_current_status()
            
            status['module_info'] = module_info
            return jsonify(status)
        else:
            return jsonify({'error': 'Channel not found', 'channel_id': channel_id})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/get_footfall_report/<channel_id>')
def get_footfall_report(channel_id):
    """Get footfall report for a channel"""
    period = request.args.get('period', '7days')
    try:
        report_data = db_manager.get_footfall_report(channel_id, period)
        return jsonify(report_data)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/get_queue_report/<channel_id>')
def get_queue_report(channel_id):
    """Get queue analytics report"""
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    try:
        report_data = db_manager.get_queue_report(channel_id, start_date, end_date)
        return jsonify(report_data)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/get_alert_gifs')
def get_alert_gifs():
    """Get alert GIFs with optional filtering"""
    channel_id = request.args.get('channel_id')
    alert_type = request.args.get('alert_type')
    limit = int(request.args.get('limit', 20))
    
    try:
        alert_gifs = db_manager.get_alert_gifs(channel_id, alert_type, limit)
        return jsonify({
            'success': True,
            'alert_gifs': alert_gifs,
            'count': len(alert_gifs)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/delete_alert_gif/<int:gif_id>', methods=['DELETE'])
def delete_alert_gif(gif_id):
    """Delete an alert GIF"""
    try:
        success = db_manager.delete_alert_gif(gif_id)
        if success:
            return jsonify({'success': True, 'message': 'Alert GIF deleted successfully'})
        else:
            return jsonify({'success': False, 'error': 'Alert GIF not found'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/clear_old_alerts', methods=['POST'])
def clear_old_alerts():
    """Clear old queue monitor and bag detection alert GIFs"""
    data = request.json
    days = data.get('days', 7)
    alert_type = data.get('alert_type', 'all')  # 'queue_alert', 'bag_unattended', or 'all'
    
    try:
        with app.app_context():
            deleted_count = db_manager.cleanup_old_alert_gifs(max_age_days=days, alert_type=alert_type)
        
        return jsonify({
            'success': True,
            'deleted_count': deleted_count,
            'message': f'Deleted {deleted_count} alerts older than {days} days'
        })
    except Exception as e:
        logger.error(f"Error clearing old alerts: {e}")
        return jsonify({'success': False, 'error': str(e)})

# Heatmap Routes
@app.route('/api/get_heatmap_snapshots')
def get_heatmap_snapshots():
    """Get heatmap snapshots with optional filtering"""
    channel_id = request.args.get('channel_id')
    limit = int(request.args.get('limit', 20))
    
    try:
        snapshots = db_manager.get_heatmap_snapshots(channel_id, limit)
        return jsonify({
            'success': True,
            'snapshots': snapshots,
            'count': len(snapshots)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/delete_heatmap_snapshot/<int:snapshot_id>', methods=['DELETE'])
def delete_heatmap_snapshot(snapshot_id):
    """Delete a heatmap snapshot"""
    try:
        success = db_manager.delete_heatmap_snapshot(snapshot_id)
        if success:
            return jsonify({'success': True, 'message': 'Heatmap snapshot deleted successfully'})
        else:
            return jsonify({'success': False, 'error': 'Snapshot not found'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/capture_heatmap_snapshot', methods=['POST'])
def capture_heatmap_snapshot():
    """Manually capture a heatmap snapshot"""
    data = request.json
    channel_id = data.get('channel_id')
    
    if not channel_id:
        return jsonify({'success': False, 'error': 'Channel ID is required'})
    
    try:
        # Check if Heatmap module is running for this channel
        if channel_id not in channel_modules or 'Heatmap' not in channel_modules[channel_id]:
            return jsonify({'success': False, 'error': 'Heatmap not running on this channel'})
        
        heatmap_module = channel_modules[channel_id]['Heatmap']
        
        # Get snapshot frame from the module
        snapshot_frame = heatmap_module.get_snapshot_frame()
        
        if snapshot_frame is None:
            return jsonify({'success': False, 'error': 'Failed to capture snapshot'})
        
        # Save the snapshot
        import cv2
        import os
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"heatmap_{channel_id}_{timestamp}.jpg"
        filepath = os.path.join('static/heatmaps', filename)
        
        # Ensure directory exists
        os.makedirs('static/heatmaps', exist_ok=True)
        
        # Save image
        cv2.imwrite(filepath, snapshot_frame)
        
        # Get file size
        file_size = os.path.getsize(filepath)
        
        # Save to database
        with app.app_context():
            snapshot_id = db_manager.save_heatmap_snapshot(
                channel_id=channel_id,
                filename=filename,
                filepath=filepath,
                file_size=file_size
            )
        
        return jsonify({
            'success': True,
            'message': 'Snapshot captured successfully',
            'snapshot_id': snapshot_id,
            'filename': filename
        })
        
    except Exception as e:
        logger.error(f"Error capturing heatmap snapshot: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/clear_old_heatmap_snapshots', methods=['POST'])
def clear_old_heatmap_snapshots():
    """Clear old heatmap snapshots"""
    data = request.json
    days = data.get('days', 7)
    
    try:
        with app.app_context():
            deleted_count = db_manager.clear_old_heatmap_snapshots(days)
        
        return jsonify({
            'success': True,
            'deleted_count': deleted_count,
            'message': f'Deleted {deleted_count} snapshots older than {days} days'
        })
    except Exception as e:
        logger.error(f"Error clearing old heatmap snapshots: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/set_heatmap_settings', methods=['POST'])
def set_heatmap_settings():
    """Set heatmap settings (decay rate, snapshot interval, intensity)"""
    data = request.json
    channel_id = data.get('channel_id')
    decay_rate = data.get('decay_rate')
    snapshot_interval = data.get('snapshot_interval')
    intensity = data.get('intensity')
    
    if not channel_id:
        return jsonify({'success': False, 'error': 'Channel ID is required'})
    
    try:
        if channel_id not in channel_modules or 'Heatmap' not in channel_modules[channel_id]:
            return jsonify({'success': False, 'error': 'Heatmap not running on this channel'})
        
        heatmap_module = channel_modules[channel_id]['Heatmap']
        
        # Update module settings
        if decay_rate is not None:
            heatmap_module.decay_rate = float(decay_rate)
        if snapshot_interval is not None:
            heatmap_module.snapshot_interval = int(snapshot_interval)
        if intensity is not None:
            heatmap_module.intensity = float(intensity)
        
        return jsonify({
            'success': True,
            'message': 'Heatmap settings updated successfully',
            'settings': {
                'decay_rate': heatmap_module.decay_rate,
                'snapshot_interval': heatmap_module.snapshot_interval,
                'intensity': heatmap_module.intensity
            }
        })
        
    except Exception as e:
        logger.error(f"Error setting heatmap settings: {e}")
        return jsonify({'success': False, 'error': str(e)})

# Configuration Management Routes
@app.route('/api/save_config', methods=['POST'])
def save_config():
    """Save channel configuration (ROI, counting line, etc.)"""
    try:
        data = request.json
        channel_id = data.get('channel_id')
        app_name = data.get('app_name')  # 'PeopleCounter' or 'QueueMonitor'
        config_type = data.get('config_type')  # 'roi' or 'counting_line'
        config_data = data.get('config_data')
        
        if not all([channel_id, app_name, config_type, config_data]):
            return jsonify({'success': False, 'error': 'Missing required fields'})
        
        db_manager.save_channel_config(channel_id, app_name, config_type, config_data)
        logger.info(f"Configuration saved: {channel_id} - {app_name} - {config_type}")
        
        return jsonify({'success': True, 'message': 'Configuration saved successfully'})
    except Exception as e:
        logger.error(f"Error saving configuration: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/get_config/<channel_id>/<app_name>/<config_type>')
def get_config(channel_id, app_name, config_type):
    """Get channel configuration"""
    try:
        config_data = db_manager.get_channel_config(channel_id, app_name, config_type)
        if config_data:
            return jsonify({'success': True, 'config': config_data})
        else:
            return jsonify({'success': False, 'message': 'No configuration found'})
    except Exception as e:
        logger.error(f"Error retrieving configuration: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/get_all_configs/<channel_id>/<app_name>')
def get_all_configs(channel_id, app_name):
    """Get all configurations for a channel and app"""
    try:
        roi_config = db_manager.get_channel_config(channel_id, app_name, 'roi')
        line_config = db_manager.get_channel_config(channel_id, app_name, 'counting_line')
        
        return jsonify({
            'success': True,
            'configs': {
                'roi': roi_config,
                'counting_line': line_config
            }
        })
    except Exception as e:
        logger.error(f"Error retrieving configurations: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/update_bag_detection_config', methods=['POST'])
def update_bag_detection_config():
    """Update bag detection configuration"""
    try:
        data = request.json
        channel_id = data.get('channel_id')
        config = data.get('config', {})
        
        if not channel_id:
            return jsonify({'success': False, 'error': 'Channel ID required'})
        
        # Update the active module if it exists
        if channel_id in channel_modules and 'BagDetection' in channel_modules[channel_id]:
            module = channel_modules[channel_id]['BagDetection']
            module.update_config(config)
            return jsonify({'success': True, 'message': 'Configuration updated'})
        else:
            return jsonify({'success': False, 'error': 'Bag detection not running on this channel'})
            
    except Exception as e:
        logger.error(f"Error updating bag detection config: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/get_bag_detection_stats/<channel_id>')
def get_bag_detection_stats(channel_id):
    """Get bag detection statistics"""
    try:
        if channel_id in channel_modules and 'BagDetection' in channel_modules[channel_id]:
            module = channel_modules[channel_id]['BagDetection']
            stats = module.get_statistics()
            return jsonify({'success': True, 'stats': stats})
        else:
            return jsonify({'success': False, 'error': 'Bag detection not running'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# Socket.IO events
@socketio.on('connect')
def handle_connect():
    logger.info('Client connected')
    emit('status', {'message': 'Connected to Sakshi.AI'})

@socketio.on('disconnect')
def handle_disconnect():
    logger.info('Client disconnected')

def create_database_tables():
    """Create database tables"""
    with app.app_context():
        db.create_all()
        logger.info("Database tables created")

if __name__ == '__main__':
    # Create database tables
    create_database_tables()
    
    # Start the application
    logger.info("Starting Sakshi.AI Video Analytics Platform")
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
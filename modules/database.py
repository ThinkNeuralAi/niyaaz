"""
Database Models for Sakshi.AI
"""
from datetime import datetime, timedelta
from flask_sqlalchemy import SQLAlchemy
import json
import logging

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, db):
        self.db = db
        self.define_models()
    
    def define_models(self):
        """Define database models"""
        
        # Footfall tracking for People Counter
        class DailyFootfall(self.db.Model):
            __tablename__ = 'daily_footfall'
            
            id = self.db.Column(self.db.Integer, primary_key=True)
            channel_id = self.db.Column(self.db.String(50), nullable=False)
            report_date = self.db.Column(self.db.Date, nullable=False)
            in_count = self.db.Column(self.db.Integer, default=0)
            out_count = self.db.Column(self.db.Integer, default=0)
            
            __table_args__ = (self.db.UniqueConstraint('channel_id', 'report_date'),)
        
        class HourlyFootfall(self.db.Model):
            __tablename__ = 'hourly_footfall'
            
            id = self.db.Column(self.db.Integer, primary_key=True)
            channel_id = self.db.Column(self.db.String(50), nullable=False)
            report_date = self.db.Column(self.db.Date, nullable=False)
            hour = self.db.Column(self.db.Integer, nullable=False)  # 0-23
            in_count = self.db.Column(self.db.Integer, default=0)
            out_count = self.db.Column(self.db.Integer, default=0)
            
            __table_args__ = (self.db.UniqueConstraint('channel_id', 'report_date', 'hour'),)
        
        # Queue analytics for Queue Monitor
        class QueueAnalytics(self.db.Model):
            __tablename__ = 'queue_analytics'
            
            id = self.db.Column(self.db.Integer, primary_key=True)
            channel_id = self.db.Column(self.db.String(50), nullable=False)
            timestamp = self.db.Column(self.db.DateTime, default=datetime.utcnow)
            queue_count = self.db.Column(self.db.Integer, default=0)
            counter_count = self.db.Column(self.db.Integer, default=0)
            alert_triggered = self.db.Column(self.db.Boolean, default=False)
            alert_message = self.db.Column(self.db.Text)
        
        # Configuration storage
        class ChannelConfig(self.db.Model):
            __tablename__ = 'channel_config'
            
            id = self.db.Column(self.db.Integer, primary_key=True)
            channel_id = self.db.Column(self.db.String(50), nullable=False)
            app_name = self.db.Column(self.db.String(50), nullable=False)
            config_type = self.db.Column(self.db.String(50), nullable=False)  # 'roi', 'counting_line', 'settings'
            config_data = self.db.Column(self.db.Text)  # JSON data
            created_at = self.db.Column(self.db.DateTime, default=datetime.utcnow)
            updated_at = self.db.Column(self.db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
            
            __table_args__ = (self.db.UniqueConstraint('channel_id', 'app_name', 'config_type'),)
        
        # Detection events for audit trail
        class DetectionEvent(self.db.Model):
            __tablename__ = 'detection_events'
            
            id = self.db.Column(self.db.Integer, primary_key=True)
            channel_id = self.db.Column(self.db.String(50), nullable=False)
            app_name = self.db.Column(self.db.String(50), nullable=False)
            event_type = self.db.Column(self.db.String(50), nullable=False)  # 'person_in', 'person_out', 'queue_alert'
            event_data = self.db.Column(self.db.Text)  # JSON data
            confidence = self.db.Column(self.db.Float)
            timestamp = self.db.Column(self.db.DateTime, default=datetime.utcnow)
        
        # RTSP channels configuration
        class RTSPChannel(self.db.Model):
            __tablename__ = 'rtsp_channels'
            
            id = self.db.Column(self.db.Integer, primary_key=True)
            channel_id = self.db.Column(self.db.String(50), unique=True, nullable=False)
            name = self.db.Column(self.db.String(100), nullable=False)
            rtsp_url = self.db.Column(self.db.String(500), nullable=False)
            description = self.db.Column(self.db.Text)
            is_active = self.db.Column(self.db.Boolean, default=True)
            created_at = self.db.Column(self.db.DateTime, default=datetime.utcnow)
            updated_at = self.db.Column(self.db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
        
        # Alert GIF recordings
        class AlertGif(self.db.Model):
            __tablename__ = 'alert_gifs'
            
            id = self.db.Column(self.db.Integer, primary_key=True)
            channel_id = self.db.Column(self.db.String(50), nullable=False)
            alert_type = self.db.Column(self.db.String(50), nullable=False)  # 'queue_alert', 'people_alert', etc.
            gif_filename = self.db.Column(self.db.String(255), nullable=False)
            gif_path = self.db.Column(self.db.String(500), nullable=False)
            alert_message = self.db.Column(self.db.Text)
            alert_data = self.db.Column(self.db.Text)  # JSON data with alert details
            frame_count = self.db.Column(self.db.Integer)
            file_size = self.db.Column(self.db.Integer)  # File size in bytes
            duration_seconds = self.db.Column(self.db.Float)
            created_at = self.db.Column(self.db.DateTime, default=datetime.utcnow)
        
        # Heatmap snapshots
        class HeatmapSnapshot(self.db.Model):
            __tablename__ = 'heatmap_snapshots'
            
            id = self.db.Column(self.db.Integer, primary_key=True)
            channel_id = self.db.Column(self.db.String(50), nullable=False)
            snapshot_filename = self.db.Column(self.db.String(255), nullable=False)
            snapshot_path = self.db.Column(self.db.String(500), nullable=False)
            hotspot_count = self.db.Column(self.db.Integer, default=0)
            hotspots_data = self.db.Column(self.db.Text)  # JSON data with hotspot locations
            created_at = self.db.Column(self.db.DateTime, default=datetime.utcnow)
        
        # Store model classes for access
        self.DailyFootfall = DailyFootfall
        self.HourlyFootfall = HourlyFootfall
        self.QueueAnalytics = QueueAnalytics
        self.ChannelConfig = ChannelConfig
        self.DetectionEvent = DetectionEvent
        self.RTSPChannel = RTSPChannel
        self.AlertGif = AlertGif
        self.HeatmapSnapshot = HeatmapSnapshot
    
    def update_footfall_count(self, channel_id, direction, count=1):
        """Update footfall counts for People Counter"""
        now = datetime.now()
        today = now.date()
        current_hour = now.hour
        
        # Update daily count
        daily_record = self.DailyFootfall.query.filter_by(
            channel_id=channel_id, 
            report_date=today
        ).first()
        
        if not daily_record:
            daily_record = self.DailyFootfall(
                channel_id=channel_id,
                report_date=today,
                in_count=0,
                out_count=0
            )
            self.db.session.add(daily_record)
        
        # Ensure counts are not None
        if daily_record.in_count is None:
            daily_record.in_count = 0
        if daily_record.out_count is None:
            daily_record.out_count = 0
        
        if direction == 'in':
            daily_record.in_count += count
        elif direction == 'out':
            daily_record.out_count += count
        
        # Update hourly count
        hourly_record = self.HourlyFootfall.query.filter_by(
            channel_id=channel_id,
            report_date=today,
            hour=current_hour
        ).first()
        
        if not hourly_record:
            hourly_record = self.HourlyFootfall(
                channel_id=channel_id,
                report_date=today,
                hour=current_hour,
                in_count=0,
                out_count=0
            )
            self.db.session.add(hourly_record)
        
        # Ensure counts are not None
        if hourly_record.in_count is None:
            hourly_record.in_count = 0
        if hourly_record.out_count is None:
            hourly_record.out_count = 0
        
        if direction == 'in':
            hourly_record.in_count += count
        elif direction == 'out':
            hourly_record.out_count += count
        
        try:
            self.db.session.commit()
        except Exception as e:
            self.db.session.rollback()
            raise e
    
    def log_queue_analytics(self, channel_id, queue_count, counter_count, alert_triggered=False, alert_message=None):
        """Log queue analytics data"""
        analytics = self.QueueAnalytics(
            channel_id=channel_id,
            queue_count=queue_count,
            counter_count=counter_count,
            alert_triggered=alert_triggered,
            alert_message=alert_message
        )
        
        self.db.session.add(analytics)
        try:
            self.db.session.commit()
        except Exception as e:
            self.db.session.rollback()
            raise e
    
    def save_channel_config(self, channel_id, app_name, config_type, config_data):
        """Save channel configuration"""
        config = self.ChannelConfig.query.filter_by(
            channel_id=channel_id,
            app_name=app_name,
            config_type=config_type
        ).first()
        
        if not config:
            config = self.ChannelConfig(
                channel_id=channel_id,
                app_name=app_name,
                config_type=config_type
            )
            self.db.session.add(config)
        
        config.config_data = json.dumps(config_data)
        config.updated_at = datetime.utcnow()
        
        try:
            self.db.session.commit()
        except Exception as e:
            self.db.session.rollback()
            raise e
    
    def get_channel_config(self, channel_id, app_name, config_type):
        """Get channel configuration"""
        config = self.ChannelConfig.query.filter_by(
            channel_id=channel_id,
            app_name=app_name,
            config_type=config_type
        ).first()
        
        if config and config.config_data:
            return json.loads(config.config_data)
        return None
    
    def log_detection_event(self, channel_id, app_name, event_type, event_data, confidence=None):
        """Log detection event"""
        event = self.DetectionEvent(
            channel_id=channel_id,
            app_name=app_name,
            event_type=event_type,
            event_data=json.dumps(event_data),
            confidence=confidence
        )
        
        self.db.session.add(event)
        try:
            self.db.session.commit()
        except Exception as e:
            self.db.session.rollback()
            raise e
    
    def get_footfall_report(self, channel_id, period='7days'):
        """Get footfall report for specified period"""
        end_date = datetime.now().date()
        
        if period == '24hours':
            start_date = end_date
            # Get hourly data for today
            hourly_data = self.HourlyFootfall.query.filter_by(
                channel_id=channel_id,
                report_date=start_date
            ).order_by(self.HourlyFootfall.hour).all()
            
            return {
                'period': period,
                'data': [
                    {
                        'hour': h.hour,
                        'in_count': h.in_count,
                        'out_count': h.out_count,
                        'total': h.in_count + h.out_count
                    } for h in hourly_data
                ]
            }
        
        elif period == '7days':
            start_date = end_date - timedelta(days=6)
        elif period == '30days':
            start_date = end_date - timedelta(days=29)
        else:
            start_date = end_date - timedelta(days=6)
        
        # Get daily data
        daily_data = self.DailyFootfall.query.filter(
            self.DailyFootfall.channel_id == channel_id,
            self.DailyFootfall.report_date >= start_date,
            self.DailyFootfall.report_date <= end_date
        ).order_by(self.DailyFootfall.report_date).all()
        
        return {
            'period': period,
            'data': [
                {
                    'date': d.report_date.isoformat(),
                    'in_count': d.in_count,
                    'out_count': d.out_count,
                    'total': d.in_count + d.out_count
                } for d in daily_data
            ]
        }
    
    def get_queue_report(self, channel_id, start_date=None, end_date=None):
        """Get queue analytics report"""
        if not start_date:
            start_date = (datetime.now() - timedelta(days=7)).date()
        if not end_date:
            end_date = datetime.now().date()
        
        # Convert to datetime for filtering
        start_datetime = datetime.combine(start_date, datetime.min.time()) if isinstance(start_date, str) else datetime.combine(start_date, datetime.min.time())
        end_datetime = datetime.combine(end_date, datetime.max.time()) if isinstance(end_date, str) else datetime.combine(end_date, datetime.max.time())
        
        analytics_data = self.QueueAnalytics.query.filter(
            self.QueueAnalytics.channel_id == channel_id,
            self.QueueAnalytics.timestamp >= start_datetime,
            self.QueueAnalytics.timestamp <= end_datetime
        ).order_by(self.QueueAnalytics.timestamp).all()
        
        return {
            'start_date': start_date.isoformat() if hasattr(start_date, 'isoformat') else start_date,
            'end_date': end_date.isoformat() if hasattr(end_date, 'isoformat') else end_date,
            'data': [
                {
                    'timestamp': a.timestamp.isoformat(),
                    'queue_count': a.queue_count,
                    'counter_count': a.counter_count,
                    'alert_triggered': a.alert_triggered,
                    'alert_message': a.alert_message
                } for a in analytics_data
            ]
        }
    
    def save_rtsp_channel(self, channel_id, name, rtsp_url, description=None):
        """Save RTSP channel configuration"""
        try:
            # Check if channel already exists
            existing = self.RTSPChannel.query.filter_by(channel_id=channel_id).first()
            
            if existing:
                # Update existing
                existing.name = name
                existing.rtsp_url = rtsp_url
                existing.description = description
                existing.updated_at = datetime.utcnow()
            else:
                # Create new
                channel = self.RTSPChannel(
                    channel_id=channel_id,
                    name=name,
                    rtsp_url=rtsp_url,
                    description=description
                )
                self.db.session.add(channel)
            
            self.db.session.commit()
            return True
            
        except Exception as e:
            self.db.session.rollback()
            raise e
    
    def get_rtsp_channels(self):
        """Get all active RTSP channels"""
        channels = self.RTSPChannel.query.filter_by(is_active=True).all()
        return [
            {
                'channel_id': c.channel_id,
                'name': c.name,
                'rtsp_url': c.rtsp_url,
                'description': c.description,
                'created_at': c.created_at.isoformat(),
                'updated_at': c.updated_at.isoformat()
            } for c in channels
        ]
    
    def get_rtsp_channel(self, channel_id):
        """Get specific RTSP channel"""
        channel = self.RTSPChannel.query.filter_by(
            channel_id=channel_id, 
            is_active=True
        ).first()
        
        if channel:
            return {
                'channel_id': channel.channel_id,
                'name': channel.name,
                'rtsp_url': channel.rtsp_url,
                'description': channel.description,
                'created_at': channel.created_at.isoformat(),
                'updated_at': channel.updated_at.isoformat()
            }
        return None
    
    def remove_rtsp_channel(self, channel_id):
        """Remove RTSP channel (soft delete)"""
        try:
            channel = self.RTSPChannel.query.filter_by(channel_id=channel_id).first()
            if channel:
                channel.is_active = False
                channel.updated_at = datetime.utcnow()
                self.db.session.commit()
                return True
            return False
            
        except Exception as e:
            self.db.session.rollback()
            raise e
    
    def save_alert_gif(self, channel_id, alert_type, gif_info, alert_message=None, alert_data=None):
        """Save alert GIF information to database"""
        try:
            import os
            
            # Get file size if path exists
            file_size = 0
            if os.path.exists(gif_info.get('gif_path', '')):
                file_size = os.path.getsize(gif_info['gif_path'])
            
            alert_gif = self.AlertGif(
                channel_id=channel_id,
                alert_type=alert_type,
                gif_filename=gif_info.get('gif_filename', ''),
                gif_path=gif_info.get('gif_path', ''),
                alert_message=alert_message,
                alert_data=json.dumps(alert_data) if alert_data else None,
                frame_count=gif_info.get('frame_count', 0),
                file_size=file_size,
                duration_seconds=gif_info.get('duration', 0.0)
            )
            
            self.db.session.add(alert_gif)
            self.db.session.commit()
            
            return alert_gif.id
            
        except Exception as e:
            self.db.session.rollback()
            raise e
    
    def get_alert_gifs(self, channel_id=None, alert_type=None, limit=50):
        """Get alert GIFs from database"""
        try:
            query = self.AlertGif.query
            
            if channel_id:
                query = query.filter_by(channel_id=channel_id)
            
            if alert_type:
                query = query.filter_by(alert_type=alert_type)
            
            # Order by creation time (newest first)
            alert_gifs = query.order_by(self.AlertGif.created_at.desc()).limit(limit).all()
            
            return [
                {
                    'id': gif.id,
                    'channel_id': gif.channel_id,
                    'alert_type': gif.alert_type,
                    'gif_filename': gif.gif_filename,
                    'gif_path': gif.gif_path,
                    'alert_message': gif.alert_message,
                    'alert_data': json.loads(gif.alert_data) if gif.alert_data else None,
                    'frame_count': gif.frame_count,
                    'file_size': gif.file_size,
                    'file_size_mb': round(gif.file_size / (1024 * 1024), 2) if gif.file_size else 0,
                    'duration_seconds': gif.duration_seconds,
                    'created_at': gif.created_at.isoformat(),
                    'gif_url': f'/static/alerts/{gif.gif_filename}'
                } for gif in alert_gifs
            ]
            
        except Exception as e:
            logger.error(f"Error getting alert GIFs: {e}")
            return []
    
    def delete_alert_gif(self, gif_id):
        """Delete alert GIF from database"""
        try:
            alert_gif = self.AlertGif.query.get(gif_id)
            if alert_gif:
                # Also try to delete the actual file
                import os
                if os.path.exists(alert_gif.gif_path):
                    os.remove(alert_gif.gif_path)
                
                self.db.session.delete(alert_gif)
                self.db.session.commit()
                return True
            return False
            
        except Exception as e:
            self.db.session.rollback()
            raise e
    
    def cleanup_old_alert_gifs(self, max_age_days=30, alert_type='all'):
        """Clean up old alert GIF records and files"""
        try:
            import os
            from datetime import timedelta
            
            cutoff_date = datetime.utcnow() - timedelta(days=max_age_days)
            
            # Build query based on alert type
            query = self.AlertGif.query.filter(
                self.AlertGif.created_at < cutoff_date
            )
            
            # Filter by alert type if specified
            if alert_type != 'all':
                query = query.filter(self.AlertGif.alert_type == alert_type)
            
            old_gifs = query.all()
            
            removed_count = 0
            for gif in old_gifs:
                # Delete file if exists
                if os.path.exists(gif.gif_path):
                    try:
                        os.remove(gif.gif_path)
                    except:
                        pass
                
                # Delete database record
                self.db.session.delete(gif)
                removed_count += 1
            
            self.db.session.commit()
            return removed_count
            
        except Exception as e:
            self.db.session.rollback()
            logger.error(f"Error cleaning up old alert GIFs: {e}")
            raise e
    
    def save_heatmap_snapshot(self, channel_id, snapshot_path, hotspot_count, hotspots):
        """Save heatmap snapshot to database"""
        try:
            import os
            
            snapshot_filename = os.path.basename(snapshot_path)
            file_size = os.path.getsize(snapshot_path) if os.path.exists(snapshot_path) else 0
            
            snapshot = self.HeatmapSnapshot(
                channel_id=channel_id,
                snapshot_filename=snapshot_filename,
                snapshot_path=snapshot_path,
                hotspot_count=hotspot_count,
                hotspots_data=json.dumps(hotspots)
            )
            
            self.db.session.add(snapshot)
            self.db.session.commit()
            
            return snapshot.id
            
        except Exception as e:
            self.db.session.rollback()
            logger.error(f"Error saving heatmap snapshot: {e}")
            raise e
    
    def get_heatmap_snapshots(self, channel_id=None, limit=20):
        """Get heatmap snapshots from database"""
        try:
            query = self.HeatmapSnapshot.query
            
            if channel_id:
                query = query.filter_by(channel_id=channel_id)
            
            snapshots = query.order_by(
                self.HeatmapSnapshot.created_at.desc()
            ).limit(limit).all()
            
            return [
                {
                    'id': snap.id,
                    'channel_id': snap.channel_id,
                    'snapshot_filename': snap.snapshot_filename,
                    'snapshot_path': snap.snapshot_path,
                    'hotspot_count': snap.hotspot_count,
                    'hotspots': json.loads(snap.hotspots_data) if snap.hotspots_data else [],
                    'created_at': snap.created_at.isoformat(),
                    'snapshot_url': f'/static/heatmaps/{snap.snapshot_filename}'
                } for snap in snapshots
            ]
            
        except Exception as e:
            logger.error(f"Error getting heatmap snapshots: {e}")
            return []
    
    def delete_heatmap_snapshot(self, snapshot_id):
        """Delete heatmap snapshot from database"""
        try:
            snapshot = self.HeatmapSnapshot.query.get(snapshot_id)
            if snapshot:
                # Delete the actual file
                import os
                if os.path.exists(snapshot.snapshot_path):
                    os.remove(snapshot.snapshot_path)
                
                self.db.session.delete(snapshot)
                self.db.session.commit()
                return True
            return False
            
        except Exception as e:
            self.db.session.rollback()
            raise e
    def clear_old_heatmap_snapshots(self, days=7):
        """Clear old heatmap snapshots older than specified days"""
        try:
            import os
            from datetime import timedelta
            
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            old_snapshots = self.HeatmapSnapshot.query.filter(
                self.HeatmapSnapshot.created_at < cutoff_date
            ).all()
            
            deleted_count = 0
            for snapshot in old_snapshots:
                # Delete file if exists
                if os.path.exists(snapshot.snapshot_path):
                    try:
                        os.remove(snapshot.snapshot_path)
                    except:
                        pass
                
                # Delete database record
                self.db.session.delete(snapshot)
                deleted_count += 1
            
            self.db.session.commit()
            return deleted_count
            
        except Exception as e:
            self.db.session.rollback()
            logger.error(f"Error clearing old heatmap snapshots: {e}")
            raise e

"""
Test script to generate sample footfall data for testing hourly logs
"""
from app import app, db_manager
from datetime import datetime, timedelta
import random

def generate_test_data():
    """Generate sample footfall data for testing"""
    with app.app_context():
        print('='*70)
        print('Generating Test Footfall Data')
        print('='*70)
        
        channel_id = 'test_channel'
        today = datetime.now().date()
        
        # Clear existing test data for today
        try:
            db_manager.HourlyFootfall.query.filter_by(
                channel_id=channel_id,
                report_date=today
            ).delete()
            
            db_manager.DailyFootfall.query.filter_by(
                channel_id=channel_id,
                report_date=today
            ).delete()
            
            db_manager.db.session.commit()
            print(f'\n✓ Cleared existing test data')
        except Exception as e:
            print(f'\n⚠ Error clearing old data: {e}')
            db_manager.db.session.rollback()
        
        # Generate hourly data for today (8 AM to current hour)
        current_hour = datetime.now().hour
        start_hour = 8
        
        print(f'\n📊 Generating hourly data from {start_hour}:00 to {current_hour}:00')
        
        total_in = 0
        total_out = 0
        
        for hour in range(start_hour, current_hour + 1):
            # Simulate realistic patterns
            if 9 <= hour <= 11:  # Morning rush
                in_count = random.randint(15, 30)
                out_count = random.randint(5, 15)
            elif 12 <= hour <= 14:  # Lunch time
                in_count = random.randint(20, 35)
                out_count = random.randint(18, 32)
            elif 17 <= hour <= 19:  # Evening rush
                in_count = random.randint(10, 25)
                out_count = random.randint(20, 40)
            else:  # Normal hours
                in_count = random.randint(5, 15)
                out_count = random.randint(5, 15)
            
            # Create hourly record
            hourly_record = db_manager.HourlyFootfall(
                channel_id=channel_id,
                report_date=today,
                hour=hour,
                in_count=in_count,
                out_count=out_count
            )
            db_manager.db.session.add(hourly_record)
            
            total_in += in_count
            total_out += out_count
            
            print(f'   Hour {hour:02d}:00 - IN: {in_count}, OUT: {out_count}')
        
        # Create daily record
        daily_record = db_manager.DailyFootfall(
            channel_id=channel_id,
            report_date=today,
            in_count=total_in,
            out_count=total_out
        )
        db_manager.db.session.add(daily_record)
        
        try:
            db_manager.db.session.commit()
            print(f'\n✅ Test data generated successfully!')
            print(f'   Total IN: {total_in}')
            print(f'   Total OUT: {total_out}')
            print(f'   Net: {total_in - total_out}')
            
        except Exception as e:
            db_manager.db.session.rollback()
            print(f'\n✗ Error saving data: {e}')
            return
        
        # Generate data for last 7 days
        print(f'\n📊 Generating daily data for last 7 days...')
        
        for days_ago in range(1, 8):
            past_date = today - timedelta(days=days_ago)
            daily_in = random.randint(80, 200)
            daily_out = random.randint(75, 195)
            
            daily_record = db_manager.DailyFootfall(
                channel_id=channel_id,
                report_date=past_date,
                in_count=daily_in,
                out_count=daily_out
            )
            db_manager.db.session.add(daily_record)
            
            print(f'   {past_date} - IN: {daily_in}, OUT: {daily_out}')
        
        try:
            db_manager.db.session.commit()
            print(f'\n✅ Historical data generated!')
            
        except Exception as e:
            db_manager.db.session.rollback()
            print(f'\n✗ Error saving historical data: {e}')
            return
        
        # Verify data
        print(f'\n🔍 Verifying data...')
        hourly_count = db_manager.HourlyFootfall.query.filter_by(
            channel_id=channel_id,
            report_date=today
        ).count()
        
        daily_count = db_manager.DailyFootfall.query.filter_by(
            channel_id=channel_id
        ).count()
        
        print(f'   ✓ Hourly records: {hourly_count}')
        print(f'   ✓ Daily records: {daily_count}')
        
        print(f'\n✅ ALL DONE! Test data is ready.')
        print(f'\n💡 Now open the dashboard and:')
        print(f'   1. Go to People Counter > Analytics & Reports')
        print(f'   2. Select "test_channel" from the dropdown')
        print(f'   3. Choose "Last 24 Hours" for hourly view or "Last 7 Days" for daily view')
        print(f'   4. Click "Load Report"')
        print('='*70)

if __name__ == '__main__':
    generate_test_data()

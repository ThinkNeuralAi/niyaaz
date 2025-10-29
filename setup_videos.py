#!/usr/bin/env python3
"""
Video Placement Helper Script
Helps organize video files for Sakshi.AI processing
"""

import os
import shutil
from pathlib import Path

def list_video_files(directory):
    """List all video files in a directory"""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(Path(directory).glob(f'*{ext}'))
        video_files.extend(Path(directory).glob(f'*{ext.upper()}'))
    
    return video_files

def copy_videos_to_project(source_directory=None):
    """Copy video files from source to project videos directory"""
    videos_dir = Path('videos')
    videos_dir.mkdir(exist_ok=True)
    
    if source_directory:
        source_path = Path(source_directory)
        if not source_path.exists():
            print(f"❌ Source directory does not exist: {source_directory}")
            return
        
        video_files = list_video_files(source_path)
        if not video_files:
            print(f"❌ No video files found in: {source_directory}")
            return
        
        print(f"📁 Found {len(video_files)} video files in {source_directory}")
        
        for video_file in video_files:
            dest_path = videos_dir / video_file.name
            if dest_path.exists():
                print(f"⚠️  Skipping {video_file.name} (already exists)")
                continue
            
            try:
                shutil.copy2(video_file, dest_path)
                print(f"✅ Copied: {video_file.name}")
            except Exception as e:
                print(f"❌ Failed to copy {video_file.name}: {e}")
    
    # List current videos
    current_videos = list_video_files(videos_dir)
    if current_videos:
        print(f"\n📹 Videos ready for processing ({len(current_videos)} files):")
        for video in current_videos:
            size_mb = video.stat().st_size / (1024 * 1024)
            print(f"   - {video.name} ({size_mb:.1f} MB)")
    else:
        print("\n⚠️  No video files in videos/ directory")
        print("   Place your video files in the videos/ directory or run:")
        print(f"   python {__file__} /path/to/your/videos/")

def main():
    """Main function"""
    import sys
    
    print("📹 Sakshi.AI Video Setup Helper")
    print("================================")
    
    if len(sys.argv) > 1:
        source_dir = sys.argv[1]
        copy_videos_to_project(source_dir)
    else:
        copy_videos_to_project()
        print("\n💡 Usage:")
        print(f"   python {sys.argv[0]} /path/to/video/directory")
        print("   This will copy all video files from the directory to videos/")

if __name__ == "__main__":
    main()
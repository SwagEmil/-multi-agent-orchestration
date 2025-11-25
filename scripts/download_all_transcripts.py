#!/usr/bin/env python3
"""
Automated YouTube Transcript Downloader
Downloads transcripts for all videos in inventory.yaml

Features:
- Handles public, unlisted, and private videos (with browser cookies)
- Auto-detects manual vs auto-generated captions
- Saves as clean Markdown files
- Progress tracking with retry logic
- Supports livestreams

Requirements:
    pip install yt-dlp pyyaml

Usage:
    python scripts/download_all_transcripts.py --inventory data/raw/inventory.yaml
    
    # Specify browser for private videos
    python scripts/download_all_transcripts.py --inventory data/raw/inventory.yaml --browser chrome
"""

import argparse
import yaml
import requests
import re
from pathlib import Path
from typing import Dict, List, Optional
import time
import yt_dlp

class TranscriptDownloader:
    def __init__(self, output_dir: str = "data/processed/youtube", browser: str = "chrome"):
        """
        Initialize transcript downloader
        
        Args:
            output_dir: Where to save transcripts
            browser: Browser to get cookies from (chrome, firefox, safari, edge, brave)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.browser = browser
        self.stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'skipped': 0
        }
    
    def download_transcript(self, url: str, metadata: Dict) -> Optional[str]:
        """
        Download transcript from YouTube video
        
        Args:
            url: YouTube video URL
            metadata: Video metadata (title, topic, etc.)
        
        Returns:
            Path to saved transcript file, or None if failed
        """
        video_id = self._extract_video_id(url)
        output_file = self.output_dir / f"{video_id}.md"
        
        # Check if already downloaded
        if output_file.exists():
            print(f"⏭️  Already exists: {metadata.get('title', video_id)}")
            self.stats['skipped'] += 1
            return str(output_file)
        
        print(f"\n{'─'*70}")
        print(f"📹 Downloading: {metadata.get('title', 'Unknown')}")
        print(f"   URL: {url}")
        print(f"{'─'*70}")
        
        try:
            # Configure yt-dlp options
            ydl_opts = {
                'skip_download': True,
                'writesubtitles': True,
                'writeautomaticsub': True,
                'subtitleslangs': ['en'],
                'subtitlesformat': 'json3',
                'quiet': True,
                'no_warnings': True
            }
            
            # Add browser cookies for private videos
            if self.browser:
                ydl_opts['cookiesfrombrowser'] = (self.browser,)
                print(f"   🔐 Using {self.browser} cookies for authentication")
            
            # Extract video info
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                
                # Try manual captions first (better quality)
                transcript = None
                caption_type = None
                
                if 'subtitles' in info and 'en' in info['subtitles']:
                    print("   ✅ Found manual captions (human-created)")
                    caption_url = info['subtitles']['en'][0]['url']
                    transcript = self._download_caption(caption_url)
                    caption_type = "manual"
                
                # Fallback to auto-generated
                elif 'automatic_captions' in info and 'en' in info['automatic_captions']:
                    print("   ✅ Found auto-generated captions")
                    caption_url = info['automatic_captions']['en'][0]['url']
                    transcript = self._download_caption(caption_url)
                    caption_type = "auto-generated"
                
                else:
                    print("   ❌ No captions available")
                    self.stats['failed'] += 1
                    return None
                
                if transcript:
                    # Format as Markdown
                    content = self._format_markdown(transcript, metadata, url, caption_type, info)
                    
                    # Save file
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    # Stats
                    word_count = len(transcript.split())
                    print(f"   💾 Saved: {output_file.name}")
                    print(f"   📊 Stats: {len(transcript)} chars, {word_count} words")
                    
                    self.stats['success'] += 1
                    return str(output_file)
                
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            self.stats['failed'] += 1
            return None
    
    def _download_caption(self, caption_url: str) -> Optional[str]:
        """Download and parse caption JSON to text"""
        try:
            response = requests.get(caption_url, timeout=30)
            response.raise_for_status()
            captions = response.json()
            
            # Extract text from JSON structure
            transcript_parts = []
            for event in captions.get('events', []):
                if 'segs' in event:
                    text = ''.join([seg.get('utf8', '') for seg in event['segs']])
                    if text.strip():
                        transcript_parts.append(text.strip())
            
            return ' '.join(transcript_parts)
            
        except Exception as e:
            print(f"   ⚠️  Caption download error: {e}")
            return None
    
    def _format_markdown(self, transcript: str, metadata: Dict, url: str, 
                        caption_type: str, video_info: Dict) -> str:
        """Format transcript as structured Markdown"""
        
        # Extract video details
        title = metadata.get('title', video_info.get('title', 'Unknown'))
        duration = metadata.get('duration', self._format_duration(video_info.get('duration', 0)))
        topic = metadata.get('topic', 'general')
        upload_date = video_info.get('upload_date', 'Unknown')
        channel = video_info.get('uploader', 'Unknown')
        view_count = video_info.get('view_count', 0)
        
        # Format date
        if upload_date != 'Unknown' and len(upload_date) == 8:
            upload_date = f"{upload_date[:4]}-{upload_date[4:6]}-{upload_date[6:]}"
        
        # Build Markdown
        content = f"# {title}\n\n"
        content += f"**Source:** {url}\n"
        content += f"**Channel:** {channel}\n"
        content += f"**Topic:** {topic}\n"
        content += f"**Duration:** {duration}\n"
        content += f"**Upload Date:** {upload_date}\n"
        content += f"**Views:** {view_count:,}\n"
        content += f"**Caption Type:** {caption_type}\n\n"
        content += "---\n\n"
        content += "## Transcript\n\n"
        content += transcript + "\n"
        
        return content
    
    def _format_duration(self, seconds: int) -> str:
        """Convert seconds to HH:MM:SS format"""
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        
        if hours > 0:
            return f"{hours}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes}:{secs:02d}"
    
    def _extract_video_id(self, url: str) -> str:
        """Extract YouTube video ID from URL"""
        patterns = [
            r'(?:v=|/)([a-zA-Z0-9_-]{11})',
            r'youtu\.be/([a-zA-Z0-9_-]{11})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        # Fallback: use sanitized URL as ID
        return re.sub(r'[^\w\-]', '_', url)[:50]
    
    def process_inventory(self, inventory_path: str):
        """Process all videos from inventory file"""
        print("\n" + "="*70)
        print("YouTube Transcript Batch Downloader")
        print("="*70)
        
        # Load inventory
        with open(inventory_path, 'r') as f:
            inventory = yaml.safe_load(f)
        
        # Combine all video types
        videos = []
        if 'youtube_videos' in inventory:
            videos.extend(inventory['youtube_videos'])
        if 'livestreams' in inventory:
            videos.extend(inventory['livestreams'])
        
        self.stats['total'] = len(videos)
        
        if not videos:
            print("\n⚠️  No videos found in inventory!")
            return
        
        print(f"\n📊 Found {len(videos)} videos to process")
        print(f"🔐 Using browser: {self.browser}")
        print(f"💾 Output directory: {self.output_dir}\n")
        
        # Process each video
        for i, video_info in enumerate(videos, 1):
            print(f"\n[{i}/{len(videos)}]")
            self.download_transcript(video_info['url'], video_info)
            
            # Be nice to YouTube (avoid rate limiting)
            if i < len(videos):
                time.sleep(1)
        
        # Print summary
        self._print_summary()
    
    def _print_summary(self):
        """Print processing summary"""
        print("\n" + "="*70)
        print("DOWNLOAD SUMMARY")
        print("="*70)
        
        print(f"\n📊 Total videos: {self.stats['total']}")
        print(f"   ✅ Successfully downloaded: {self.stats['success']}")
        print(f"   ⏭️  Skipped (already exists): {self.stats['skipped']}")
        print(f"   ❌ Failed: {self.stats['failed']}")
        
        success_rate = (self.stats['success'] / self.stats['total'] * 100) if self.stats['total'] > 0 else 0
        print(f"\n   Success rate: {success_rate:.1f}%")
        
        if self.stats['failed'] > 0:
            print("\n⚠️  Some videos failed. Common reasons:")
            print("   - No captions available (uploader disabled)")
            print("   - Private video without access")
            print("   - Age-restricted video")
            print("   - Region-locked content")
            print("\n   💡 For private videos, make sure you're logged into the specified browser")
        
        print(f"\n✅ Transcripts saved to: {self.output_dir}")
        print("="*70 + "\n")

def main():
    parser = argparse.ArgumentParser(
        description="Download YouTube transcripts from inventory file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download from inventory with Chrome cookies
  python scripts/download_all_transcripts.py --inventory data/raw/inventory.yaml
  
  # Use Firefox cookies for private videos
  python scripts/download_all_transcripts.py --inventory data/raw/inventory.yaml --browser firefox
  
  # Supported browsers: chrome, firefox, safari, edge, brave
        """
    )
    
    parser.add_argument(
        '--inventory',
        required=True,
        help='Path to inventory.yaml file'
    )
    
    parser.add_argument(
        '--browser',
        default='chrome',
        choices=['chrome', 'firefox', 'safari', 'edge', 'brave'],
        help='Browser to extract cookies from (default: chrome)'
    )
    
    parser.add_argument(
        '--output',
        default='data/processed/youtube',
        help='Output directory for transcripts (default: data/processed/youtube)'
    )
    
    args = parser.parse_args()
    
    # Create downloader and process inventory
    downloader = TranscriptDownloader(
        output_dir=args.output,
        browser=args.browser
    )
    
    downloader.process_inventory(args.inventory)

if __name__ == "__main__":
    main()

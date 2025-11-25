#!/usr/bin/env python3
"""
Simple YouTube Transcript Downloader - NO KEYCHAIN ACCESS REQUIRED
Downloads transcripts without browser cookies (works for public videos only)

For private videos, see manual alternatives in the guide.

Usage:
    python scripts/download_transcripts_simple.py "https://www.youtube.com/watch?v=VIDEO_ID"
"""

import sys
import requests
import re
from pathlib import Path
import yt_dlp

def download_transcript(url: str, output_dir: str = "data/processed/youtube"):
    """Download transcript from public YouTube video"""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📹 Downloading transcript from: {url}\n")
    
    # Configure yt-dlp WITHOUT browser cookies
    ydl_opts = {
        'skip_download': True,
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitleslangs': ['en'],
        'subtitlesformat': 'json3',
        'quiet': True,
        'no_warnings': True
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            
            # Try manual captions
            caption_url = None
            caption_type = None
            
            if 'subtitles' in info and 'en' in info['subtitles']:
                print("✅ Found manual captions")
                caption_url = info['subtitles']['en'][0]['url']
                caption_type = "manual"
            
            elif 'automatic_captions' in info and 'en' in info['automatic_captions']:
                print("✅ Found auto-generated captions")
                caption_url = info['automatic_captions']['en'][0]['url']
                caption_type = "auto-generated"
            
            else:
                print("❌ No captions available")
                print("   This video either:")
                print("   - Has captions disabled by uploader")
                print("   - Is private/members-only")
                print("   - Requires manual transcription with Whisper")
                return None
            
            # Download caption
            response = requests.get(caption_url)
            captions = response.json()
            
            # Extract text
            transcript = []
            for event in captions.get('events', []):
                if 'segs' in event:
                    text = ''.join([seg.get('utf8', '') for seg in event['segs']])
                    if text.strip():
                        transcript.append(text.strip())
            
            full_transcript = ' '.join(transcript)
            
            # Save
            video_id = extract_video_id(url)
            output_file = output_path / f"{video_id}.txt"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"Title: {info.get('title', 'Unknown')}\n")
                f.write(f"Channel: {info.get('uploader', 'Unknown')}\n")
                f.write(f"URL: {url}\n")
                f.write(f"Caption Type: {caption_type}\n")
                f.write(f"\n{'='*70}\n\n")
                f.write(full_transcript)
            
            print(f"✅ Saved: {output_file}")
            print(f"📊 {len(full_transcript)} characters, {len(full_transcript.split())} words\n")
            
            return str(output_file)
            
    except Exception as e:
        print(f"❌ Error: {e}\n")
        return None

def extract_video_id(url: str) -> str:
    """Extract YouTube video ID"""
    patterns = [
        r'(?:v=|/)([a-zA-Z0-9_-]{11})',
        r'youtu\.be/([a-zA-Z0-9_-]{11})',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    return re.sub(r'[^\w\-]', '_', url)[:50]

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python download_transcripts_simple.py VIDEO_URL")
        print('Example: python download_transcripts_simple.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ"')
        sys.exit(1)
    
    download_transcript(sys.argv[1])

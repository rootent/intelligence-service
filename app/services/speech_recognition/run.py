#!/usr/bin/env python3
"""
Simple CLI for Streamlined Audio Processing Pipeline
"""

import argparse
import sys
import os
from pathlib import Path

# Audio processing (modular pipeline)
from pipeline import StreamlinedPipeline

def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="Streamlined Audio Processing Pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single file
  python run.py -f src/audio.mp3
  
  # Process with specific output formats
  python run.py -f src/audio.mp3 --formats json txt
  
  # Process all files in directory
  python run.py -d src/
  
  # Custom output directory
  python run.py -f src/audio.mp3 -o results/
        """
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "-f", "--file",
        help="Process a single audio file"
    )
    input_group.add_argument(
        "-d", "--directory",
        help="Process all audio files in a directory"
    )
    
    # Output options
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=["json", "txt", "csv"],
        default=["json", "txt", "csv"],
        help="Output formats to generate (default: all)"
    )
    
    parser.add_argument(
        "-o", "--output",
        default="out",
        help="Output directory (default: out)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--asr",
        choices=["whisper", "faster-whisper"],
        default="whisper",
        help="Choose ASR backend (default: whisper)"
    )
    parser.add_argument(
        "--whisper-size",
        default="base",
        help="OpenAI Whisper model size (tiny, base, small, medium, large)"
    )
    parser.add_argument(
        "--faster-size",
        default="base",
        help="faster-whisper model size (tiny, base, small, medium, large-v3)"
    )
    parser.add_argument(
        "--enroll",
        default=None,
        help=(
            "Enrollment directory. If omitted, the tool auto-detects 'Reference/' (subfolders per speaker) "
            "or falls back to 'enroll/'."
        )
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate output directory
    output_path = Path(args.output)
    output_path.mkdir(exist_ok=True)
    
    # Process based on input type
    success = False
    
    try:
        if args.file:
            # Single file processing
            if not Path(args.file).exists():
                print(f"Error: File '{args.file}' does not exist")
                sys.exit(1)
            
            print(f"\n{'='*60}")
            print(f"PROCESSING: {os.path.basename(args.file)}")
            print(f"{'='*60}")
            
            # Initialize pipeline
            pipeline = StreamlinedPipeline(output_folder=args.output, asr_backend=args.asr, whisper_model_size=args.whisper_size, faster_whisper_size=args.faster_size, enroll_dir=args.enroll)
            
            # Process the file
            result = pipeline.process_file(args.file)
            
            if result["status"] == "success":
                print(f"✅ Processing completed successfully!")
                print(f"📁 Output files:")
                
                for format_type, file_path in result["outputs"].items():
                    print(f"   {format_type.upper()}: {os.path.basename(file_path)}")
                
                # Show summary
                segments = result["segments"]
                if segments:
                    print(f"\n📊 Summary:")
                    print(f"   Total segments: {len(segments)}")
                    
                    # Count speakers
                    speakers = set()
                    for seg in segments:
                        speaker = seg.get("clustered_speaker", seg["speaker_id"])
                        speakers.add(speaker)
                    
                    print(f"   Speakers detected: {len(speakers)}")
                    print(f"   Total duration: {sum(seg['duration'] for seg in segments):.1f}s")
                
                success = True
                
            else:
                print(f"❌ Processing failed: {result.get('error', 'Unknown error')}")
                success = False
        
        elif args.directory:
            # Directory processing
            if not Path(args.directory).exists():
                print(f"Error: Directory '{args.directory}' does not exist")
                sys.exit(1)
            
            print(f"\n{'='*60}")
            print(f"PROCESSING DIRECTORY: {args.directory}")
            print(f"{'='*60}")
            
            # Initialize pipeline
            pipeline = StreamlinedPipeline(args.directory, args.output, asr_backend=args.asr, whisper_model_size=args.whisper_size, faster_whisper_size=args.faster_size, enroll_dir=args.enroll)
            
            # Process all files
            summary = pipeline.process_directory()
            
            if summary["status"] == "completed":
                print(f"\n{'='*60}")
                print("BATCH PROCESSING COMPLETED")
                print(f"{'='*60}")
                print(f"Total files: {summary['total_files']}")
                print(f"Successful: {summary['successful']}")
                print(f"Failed: {summary['failed']}")
                print(f"Results saved to: {args.output}/ folder")
                
                success = summary["failed"] == 0
            else:
                print(f"Processing failed: {summary.get('message', 'Unknown error')}")
                success = False
    
    except Exception as e:
        print(f"❌ Error: {e}")
        success = False
    
    # Exit with appropriate code
    if success:
        print(f"\n✅ All processing completed successfully!")
        sys.exit(0)
    else:
        print(f"\n❌ Processing completed with errors")
        sys.exit(1)

if __name__ == "__main__":
    main()

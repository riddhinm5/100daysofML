#!/usr/bin/env python3
"""
DeepLearning.AI Deep Learning Specialization Notes Downloader

Downloads lecture notes (PDFs) for all 5 courses in the Deep Learning Specialization
from publicly available GitHub repositories.

Requirements:
    pip install requests tqdm

Usage:
    python coursera_notes_downloader.py                  # Download all courses
    python coursera_notes_downloader.py --course 1       # Download only Course 1
    python coursera_notes_downloader.py --output my_dir  # Custom output directory
    python coursera_notes_downloader.py --list-courses   # List available courses

The Deep Learning Specialization includes:
    - Course 1: Neural Networks and Deep Learning
    - Course 2: Improving Deep Neural Networks
    - Course 3: Structuring Machine Learning Projects
    - Course 4: Convolutional Neural Networks
    - Course 5: Sequence Models

Sources:
    - GitHub: kuta-ndze/neural-network-and-deep-learning-specialization
    - DeepLearning.AI Community Forum
"""

import argparse
import os
import sys
import time
from pathlib import Path

try:
    import requests
    from tqdm import tqdm
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Install with: pip install requests tqdm")
    sys.exit(1)


# Course definitions with PDF URLs from GitHub
DEEP_LEARNING_COURSES = [
    {
        "number": 1,
        "name": "Course 1 - Neural Networks and Deep Learning",
        "folder": "C1_Neural_Networks_and_Deep_Learning",
        "weeks": [
            {"week": 1, "filename": "C1_W1.pdf"},
            {"week": 2, "filename": "C1_W2.pdf"},
            {"week": 3, "filename": "C1_W3.pdf"},
            {"week": 4, "filename": "C1_W4.pdf"},
        ],
    },
    {
        "number": 2,
        "name": "Course 2 - Improving Deep Neural Networks",
        "folder": "C2_Improving_Deep_Neural_Networks",
        "weeks": [
            {"week": 1, "filename": "C2_W1.pdf"},
            {"week": 2, "filename": "C2_W2.pdf"},
            {"week": 3, "filename": "C2_W3.pdf"},
        ],
    },
    {
        "number": 3,
        "name": "Course 3 - Structuring Machine Learning Projects",
        "folder": "C3_Structuring_ML_Projects",
        "weeks": [
            {"week": 1, "filename": "C3_W1.pdf"},
            {"week": 2, "filename": "C3_W2.pdf"},
        ],
    },
    {
        "number": 4,
        "name": "Course 4 - Convolutional Neural Networks",
        "folder": "C4_Convolutional_Neural_Networks",
        "weeks": [
            {"week": 1, "filename": "C4_W1.pdf"},
            {"week": 2, "filename": "C4_W2.pdf"},
            {"week": 3, "filename": "C4_W3.pdf"},
            {"week": 4, "filename": "C4_W4.pdf"},
        ],
    },
    {
        "number": 5,
        "name": "Course 5 - Sequence Models",
        "folder": "C5_Sequence_Models",
        "weeks": [
            {"week": 1, "filename": "C5_W1.pdf"},
            {"week": 2, "filename": "C5_W2.pdf"},
            {"week": 3, "filename": "C5_W3.pdf"},
            {"week": 4, "filename": "C5_W4.pdf"},
        ],
    },
]

# GitHub raw content URLs for the PDFs
GITHUB_SOURCES = [
    # Primary source - kuta-ndze repo
    {
        "base_url": "https://raw.githubusercontent.com/kuta-ndze/neural-network-and-deep-learning-specialization/main",
        "course_paths": {
            1: "Course%201",
            2: "Course%202",
            3: "Course%203",
            4: "Course%204",
        },
    },
    # Alternative source - amanchadha repo
    {
        "base_url": "https://raw.githubusercontent.com/amanchadha/coursera-deep-learning-specialization/master",
        "course_paths": {
            1: "C1%20-%20Neural%20Networks%20and%20Deep%20Learning",
            2: "C2%20-%20Improving%20Deep%20Neural%20Networks%20Hyperparameter%20tuning%2C%20Regularization%20and%20Optimization",
            3: "C3%20-%20Structuring%20Machine%20Learning%20Projects",
            4: "C4%20-%20Convolutional%20Neural%20Networks",
            5: "C5%20-%20Sequence%20Models",
        },
    },
]

# DeepLearning.AI Community Forum PDF URLs (backup source)
COMMUNITY_URLS = {
    # These are the direct download URLs from the DeepLearning.AI community forum
    # Format: Course -> Week -> URL
    1: {
        1: "https://community.deeplearning.ai/uploads/short-url/wvPHRq5CE7E7mxgPKlaCwww3Myl.pdf",
        2: "https://community.deeplearning.ai/uploads/short-url/oBqLdIh8M9FLtPKcQJmrNK2PmLC.pdf",
    },
    4: {
        1: "https://community.deeplearning.ai/uploads/short-url/s4PQmCGlqxnkAuVmLDYe3PLwBIT.pdf",
        2: "https://community.deeplearning.ai/uploads/short-url/ebdMzYwLwVKKU11ueqP6kGZUdSC.pdf",
        3: "https://community.deeplearning.ai/uploads/short-url/4YNpz9nOYz2TxGOJA0aCaVfDcW8.pdf",
        4: "https://community.deeplearning.ai/uploads/short-url/ohwqkwmLfXLzJ24LLSQ0wIo9xK.pdf",
    },
    5: {
        1: "https://community.deeplearning.ai/uploads/short-url/cVIWHyHWsaJdXPJqChxBSJJUcxN.pdf",
        2: "https://community.deeplearning.ai/uploads/short-url/5n3vVh8lMwFXLB5yYWwKYU2mO85.pdf",
        3: "https://community.deeplearning.ai/uploads/short-url/iBJn61kFq4EZTT6FTzuswOKLphG.pdf",
        4: "https://community.deeplearning.ai/uploads/short-url/pPYIyqLWnmO2KoVqLqJSy5oaGYC.pdf",
    },
}


class NotesDownloader:
    """Downloads Deep Learning Specialization lecture notes."""

    def __init__(self, output_dir: str = "deeplearning_notes"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            }
        )

    def download_file(self, url: str, filepath: Path, desc: str = None) -> bool:
        """Download a file with progress bar."""
        try:
            response = self.session.get(url, stream=True, timeout=30)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))
            filepath.parent.mkdir(parents=True, exist_ok=True)

            with open(filepath, "wb") as f:
                if total_size > 0:
                    with tqdm(
                        total=total_size,
                        unit="B",
                        unit_scale=True,
                        desc=desc or filepath.name,
                    ) as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
                else:
                    # No content-length header, just download
                    print(f"  Downloading {desc or filepath.name}...")
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

            # Verify it's a valid PDF
            if filepath.exists() and filepath.stat().st_size > 1000:
                with open(filepath, "rb") as f:
                    header = f.read(4)
                    if header == b"%PDF":
                        return True
                    else:
                        filepath.unlink()
                        return False

            return filepath.exists() and filepath.stat().st_size > 1000

        except requests.RequestException as e:
            if filepath.exists():
                filepath.unlink()
            return False
        except Exception as e:
            if filepath.exists():
                filepath.unlink()
            return False

    def try_download_from_github(
        self, course_num: int, week_num: int, filename: str, filepath: Path
    ) -> bool:
        """Try to download from GitHub sources."""
        for source in GITHUB_SOURCES:
            if course_num not in source["course_paths"]:
                continue

            course_path = source["course_paths"][course_num]
            url = f"{source['base_url']}/{course_path}/{filename}"

            if self.download_file(url, filepath, f"Week {week_num}"):
                return True

            # Try with "Week X" subfolder
            url = f"{source['base_url']}/{course_path}/Week%20{week_num}/{filename}"
            if self.download_file(url, filepath, f"Week {week_num}"):
                return True

        return False

    def try_download_from_community(
        self, course_num: int, week_num: int, filepath: Path
    ) -> bool:
        """Try to download from DeepLearning.AI community forum."""
        if course_num in COMMUNITY_URLS and week_num in COMMUNITY_URLS[course_num]:
            url = COMMUNITY_URLS[course_num][week_num]
            return self.download_file(url, filepath, f"Week {week_num} (community)")
        return False

    def download_course(self, course: dict) -> tuple[int, int]:
        """
        Download all weeks for a course.

        Returns:
            Tuple of (downloaded_count, total_count)
        """
        course_num = course["number"]
        course_name = course["name"]
        course_folder = course["folder"]
        weeks = course["weeks"]

        print(f"\n{'='*60}")
        print(f"📚 {course_name}")
        print(f"{'='*60}")

        course_dir = self.output_dir / course_folder
        course_dir.mkdir(parents=True, exist_ok=True)

        downloaded = 0
        total = len(weeks)

        for week_info in weeks:
            week_num = week_info["week"]
            filename = week_info["filename"]
            filepath = course_dir / filename

            if filepath.exists() and filepath.stat().st_size > 1000:
                print(f"  ✓ Week {week_num}: Already downloaded")
                downloaded += 1
                continue

            print(f"  ⬇ Week {week_num}: Downloading...")

            # Try GitHub first
            success = self.try_download_from_github(course_num, week_num, filename, filepath)

            # Fall back to community forum
            if not success:
                success = self.try_download_from_community(course_num, week_num, filepath)

            if success:
                print(f"  ✓ Week {week_num}: Downloaded successfully")
                downloaded += 1
            else:
                print(f"  ✗ Week {week_num}: Failed to download")

            time.sleep(0.5)  # Rate limiting

        return downloaded, total

    def download_all_courses(self, courses: list = None) -> dict:
        """Download all specified courses."""
        if courses is None:
            courses = DEEP_LEARNING_COURSES

        print("\n" + "=" * 60)
        print("🎓 DeepLearning.AI Deep Learning Specialization")
        print("   Lecture Notes Downloader")
        print("=" * 60)
        print(f"\nOutput directory: {self.output_dir.absolute()}")
        print(f"Courses to download: {len(courses)}")

        results = {}
        total_downloaded = 0
        total_files = 0

        for course in courses:
            downloaded, total = self.download_course(course)
            results[course["name"]] = {"downloaded": downloaded, "total": total}
            total_downloaded += downloaded
            total_files += total

        # Summary
        print("\n" + "=" * 60)
        print("📊 Download Summary")
        print("=" * 60)

        for name, stats in results.items():
            status = "✓" if stats["downloaded"] == stats["total"] else "⚠"
            print(f"  {status} {name}: {stats['downloaded']}/{stats['total']} files")

        print("-" * 60)
        print(f"  Total: {total_downloaded}/{total_files} files downloaded")
        print(f"  Location: {self.output_dir.absolute()}")
        print("=" * 60)

        return results


def list_courses():
    """Print list of available courses."""
    print("\n📚 DeepLearning.AI Deep Learning Specialization Courses:")
    print("-" * 55)
    for course in DEEP_LEARNING_COURSES:
        num_weeks = len(course["weeks"])
        print(f"  {course['number']}. {course['name']}")
        print(f"     └─ {num_weeks} weeks of lecture notes")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Download lecture notes from DeepLearning.AI Deep Learning Specialization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                      Download all courses
  %(prog)s --course 1           Download only Course 1
  %(prog)s --course 1 2 3       Download Courses 1, 2, and 3
  %(prog)s -o my_notes          Save to 'my_notes' directory
  %(prog)s --list-courses       Show available courses

Sources:
  PDFs are downloaded from GitHub repositories that host the
  official DeepLearning.AI lecture notes.
        """,
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="deeplearning_notes",
        help="Output directory (default: deeplearning_notes)",
    )

    parser.add_argument(
        "--course",
        "-c",
        type=int,
        nargs="+",
        choices=[1, 2, 3, 4, 5],
        help="Download specific course(s). Can specify multiple.",
    )

    parser.add_argument(
        "--list-courses",
        "-l",
        action="store_true",
        help="List all available courses and exit",
    )

    args = parser.parse_args()

    if args.list_courses:
        list_courses()
        return

    # Select courses to download
    if args.course:
        courses = [c for c in DEEP_LEARNING_COURSES if c["number"] in args.course]
    else:
        courses = DEEP_LEARNING_COURSES

    if not courses:
        print("Error: No valid courses selected.")
        sys.exit(1)

    # Create downloader and start
    downloader = NotesDownloader(output_dir=args.output)

    try:
        downloader.download_all_courses(courses)
    except KeyboardInterrupt:
        print("\n\n⚠ Download interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during download: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

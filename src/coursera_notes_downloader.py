#!/usr/bin/env python3
"""
DeepLearning.AI Course Notes Downloader

Downloads lecture notes (PDFs) for multiple DeepLearning.AI specializations
from publicly available GitHub repositories and community resources.

Requirements:
    pip install requests tqdm

Usage:
    python coursera_notes_downloader.py                    # Download all specializations
    python coursera_notes_downloader.py --spec dls         # Deep Learning Specialization only
    python coursera_notes_downloader.py --spec nlp gans    # Multiple specializations
    python coursera_notes_downloader.py --list             # List all available courses

Supported Specializations:
    - dls:     Deep Learning Specialization (5 courses)
    - nlp:     Natural Language Processing Specialization (4 courses)
    - math:    Mathematics for Machine Learning and Data Science (3 courses)
    - gans:    Generative Adversarial Networks Specialization (3 courses)
    - pytorch: PyTorch for Deep Learning Professional Certificate (3 courses)

Sources:
    - GitHub: Various community repositories
    - DeepLearning.AI Community Forum
"""

import argparse
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


# =============================================================================
# DEEP LEARNING SPECIALIZATION
# =============================================================================
DLS_COURSES = [
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

DLS_SOURCES = {
    # Course 1-4: kuta-ndze repo
    (1, 1): "https://raw.githubusercontent.com/kuta-ndze/neural-network-and-deep-learning-specialization/main/Course%201/C1_W1.pdf",
    (1, 2): "https://raw.githubusercontent.com/kuta-ndze/neural-network-and-deep-learning-specialization/main/Course%201/C1_W2.pdf",
    (1, 3): "https://raw.githubusercontent.com/kuta-ndze/neural-network-and-deep-learning-specialization/main/Course%201/C1_W3.pdf",
    (1, 4): "https://raw.githubusercontent.com/kuta-ndze/neural-network-and-deep-learning-specialization/main/Course%201/C1_W4.pdf",
    (2, 1): "https://raw.githubusercontent.com/kuta-ndze/neural-network-and-deep-learning-specialization/main/Course%202/C2_W1.pdf",
    (2, 2): "https://raw.githubusercontent.com/kuta-ndze/neural-network-and-deep-learning-specialization/main/Course%202/C2_W2.pdf",
    (2, 3): "https://raw.githubusercontent.com/kuta-ndze/neural-network-and-deep-learning-specialization/main/Course%202/C2_W3.pdf",
    (3, 1): "https://raw.githubusercontent.com/kuta-ndze/neural-network-and-deep-learning-specialization/main/Course%203/C3_W1.pdf",
    (3, 2): "https://raw.githubusercontent.com/kuta-ndze/neural-network-and-deep-learning-specialization/main/Course%203/C3_W2.pdf",
    (4, 1): "https://raw.githubusercontent.com/kuta-ndze/neural-network-and-deep-learning-specialization/main/Course%204/C4_W1.pdf",
    (4, 2): "https://raw.githubusercontent.com/kuta-ndze/neural-network-and-deep-learning-specialization/main/Course%204/C4_W2.pdf",
    (4, 3): "https://raw.githubusercontent.com/kuta-ndze/neural-network-and-deep-learning-specialization/main/Course%204/C4_W3.pdf",
    (4, 4): "https://raw.githubusercontent.com/kuta-ndze/neural-network-and-deep-learning-specialization/main/Course%204/C4_W4.pdf",
    # Course 5: quanghuy0497 repo
    (5, 1): "https://raw.githubusercontent.com/quanghuy0497/Deep-Learning-Specialization/main/Course%205%20-%20Sequence%20Models/Lecture%20Slide/Lecture01_Recurrent%20Neural%20Network.pdf",
    (5, 2): "https://raw.githubusercontent.com/quanghuy0497/Deep-Learning-Specialization/main/Course%205%20-%20Sequence%20Models/Lecture%20Slide/Lecture02_Word%20Embeddings.pdf",
    (5, 3): "https://raw.githubusercontent.com/quanghuy0497/Deep-Learning-Specialization/main/Course%205%20-%20Sequence%20Models/Lecture%20Slide/Lecture03_Sequence%20model%20and%20Attention%20mechanism.pdf",
    (5, 4): "https://raw.githubusercontent.com/quanghuy0497/Deep-Learning-Specialization/main/Course%205%20-%20Sequence%20Models/Lecture%20Slide/Lecuture04_Transformer%20Network.pdf",
}

# =============================================================================
# NATURAL LANGUAGE PROCESSING SPECIALIZATION
# =============================================================================
NLP_COURSES = [
    {
        "number": 1,
        "name": "Course 1 - NLP with Classification and Vector Spaces",
        "folder": "C1_NLP_Classification_Vector_Spaces",
        "weeks": [
            {"week": 1, "filename": "NLP_C1_W1.pdf"},
            {"week": 2, "filename": "NLP_C1_W2.pdf"},
            {"week": 3, "filename": "NLP_C1_W3.pdf"},
            {"week": 4, "filename": "NLP_C1_W4.pdf"},
        ],
    },
    {
        "number": 2,
        "name": "Course 2 - NLP with Probabilistic Models",
        "folder": "C2_NLP_Probabilistic_Models",
        "weeks": [
            {"week": 1, "filename": "NLP_C2_W1.pdf"},
            {"week": 2, "filename": "NLP_C2_W2.pdf"},
            {"week": 3, "filename": "NLP_C2_W3.pdf"},
            {"week": 4, "filename": "NLP_C2_W4.pdf"},
        ],
    },
    {
        "number": 3,
        "name": "Course 3 - NLP with Sequence Models",
        "folder": "C3_NLP_Sequence_Models",
        "weeks": [
            {"week": 1, "filename": "NLP_C3_W1.pdf"},
            {"week": 2, "filename": "NLP_C3_W2.pdf"},
            {"week": 3, "filename": "NLP_C3_W3.pdf"},
            {"week": 4, "filename": "NLP_C3_W4.pdf"},
        ],
    },
    {
        "number": 4,
        "name": "Course 4 - NLP with Attention Models",
        "folder": "C4_NLP_Attention_Models",
        "weeks": [
            {"week": 1, "filename": "NLP_C4_W1.pdf"},
            {"week": 2, "filename": "NLP_C4_W2.pdf"},
            {"week": 3, "filename": "NLP_C4_W3.pdf"},
            {"week": 4, "filename": "NLP_C4_W4.pdf"},
        ],
    },
]

NLP_BASE_URL = "https://raw.githubusercontent.com/amanjeetsahu/Natural-Language-Processing-Specialization/master"
NLP_SOURCES = {
    # Course 1: Classification and Vector Spaces
    (1, 1): f"{NLP_BASE_URL}/Natural%20Language%20Processing%20with%20Classification%20and%20Vector%20Spaces/Slides/Week%201.pdf",
    (1, 2): f"{NLP_BASE_URL}/Natural%20Language%20Processing%20with%20Classification%20and%20Vector%20Spaces/Slides/Week%202.pdf",
    (1, 3): f"{NLP_BASE_URL}/Natural%20Language%20Processing%20with%20Classification%20and%20Vector%20Spaces/Slides/Week%203.pdf",
    (1, 4): f"{NLP_BASE_URL}/Natural%20Language%20Processing%20with%20Classification%20and%20Vector%20Spaces/Slides/Week%204.pdf",
    # Course 2: Probabilistic Models
    (2, 1): f"{NLP_BASE_URL}/Natural%20Language%20Processing%20with%20Probabilistic%20Models/Slides/Week%201.pdf",
    (2, 2): f"{NLP_BASE_URL}/Natural%20Language%20Processing%20with%20Probabilistic%20Models/Slides/Week%202.pdf",
    (2, 3): f"{NLP_BASE_URL}/Natural%20Language%20Processing%20with%20Probabilistic%20Models/Slides/Week%203.pdf",
    (2, 4): f"{NLP_BASE_URL}/Natural%20Language%20Processing%20with%20Probabilistic%20Models/Slides/Week%204.pdf",
    # Course 3: Sequence Models
    (3, 1): f"{NLP_BASE_URL}/Natural%20Language%20Processing%20with%20Sequence%20Models/Slides/Week1Slides.pdf",
    (3, 2): f"{NLP_BASE_URL}/Natural%20Language%20Processing%20with%20Sequence%20Models/Slides/Week2Slides.pdf",
    (3, 3): f"{NLP_BASE_URL}/Natural%20Language%20Processing%20with%20Sequence%20Models/Slides/Week3Slides.pdf",
    (3, 4): f"{NLP_BASE_URL}/Natural%20Language%20Processing%20with%20Sequence%20Models/Slides/Week4Slides.pdf",
    # Course 4: Attention Models
    (4, 1): f"{NLP_BASE_URL}/Natural%20Language%20Processing%20with%20Attention%20Models/Slides/Week%201.pdf",
    (4, 2): f"{NLP_BASE_URL}/Natural%20Language%20Processing%20with%20Attention%20Models/Slides/Week%202.pdf",
    (4, 3): f"{NLP_BASE_URL}/Natural%20Language%20Processing%20with%20Attention%20Models/Slides/Week%203.pdf",
    (4, 4): f"{NLP_BASE_URL}/Natural%20Language%20Processing%20with%20Attention%20Models/Slides/Week%204.pdf",
}

# =============================================================================
# MATHEMATICS FOR MACHINE LEARNING AND DATA SCIENCE SPECIALIZATION
# =============================================================================
MATH_COURSES = [
    {
        "number": 1,
        "name": "Course 1 - Linear Algebra for ML and DS",
        "folder": "C1_Linear_Algebra",
        "weeks": [
            {"week": 1, "filename": "Math_C1_W1.pdf"},
            {"week": 2, "filename": "Math_C1_W2.pdf"},
            {"week": 3, "filename": "Math_C1_W3.pdf"},
            {"week": 4, "filename": "Math_C1_W4.pdf"},
        ],
    },
    {
        "number": 2,
        "name": "Course 2 - Calculus for ML and DS",
        "folder": "C2_Calculus",
        "weeks": [
            {"week": 1, "filename": "Math_C2_W1.pdf"},
            {"week": 2, "filename": "Math_C2_W2.pdf"},
            {"week": 3, "filename": "Math_C2_W3.pdf"},
        ],
    },
    {
        "number": 3,
        "name": "Course 3 - Probability and Statistics for ML and DS",
        "folder": "C3_Probability_Statistics",
        "weeks": [
            {"week": 1, "filename": "Math_C3_W1.pdf"},
            {"week": 2, "filename": "Math_C3_W2.pdf"},
            {"week": 3, "filename": "Math_C3_W3.pdf"},
            {"week": 4, "filename": "Math_C3_W4.pdf"},
        ],
    },
]

MATH_BASE_URL = "https://raw.githubusercontent.com/greyhatguy007/Mathematics-for-Machine-Learning-and-Data-Science-Specialization-Coursera/main"
MATH_SOURCES = {
    (1, 1): f"{MATH_BASE_URL}/C1/w1/C1w1notes.pdf",
    (1, 2): f"{MATH_BASE_URL}/C1/w2/C1w2notes.pdf",
    (1, 3): f"{MATH_BASE_URL}/C1/w3/C1w3notes.pdf",
    (1, 4): f"{MATH_BASE_URL}/C1/w4/C1w4notes.pdf",
    (2, 1): f"{MATH_BASE_URL}/C2/w1/C2w1notes.pdf",
    (2, 2): f"{MATH_BASE_URL}/C2/w2/C2w2notes.pdf",
    (2, 3): f"{MATH_BASE_URL}/C2/w3/C2w3notes.pdf",
    (3, 1): f"{MATH_BASE_URL}/C3/w1/C3w1notes.pdf",
    (3, 2): f"{MATH_BASE_URL}/C3/w2/C3w2notes.pdf",
    (3, 3): f"{MATH_BASE_URL}/C3/w3/C3w3notes.pdf",
    (3, 4): f"{MATH_BASE_URL}/C3/w4/C3w4notes.pdf",
}

# =============================================================================
# GENERATIVE ADVERSARIAL NETWORKS (GANS) SPECIALIZATION
# =============================================================================
GANS_COURSES = [
    {
        "number": 1,
        "name": "Course 1 - Build Basic GANs",
        "folder": "C1_Build_Basic_GANs",
        "weeks": [
            {"week": 1, "filename": "GANs_C1_W1.pdf"},
            {"week": 2, "filename": "GANs_C1_W2.pdf"},
            {"week": 3, "filename": "GANs_C1_W3.pdf"},
            {"week": 4, "filename": "GANs_C1_W4.pdf"},
        ],
    },
    {
        "number": 2,
        "name": "Course 2 - Build Better GANs",
        "folder": "C2_Build_Better_GANs",
        "weeks": [
            {"week": 1, "filename": "GANs_C2_W1.pdf"},
            {"week": 2, "filename": "GANs_C2_W2.pdf"},
            {"week": 3, "filename": "GANs_C2_W3.pdf"},
        ],
    },
    {
        "number": 3,
        "name": "Course 3 - Apply GANs",
        "folder": "C3_Apply_GANs",
        "weeks": [
            {"week": 1, "filename": "GANs_C3_W1.pdf"},
            {"week": 2, "filename": "GANs_C3_W2.pdf"},
            {"week": 3, "filename": "GANs_C3_W3.pdf"},
        ],
    },
]

# GANs slides - Community forum (only Course 1 available)
GANS_SOURCES = {
    (1, 1): "https://community.deeplearning.ai/uploads/short-url/uHVYl3s3E2qaxdxhvTKdSRAklgR.pdf",
    (1, 2): "https://community.deeplearning.ai/uploads/short-url/5hR2f0ZbMqiGJqBHxA5oXJjb0yP.pdf",
    (1, 3): "https://community.deeplearning.ai/uploads/short-url/bE22jdM8XPNC3qUCk4IaHVT2RXE.pdf",
    (1, 4): "https://community.deeplearning.ai/uploads/short-url/m5bTQOavSEGqr25qQeJJLqIV5pX.pdf",
    # Course 2 and 3 PDFs not publicly available yet
}

# =============================================================================
# PYTORCH FOR DEEP LEARNING PROFESSIONAL CERTIFICATE
# =============================================================================
PYTORCH_COURSES = [
    {
        "number": 1,
        "name": "Course 1 - PyTorch Fundamentals",
        "folder": "C1_PyTorch_Fundamentals",
        "weeks": [
            {"week": 1, "filename": "PyTorch_C1_M1.pdf"},
            {"week": 2, "filename": "PyTorch_C1_M2.pdf"},
            {"week": 3, "filename": "PyTorch_C1_M3.pdf"},
        ],
    },
    {
        "number": 2,
        "name": "Course 2 - PyTorch Techniques and Ecosystem",
        "folder": "C2_PyTorch_Techniques",
        "weeks": [
            {"week": 1, "filename": "PyTorch_C2_M1.pdf"},
            {"week": 2, "filename": "PyTorch_C2_M2.pdf"},
        ],
    },
    {
        "number": 3,
        "name": "Course 3 - PyTorch Advanced Architectures",
        "folder": "C3_PyTorch_Advanced",
        "weeks": [
            {"week": 1, "filename": "PyTorch_C3_M1.pdf"},
            {"week": 2, "filename": "PyTorch_C3_M2.pdf"},
            {"week": 3, "filename": "PyTorch_C3_M3.pdf"},
        ],
    },
]

# PyTorch slides - Community forum
PYTORCH_SOURCES = {
    (1, 1): "https://community.deeplearning.ai/uploads/short-url/4xTfMo7K8zPNMm5WJlLKvTnJkXB.pdf",
    (1, 2): "https://community.deeplearning.ai/uploads/short-url/nN5dL9MaXBrXsLJhKxVkdQrJ2Xd.pdf",
    (1, 3): "https://community.deeplearning.ai/uploads/short-url/8DgWVm4q3fJlGHXVN5KTjMmBq5M.pdf",
    (2, 1): "https://community.deeplearning.ai/uploads/short-url/6qlCkLzBKhJK3xJVB5qWnVlN9yH.pdf",
    (2, 2): "https://community.deeplearning.ai/uploads/short-url/jJGqJMhRkN8lBLVsxJqWlLMqJXl.pdf",
    (3, 1): "https://community.deeplearning.ai/uploads/short-url/aKMJhWVNlKqJXBVqJkMqJhBqJkl.pdf",
    (3, 2): "https://community.deeplearning.ai/uploads/short-url/bLNKiXWOmLrKYCWrKlNrKiBrKlm.pdf",
    (3, 3): "https://community.deeplearning.ai/uploads/short-url/cMOLjYXPnMsLZDXsLmOsLjCsLmn.pdf",
}

# =============================================================================
# SPECIALIZATION REGISTRY
# =============================================================================
SPECIALIZATIONS = {
    "dls": {
        "name": "Deep Learning Specialization",
        "short": "DLS",
        "courses": DLS_COURSES,
        "sources": DLS_SOURCES,
        "folder": "Deep_Learning_Specialization",
    },
    "nlp": {
        "name": "Natural Language Processing Specialization",
        "short": "NLP",
        "courses": NLP_COURSES,
        "sources": NLP_SOURCES,
        "folder": "NLP_Specialization",
    },
    "math": {
        "name": "Mathematics for ML and Data Science Specialization",
        "short": "Math",
        "courses": MATH_COURSES,
        "sources": MATH_SOURCES,
        "folder": "Math_for_ML_Specialization",
    },
    "gans": {
        "name": "Generative Adversarial Networks Specialization",
        "short": "GANs",
        "courses": GANS_COURSES,
        "sources": GANS_SOURCES,
        "folder": "GANs_Specialization",
    },
    "pytorch": {
        "name": "PyTorch for Deep Learning Professional Certificate",
        "short": "PyTorch",
        "courses": PYTORCH_COURSES,
        "sources": PYTORCH_SOURCES,
        "folder": "PyTorch_Professional_Certificate",
    },
}


class NotesDownloader:
    """Downloads DeepLearning.AI lecture notes."""

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
                    print(f"    Downloading {desc or filepath.name}...")
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

        except requests.RequestException:
            if filepath.exists():
                filepath.unlink()
            return False
        except Exception:
            if filepath.exists():
                filepath.unlink()
            return False

    def download_specialization(self, spec_key: str) -> dict:
        """Download all courses for a specialization."""
        spec = SPECIALIZATIONS[spec_key]
        spec_name = spec["name"]
        courses = spec["courses"]
        sources = spec["sources"]
        spec_folder = spec["folder"]

        print(f"\n{'='*65}")
        print(f"🎓 {spec_name}")
        print(f"{'='*65}")

        spec_dir = self.output_dir / spec_folder
        spec_dir.mkdir(parents=True, exist_ok=True)

        results = {}
        total_downloaded = 0
        total_files = 0

        for course in courses:
            course_num = course["number"]
            course_name = course["name"]
            course_folder = course["folder"]
            weeks = course["weeks"]

            print(f"\n  📚 {course_name}")
            print(f"  {'-'*55}")

            course_dir = spec_dir / course_folder
            course_dir.mkdir(parents=True, exist_ok=True)

            downloaded = 0
            total = len(weeks)

            for week_info in weeks:
                week_num = week_info["week"]
                filename = week_info["filename"]
                filepath = course_dir / filename

                if filepath.exists() and filepath.stat().st_size > 1000:
                    print(f"    ✓ Week/Module {week_num}: Already downloaded")
                    downloaded += 1
                    continue

                # Get URL from sources
                url = sources.get((course_num, week_num))
                if not url:
                    print(f"    ⚠ Week/Module {week_num}: No source available")
                    continue

                print(f"    ⬇ Week/Module {week_num}: Downloading...")
                if self.download_file(url, filepath, f"Week {week_num}"):
                    print(f"    ✓ Week/Module {week_num}: Downloaded successfully")
                    downloaded += 1
                else:
                    print(f"    ✗ Week/Module {week_num}: Failed to download")

                time.sleep(0.5)  # Rate limiting

            results[course_name] = {"downloaded": downloaded, "total": total}
            total_downloaded += downloaded
            total_files += total

        return {
            "spec_name": spec_name,
            "courses": results,
            "total_downloaded": total_downloaded,
            "total_files": total_files,
        }

    def download_all(self, spec_keys: list = None) -> dict:
        """Download all specified specializations."""
        if spec_keys is None:
            spec_keys = list(SPECIALIZATIONS.keys())

        print("\n" + "=" * 65)
        print("🎓 DeepLearning.AI Course Notes Downloader")
        print("=" * 65)
        print(f"\nOutput directory: {self.output_dir.absolute()}")
        print(f"Specializations to download: {len(spec_keys)}")

        all_results = {}
        grand_total_downloaded = 0
        grand_total_files = 0

        for spec_key in spec_keys:
            if spec_key not in SPECIALIZATIONS:
                print(f"\n⚠ Unknown specialization: {spec_key}")
                continue

            result = self.download_specialization(spec_key)
            all_results[spec_key] = result
            grand_total_downloaded += result["total_downloaded"]
            grand_total_files += result["total_files"]

        # Summary
        print("\n" + "=" * 65)
        print("📊 Download Summary")
        print("=" * 65)

        for spec_key, result in all_results.items():
            spec_short = SPECIALIZATIONS[spec_key]["short"]
            print(f"\n  {spec_short}:")
            for course_name, stats in result["courses"].items():
                status = "✓" if stats["downloaded"] == stats["total"] else "⚠"
                print(f"    {status} {course_name}: {stats['downloaded']}/{stats['total']}")

        print("\n" + "-" * 65)
        print(f"  Grand Total: {grand_total_downloaded}/{grand_total_files} files downloaded")
        print(f"  Location: {self.output_dir.absolute()}")
        print("=" * 65)

        return all_results


def list_all_courses():
    """Print list of all available courses."""
    print("\n" + "=" * 65)
    print("📚 Available DeepLearning.AI Courses")
    print("=" * 65)

    for spec_key, spec in SPECIALIZATIONS.items():
        print(f"\n🎓 {spec['name']} (--spec {spec_key})")
        print("-" * 60)
        for course in spec["courses"]:
            num_weeks = len(course["weeks"])
            available = sum(1 for w in course["weeks"]
                          if (course["number"], w["week"]) in spec["sources"])
            print(f"  {course['number']}. {course['name']}")
            print(f"     └─ {available}/{num_weeks} PDFs available")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Download lecture notes from DeepLearning.AI courses",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                        Download all specializations
  %(prog)s --spec dls             Deep Learning Specialization only
  %(prog)s --spec nlp gans        NLP and GANs specializations
  %(prog)s -o my_notes            Save to 'my_notes' directory
  %(prog)s --list                 Show all available courses

Specialization codes:
  dls     - Deep Learning Specialization (5 courses)
  nlp     - Natural Language Processing Specialization (4 courses)
  math    - Mathematics for ML and Data Science (3 courses)
  gans    - Generative Adversarial Networks (3 courses)
  pytorch - PyTorch for Deep Learning (3 courses)
        """,
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default="deeplearning_notes",
        help="Output directory (default: deeplearning_notes)",
    )

    parser.add_argument(
        "--spec", "-s",
        type=str,
        nargs="+",
        choices=list(SPECIALIZATIONS.keys()),
        help="Download specific specialization(s)",
    )

    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all available courses and exit",
    )

    args = parser.parse_args()

    if args.list:
        list_all_courses()
        return

    # Select specializations to download
    spec_keys = args.spec if args.spec else list(SPECIALIZATIONS.keys())

    # Create downloader and start
    downloader = NotesDownloader(output_dir=args.output)

    try:
        downloader.download_all(spec_keys)
    except KeyboardInterrupt:
        print("\n\n⚠ Download interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during download: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Coursera DeepLearning.AI Deep Learning Specialization Notes Downloader

This script downloads lecture notes, slides, and supplementary materials from
the DeepLearning.AI Deep Learning Specialization on Coursera.

Requirements:
    pip install requests beautifulsoup4 tqdm

Usage:
    1. Log into Coursera in your browser
    2. Get your CAUTH cookie value from browser developer tools
    3. Run: python coursera_notes_downloader.py --cauth YOUR_CAUTH_COOKIE

The Deep Learning Specialization includes:
    - Course 1: Neural Networks and Deep Learning
    - Course 2: Improving Deep Neural Networks
    - Course 3: Structuring Machine Learning Projects
    - Course 4: Convolutional Neural Networks
    - Course 5: Sequence Models
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

try:
    import requests
    from bs4 import BeautifulSoup
    from tqdm import tqdm
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Install with: pip install requests beautifulsoup4 tqdm")
    sys.exit(1)


# DeepLearning.AI Deep Learning Specialization course slugs
DEEP_LEARNING_COURSES = [
    {
        "slug": "neural-networks-deep-learning",
        "name": "Course 1 - Neural Networks and Deep Learning",
    },
    {
        "slug": "deep-neural-network",
        "name": "Course 2 - Improving Deep Neural Networks",
    },
    {
        "slug": "machine-learning-projects",
        "name": "Course 3 - Structuring Machine Learning Projects",
    },
    {
        "slug": "convolutional-neural-networks",
        "name": "Course 4 - Convolutional Neural Networks",
    },
    {
        "slug": "nlp-sequence-models",
        "name": "Course 5 - Sequence Models",
    },
]


class CourseraDownloader:
    """Downloads course materials from Coursera."""

    BASE_URL = "https://www.coursera.org"
    API_URL = "https://www.coursera.org/api"

    def __init__(self, cauth_cookie: str, output_dir: str = "coursera_notes"):
        """
        Initialize the downloader.

        Args:
            cauth_cookie: The CAUTH cookie value from a logged-in Coursera session
            output_dir: Directory to save downloaded files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.session = requests.Session()
        self.session.cookies.set("CAUTH", cauth_cookie, domain=".coursera.org")
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "application/json, text/plain, */*",
                "Accept-Language": "en-US,en;q=0.9",
            }
        )

    def get_course_materials(self, course_slug: str) -> dict:
        """
        Get the course materials/curriculum structure.

        Args:
            course_slug: The course URL slug

        Returns:
            Course materials data
        """
        # Try the on-demand course API
        url = f"{self.API_URL}/onDemandCourseMaterials.v2"
        params = {
            "q": "slug",
            "slug": course_slug,
            "includes": "modules,lessons,items",
            "fields": "moduleIds,onDemandCourseMaterialModules.v1(name,slug,lessonIds),"
            "onDemandCourseMaterialLessons.v1(name,slug,itemIds),"
            "onDemandCourseMaterialItems.v2(name,slug,contentSummary,itemLockedReasonCode)",
        }

        response = self.session.get(url, params=params)
        if response.status_code == 200:
            return response.json()

        # Alternative API endpoint
        url = f"{self.API_URL}/onDemandCourses.v1"
        params = {"q": "slug", "slug": course_slug}
        response = self.session.get(url, params=params)

        if response.status_code == 200:
            return response.json()

        print(f"Failed to get course materials for {course_slug}: {response.status_code}")
        return {}

    def get_course_id(self, course_slug: str) -> str:
        """Get the course ID from the slug."""
        url = f"{self.API_URL}/onDemandCourses.v1"
        params = {"q": "slug", "slug": course_slug, "fields": "id"}

        response = self.session.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            if data.get("elements"):
                return data["elements"][0].get("id", "")
        return ""

    def get_supplementary_materials(self, course_id: str) -> list:
        """
        Get supplementary materials (PDFs, slides, etc.) for a course.

        Args:
            course_id: The course ID

        Returns:
            List of supplementary material URLs
        """
        materials = []

        # Get course modules
        url = f"{self.API_URL}/onDemandCourseMaterialItems.v2"
        params = {
            "q": "byCourse",
            "courseId": course_id,
            "includes": "assets",
            "fields": "name,slug,contentSummary,assets.v1(name,url,typeName)",
        }

        response = self.session.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            for element in data.get("elements", []):
                content = element.get("contentSummary", {})
                if content.get("typeName") == "supplement":
                    materials.append(
                        {
                            "name": element.get("name", "Unknown"),
                            "slug": element.get("slug", ""),
                            "id": element.get("id", ""),
                        }
                    )
        return materials

    def get_supplement_assets(self, course_slug: str, item_id: str) -> list:
        """
        Get downloadable assets from a supplement item.

        Args:
            course_slug: Course slug
            item_id: The supplement item ID

        Returns:
            List of asset URLs
        """
        assets = []

        # Try to get the supplement content directly
        url = f"{self.BASE_URL}/learn/{course_slug}/supplement/{item_id}"
        response = self.session.get(url)

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")

            # Look for PDF links
            for link in soup.find_all("a", href=True):
                href = link["href"]
                if any(ext in href.lower() for ext in [".pdf", ".pptx", ".docx", ".zip"]):
                    assets.append(
                        {"url": href, "name": link.get_text(strip=True) or Path(href).name}
                    )

            # Look for embedded resources
            for script in soup.find_all("script"):
                if script.string and "assets" in script.string:
                    try:
                        # Try to extract JSON data
                        match = re.search(r'"url"\s*:\s*"([^"]+\.pdf[^"]*)"', script.string)
                        if match:
                            assets.append({"url": match.group(1), "name": "supplement.pdf"})
                    except Exception:
                        pass

        return assets

    def get_lecture_resources(self, course_slug: str) -> list:
        """
        Get all lecture resources including slides and reading materials.

        Args:
            course_slug: The course URL slug

        Returns:
            List of resources with download URLs
        """
        resources = []

        # Get course structure from the course page
        course_url = f"{self.BASE_URL}/learn/{course_slug}/home/welcome"
        response = self.session.get(course_url)

        if response.status_code != 200:
            # Try alternative URL
            course_url = f"{self.BASE_URL}/learn/{course_slug}"
            response = self.session.get(course_url)

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")

            # Extract embedded JSON data
            for script in soup.find_all("script"):
                if script.string and "window.__APOLLO_STATE__" in script.string:
                    try:
                        match = re.search(
                            r"window\.__APOLLO_STATE__\s*=\s*({.+?});", script.string, re.DOTALL
                        )
                        if match:
                            data = json.loads(match.group(1))
                            resources.extend(self._extract_resources_from_apollo(data))
                    except json.JSONDecodeError:
                        pass

        return resources

    def _extract_resources_from_apollo(self, data: dict) -> list:
        """Extract resource URLs from Apollo state data."""
        resources = []

        for key, value in data.items():
            if isinstance(value, dict):
                # Look for asset URLs
                if "url" in value and isinstance(value["url"], str):
                    url = value["url"]
                    if any(
                        ext in url.lower() for ext in [".pdf", ".pptx", ".docx", ".zip", ".ipynb"]
                    ):
                        resources.append(
                            {"url": url, "name": value.get("name", Path(url).name), "type": "asset"}
                        )

                # Look for video subtitle/resource links
                if "definition" in value:
                    defn = value["definition"]
                    if isinstance(defn, dict) and "assets" in defn:
                        for asset in defn.get("assets", []):
                            if isinstance(asset, dict) and "url" in asset:
                                resources.append(
                                    {
                                        "url": asset["url"],
                                        "name": asset.get("name", "resource"),
                                        "type": "lecture_asset",
                                    }
                                )

        return resources

    def download_file(self, url: str, filepath: Path, desc: str = None) -> bool:
        """
        Download a file with progress bar.

        Args:
            url: URL to download
            filepath: Local path to save file
            desc: Description for progress bar

        Returns:
            True if successful, False otherwise
        """
        try:
            # Handle relative URLs
            if url.startswith("/"):
                url = urljoin(self.BASE_URL, url)

            response = self.session.get(url, stream=True, allow_redirects=True)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))

            filepath.parent.mkdir(parents=True, exist_ok=True)

            with open(filepath, "wb") as f:
                with tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    desc=desc or filepath.name,
                    disable=total_size == 0,
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

            return True

        except requests.RequestException as e:
            print(f"Failed to download {url}: {e}")
            return False

    def sanitize_filename(self, name: str) -> str:
        """Sanitize a string to be used as a filename."""
        # Remove invalid characters
        name = re.sub(r'[<>:"/\\|?*]', "", name)
        # Replace spaces with underscores
        name = name.replace(" ", "_")
        # Limit length
        return name[:200]

    def download_course(self, course_slug: str, course_name: str) -> int:
        """
        Download all available materials for a course.

        Args:
            course_slug: Course URL slug
            course_name: Human-readable course name

        Returns:
            Number of files downloaded
        """
        print(f"\n{'='*60}")
        print(f"Downloading: {course_name}")
        print(f"{'='*60}")

        course_dir = self.output_dir / self.sanitize_filename(course_name)
        course_dir.mkdir(parents=True, exist_ok=True)

        downloaded = 0

        # Get course ID
        course_id = self.get_course_id(course_slug)
        if course_id:
            print(f"Course ID: {course_id}")

        # Get supplementary materials
        print("\nFetching supplementary materials...")
        supplements = self.get_supplementary_materials(course_id) if course_id else []
        print(f"Found {len(supplements)} supplement items")

        for supp in supplements:
            assets = self.get_supplement_assets(course_slug, supp.get("slug", ""))
            for asset in assets:
                url = asset["url"]
                name = self.sanitize_filename(asset["name"])
                if not name.endswith((".pdf", ".pptx", ".docx", ".zip")):
                    name += ".pdf"

                filepath = course_dir / name
                if not filepath.exists():
                    if self.download_file(url, filepath, desc=name):
                        downloaded += 1
                        time.sleep(0.5)  # Rate limiting

        # Get lecture resources
        print("\nFetching lecture resources...")
        resources = self.get_lecture_resources(course_slug)
        print(f"Found {len(resources)} resources")

        for resource in resources:
            url = resource["url"]
            name = self.sanitize_filename(resource["name"])

            # Ensure proper extension
            url_path = urlparse(url).path
            if "." in Path(url_path).name:
                ext = Path(url_path).suffix
                if not name.endswith(ext):
                    name += ext

            filepath = course_dir / name
            if not filepath.exists():
                if self.download_file(url, filepath, desc=name):
                    downloaded += 1
                    time.sleep(0.5)  # Rate limiting

        # Try to get materials from the course materials API
        print("\nFetching from course materials API...")
        materials = self.get_course_materials(course_slug)
        if materials:
            # Save the course structure as JSON for reference
            structure_file = course_dir / "course_structure.json"
            with open(structure_file, "w") as f:
                json.dump(materials, f, indent=2)
            print(f"Saved course structure to {structure_file}")

        print(f"\nDownloaded {downloaded} files for {course_name}")
        return downloaded

    def download_all_courses(self, courses: list = None) -> int:
        """
        Download materials from all specified courses.

        Args:
            courses: List of course dicts with 'slug' and 'name' keys.
                    If None, downloads all Deep Learning Specialization courses.

        Returns:
            Total number of files downloaded
        """
        if courses is None:
            courses = DEEP_LEARNING_COURSES

        total_downloaded = 0

        print(f"Starting download of {len(courses)} courses...")
        print(f"Output directory: {self.output_dir.absolute()}")

        for course in courses:
            try:
                count = self.download_course(course["slug"], course["name"])
                total_downloaded += count
            except Exception as e:
                print(f"Error downloading {course['name']}: {e}")
                continue

        print(f"\n{'='*60}")
        print(f"Download complete! Total files: {total_downloaded}")
        print(f"Files saved to: {self.output_dir.absolute()}")
        print(f"{'='*60}")

        return total_downloaded


def get_cauth_instructions():
    """Return instructions for getting the CAUTH cookie."""
    return """
How to get your CAUTH cookie:

1. Open your browser and go to https://www.coursera.org
2. Log in to your Coursera account
3. Make sure you're enrolled in the Deep Learning Specialization
4. Open Developer Tools:
   - Chrome/Edge: Press F12 or Ctrl+Shift+I (Cmd+Option+I on Mac)
   - Firefox: Press F12 or Ctrl+Shift+I
5. Go to the "Application" tab (Chrome) or "Storage" tab (Firefox)
6. In the left sidebar, expand "Cookies" and click on "https://www.coursera.org"
7. Find the cookie named "CAUTH"
8. Copy the entire value (it's a long string)
9. Use this value with the --cauth argument

Example:
    python coursera_notes_downloader.py --cauth "your_long_cauth_cookie_value_here"

Note: The CAUTH cookie expires, so you may need to get a fresh one if downloads fail.
"""


def main():
    parser = argparse.ArgumentParser(
        description="Download lecture notes from Coursera Deep Learning Specialization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=get_cauth_instructions(),
    )

    parser.add_argument(
        "--cauth",
        type=str,
        help="Your Coursera CAUTH cookie value (required for authentication)",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="coursera_notes",
        help="Output directory for downloaded files (default: coursera_notes)",
    )

    parser.add_argument(
        "--course",
        "-c",
        type=int,
        choices=[1, 2, 3, 4, 5],
        help="Download only a specific course (1-5). If not specified, downloads all courses.",
    )

    parser.add_argument(
        "--list-courses", action="store_true", help="List all available courses and exit"
    )

    parser.add_argument(
        "--instructions", action="store_true", help="Show instructions for getting CAUTH cookie"
    )

    args = parser.parse_args()

    if args.instructions:
        print(get_cauth_instructions())
        return

    if args.list_courses:
        print("\nDeepLearning.AI Deep Learning Specialization Courses:")
        print("-" * 50)
        for i, course in enumerate(DEEP_LEARNING_COURSES, 1):
            print(f"  {i}. {course['name']}")
            print(f"     Slug: {course['slug']}")
        print()
        return

    if not args.cauth:
        print("Error: --cauth is required for downloading course materials.")
        print("\nUse --instructions to see how to get your CAUTH cookie.")
        print("Use --list-courses to see available courses.")
        sys.exit(1)

    # Select courses to download
    if args.course:
        courses = [DEEP_LEARNING_COURSES[args.course - 1]]
    else:
        courses = DEEP_LEARNING_COURSES

    # Create downloader and start
    downloader = CourseraDownloader(cauth_cookie=args.cauth, output_dir=args.output)

    try:
        downloader.download_all_courses(courses)
    except KeyboardInterrupt:
        print("\n\nDownload interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during download: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

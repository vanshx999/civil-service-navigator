import os
import time
import hashlib
from datetime import datetime

import requests
from bs4 import BeautifulSoup

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")
os.makedirs(RAW_DIR, exist_ok=True)

USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) CivicSenseNavigator/1.0 (hackathon project; citing sources)"


def _safe_text(elem):
    return elem.get_text(strip=True) if elem else ""


def _save(source_name: str, url: str, content: str):
    safe_name = source_name.lower().replace(" ", "_").replace("/", "_")
    ts = datetime.now().strftime("%Y%m%d")
    filename = f"{safe_name}_{ts}.md"
    filepath = os.path.join(RAW_DIR, filename)
    full = f"# {source_name}\n\n- **Source URL**: {url}\n- **Scraped**: {datetime.now().isoformat()}\n- **Source**: {source_name}\n\n---\n\n{content}\n"
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(full)
    print(f"  Saved: {filepath} ({len(content)} chars)")
    return filepath


def scrape_waste_management():
    url = "https://environment.delhi.gov.in/environment/waste-management"
    print(f"\n[1/6] Scraping: Delhi Environment Dept - Waste Management")
    try:
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        content_div = soup.find("div", class_="field__item")
        if not content_div:
            content_div = soup.find("div", class_="content")
        if not content_div:
            content_div = soup.find("article")
        if not content_div:
            content_div = soup
        text = content_div.get_text(separator="\n\n", strip=True)
        # Clean up
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        text = "\n\n".join(lines)
        _save("Delhi_Environment_Dept_Waste_Management", url, text)
    except Exception as e:
        print(f"  ERROR: {e}")


def scrape_pib_swm_rules():
    url = "https://pib.gov.in/PressReleasePage.aspx?PRID=2219676"
    print(f"\n[2/6] Scraping: PIB - Solid Waste Management Rules 2026")
    try:
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        content_div = soup.find("div", class_="page-container")
        if not content_div:
            content_div = soup.find("div", id="print-content")
        if not content_div:
            content_div = soup.find("div", class_="content")
        if not content_div:
            content_div = soup
        text = content_div.get_text(separator="\n\n", strip=True)
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        text = "\n\n".join(lines)
        _save("PIB_SWM_Rules_2026", url, text)
    except Exception as e:
        print(f"  ERROR: {e}")


def scrape_pib_air_pollution():
    url = "https://pib.gov.in/PressReleasePage.aspx?PRID=2265580"
    print(f"\n[3/6] Scraping: PIB - Delhi Air Pollution Action Plan Review")
    try:
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        content_div = soup.find("div", class_="page-container")
        if not content_div:
            content_div = soup.find("div", id="print-content")
        if not content_div:
            content_div = soup.find("div", class_="content")
        if not content_div:
            content_div = soup
        text = content_div.get_text(separator="\n\n", strip=True)
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        text = "\n\n".join(lines)
        _save("PIB_Delhi_Air_Pollution_Review", url, text)
    except Exception as e:
        print(f"  ERROR: {e}")


def scrape_smart_cities_delhi():
    url = "https://smartcities.data.gov.in/ministrydepartment/Delhi"
    print(f"\n[4/6] Scraping: Smart Cities Data Portal - Delhi")
    try:
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        content_div = soup.find("div", class_="content")
        if not content_div:
            content_div = soup.find("main")
        if not content_div:
            content_div = soup
        text = content_div.get_text(separator="\n\n", strip=True)
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        text = "\n\n".join(lines)
        _save("Smart_Cities_Data_Delhi", url, text)
    except Exception as e:
        print(f"  ERROR: {e}")


def scrape_delhi_gov():
    print(f"\n[5/6] Scraping: Delhi Government Portal")
    urls = [
        ("https://delhi.gov.in/", "Delhi_Government_Home"),
        ("https://delhi.gov.in/departments", "Delhi_Government_Departments"),
        ("https://delhi.gov.in/help-line-numbers", "Delhi_Helplines"),
    ]
    for url, name in urls:
        try:
            resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "lxml")
            content_div = soup.find("div", class_="content")
            if not content_div:
                content_div = soup.find("main")
            if not content_div:
                content_div = soup
            text = content_div.get_text(separator="\n\n", strip=True)
            lines = [l.strip() for l in text.split("\n") if l.strip()]
            text = "\n\n".join(lines)
            _save(name, url, text)
        except Exception as e:
            print(f"  ERROR scraping {url}: {e}")


def scrape_mcd():
    print(f"\n[6/6] Scraping: MCD (Municipal Corporation of Delhi)")
    urls = [
        ("https://mcdonline.nic.in/", "MCD_Home"),
    ]
    for url, name in urls:
        try:
            resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "lxml")
            content_div = soup.find("div", class_="content")
            if not content_div:
                content_div = soup.find("main")
            if not content_div:
                content_div = soup
            text = content_div.get_text(separator="\n\n", strip=True)
            lines = [l.strip() for l in text.split("\n") if l.strip()]
            text = "\n\n".join(lines)
            _save(name, url, text)
        except Exception as e:
            print(f"  ERROR scraping {url}: {e}")


def scrape_times_of_india_articles():
    print(f"\n[BONUS] Scraping: News articles about Delhi civic issues")
    articles = [
        ("MCD waste management roadmap", "https://timesofindia.indiatimes.com/city/delhi/mcd-prepares-roadmap-to-implement-waste-mgmt-rules/articleshow/129934258.cms"),
        ("MCD IIT Delhi partnership", "https://timesofindia.indiatimes.com/city/delhi/mcd-ties-up-with-iit-delhi-to-push-zero-waste-to-landfill-goal/articleshow/130982323.cms"),
        ("MCD budget hike", "https://timesofindia.indiatimes.com/city/delhi/to-boost-civic-infra-and-waste-management-allocation-for-mcd-hiked/articleshow/129784814.cms"),
    ]
    for name, url in articles:
        try:
            resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "lxml")
            art_div = soup.find("div", class_="article_content")
            if not art_div:
                art_div = soup.find("div", class_="_s30J")
            if not art_div:
                art_div = soup.find("article")
            if not art_div:
                art_div = soup
            text = art_div.get_text(separator="\n\n", strip=True)
            lines = [l.strip() for l in text.split("\n") if l.strip()]
            text = "\n\n".join(lines)
            _save(f"TOI_{name.replace(' ', '_')}", url, text)
        except Exception as e:
            print(f"  ERROR scraping {url}: {e}")


def scrape_swachh_survekshan():
    print(f"\n[BONUS] Scraping: Swachh Survekshan / Clean India info")
    urls = [
        ("https://swachhsurvekshan2025.org/", "Swachh_Survekshan"),
    ]
    for url, name in urls:
        try:
            resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "lxml")
            text = soup.get_text(separator="\n\n", strip=True)
            lines = [l.strip() for l in text.split("\n") if l.strip()]
            text = "\n\n".join(lines)
            _save(name, url, text)
        except Exception as e:
            print(f"  ERROR scraping {url}: {e}")


def scrape_ncrb_crime_data():
    print(f"\n[BONUS] Scraping: NCRB Crime Data (Delhi)")
    url = "https://ncrb.gov.in/"
    try:
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        text = soup.get_text(separator="\n\n", strip=True)
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        text = "\n\n".join(lines)
        _save("NCRB_Crime_Data", url, text)
    except Exception as e:
        print(f"  ERROR: {e}")


def scrape_dpcc():
    print(f"\n[BONUS] Scraping: Delhi Pollution Control Committee")
    url = "https://dpcc.delhi.gov.in/"
    try:
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        content_div = soup.find("div", class_="content")
        if not content_div:
            content_div = soup
        text = content_div.get_text(separator="\n\n", strip=True)
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        text = "\n\n".join(lines)
        _save("DPCC_Delhi_Pollution_Control", url, text)
    except Exception as e:
        print(f"  ERROR: {e}")


def scrape_the_hindu_landfill():
    print(f"\n[BONUS] Scraping: The Hindu - Landfill clearance deadline")
    url = "https://www.thehindu.com/news/cities/Delhi/cm-rekha-gupta-directs-mcd-to-clear-landfill-sites-by-december-2026/article69983042.ece"
    try:
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        art_div = soup.find("div", class_="articlebody")
        if not art_div:
            art_div = soup.find("div", class_="content")
        if not art_div:
            art_div = soup
        text = art_div.get_text(separator="\n\n", strip=True)
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        text = "\n\n".join(lines)
        _save("The_Hindu_Landfill_Deadline", url, text)
    except Exception as e:
        print(f"  ERROR: {e}")


def main():
    print("=" * 60)
    print("Delhi Civic Sense Navigator - Data Scraper")
    print(f"Output: {RAW_DIR}")
    print("=" * 60)

    scrape_waste_management()
    scrape_pib_swm_rules()
    scrape_pib_air_pollution()
    scrape_smart_cities_delhi()
    scrape_delhi_gov()
    scrape_mcd()
    scrape_times_of_india_articles()
    scrape_swachh_survekshan()
    scrape_ncrb_crime_data()
    scrape_dpcc()
    scrape_the_hindu_landfill()

    print(f"\n{'=' * 60}")
    print(f"Done! Files saved to {RAW_DIR}")
    files = [f for f in os.listdir(RAW_DIR) if f.endswith(".md")]
    print(f"Total files: {len(files)}")
    for f in sorted(files):
        size = os.path.getsize(os.path.join(RAW_DIR, f))
        print(f"  {f} ({size:,} bytes)")
    print("=" * 60)


if __name__ == "__main__":
    main()

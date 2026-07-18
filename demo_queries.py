"""
Demo queries for the Delhi Civic Sense Navigator.
Run: python demo_queries.py
"""
import json
import sys
import urllib.request
import urllib.error

API_URL = "http://localhost:8000/api/ask"

DEMO_QUERIES = [
    "What are the new Solid Waste Management Rules 2026?",
    "How do I report a pothole or garbage in Delhi?",
    "What helplines exist for water, electricity, and emergencies in Delhi?",
    "What is being done about Delhi air pollution?",
    "Tell me about MCD waste management in Delhi",
    "What is the status of landfill sites in Delhi?",
    "How does Delhi handle biomedical waste?",
    "What are the penalties for using plastic bags in Delhi?",
    "How can I contact MCD for civic issues?",
    "What is the MCD 311 app and how do I use it?",
    "Tell me about the new Extended Bulk Waste Generator Responsibility (EBWGR)",
    "What is being done about road dust pollution in Delhi?",
    "What are the Waste to Energy plants in Delhi?",
    "How do I report waterlogging during monsoon in Delhi?",
    "What is DPCC and what does it do?",
    "Tell me about construction and demolition waste management in Delhi",
    "What is the Clean Air Programme for Delhi?",
    "How do I apply for a health trade license in Delhi?",
    "What is the role of MCD in waste segregation?",
    "Tell me about MCD's partnership with IIT Delhi for waste management",
    "What are the main causes of air pollution in Delhi?",
    "How can citizens participate in keeping Delhi clean?",
    "What is the Swachh Survekshan and how does it affect Delhi?",
    "What should I do if I see open waste burning in my area?",
    "What are the different types of waste and how should they be segregated?",
]


def run_query(query: str, idx: int, total: int):
    data = json.dumps({"query": query}).encode("utf-8")
    req = urllib.request.Request(
        API_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        print(f"\n{'=' * 70}")
        print(f"Q{idx}/{total}: {query}")
        print(f"{'=' * 70}")
        print(f"Answer: {result['answer'][:600]}...")
        print(f"\nCitations ({len(result['citations'])}):")
        for c in result["citations"]:
            print(f"  [{c['id']}] {c['source']} - {c['url']}")
        print(f"Chunks retrieved: {result.get('chunks_retrieved', 0)}")
        return True
    except urllib.error.HTTPError as e:
        print(f"\nERROR Q{idx}: {query}")
        print(f"  HTTP {e.code}: {e.read().decode()[:200]}")
        return False
    except Exception as e:
        print(f"\nERROR Q{idx}: {query}")
        print(f"  {e}")
        return False


def main():
    print("=" * 70)
    print("Delhi Civic Sense Navigator - Demo Queries")
    print(f"API: {API_URL}")
    print(f"Total demo queries: {len(DEMO_QUERIES)}")
    print("=" * 70)

    # First check if server is running
    try:
        urllib.request.urlopen("http://localhost:8000/health", timeout=5)
        print("\nServer is running!\n")
    except Exception:
        print("\nWARNING: Server not running at localhost:8000")
        print("Start it with: uvicorn src.main:app --reload --port 8000")
        print("Or: python -m src.main")
        cont = input("\nContinue anyway? (y/n): ").strip().lower()
        if cont != "y":
            return

    success = 0
    failed = 0

    for i, query in enumerate(DEMO_QUERIES, 1):
        if run_query(query, i, len(DEMO_QUERIES)):
            success += 1
        else:
            failed += 1

    print(f"\n{'=' * 70}")
    print(f"Done! {success}/{len(DEMO_QUERIES)} queries succeeded, {failed} failed")
    print("=" * 70)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Preprocess the Enron email corpus into a PyG TemporalData object.

This script parses all emails from the raw Enron corpus and builds a 
continuous-time dynamic graph (CTDG) represented as a chronologically 
sorted stream of events (sender → recipient at timestamp).

The Enron corpus structure:
- 150 directories, each named after a person (e.g., "germany-c", "davis-d")
- Each person's directory contains:
    - "_sent_mail" or "sent_items" or "send" folders: emails sent BY that person
    - Other folders: emails received BY that person
- We only consider edges where BOTH sender and recipient are among the 150 people

Usage:
    python scripts/preprocess_enron.py [--force] [--workers NUM]

Output:
    data/processed/enron_temporal.pt - A torch_geometric.data.TemporalData object
"""

import argparse
import re
import os
import shutil
from datetime import datetime
from email.utils import parsedate_to_datetime
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from torch_geometric.data import TemporalData


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "enron_mail_corpus"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = PROCESSED_DATA_DIR / "enron_temporal.pt"

# Pre-compiled regex patterns for performance
_EMAIL_REGEX = re.compile(r"[\w.+'-]+@[\w.-]+\.\w+")


def fix_corpus_issues():
    parent_dir = RAW_DATA_DIR / "stokley-c"
    child_dir = parent_dir / "chris_stokley"
    if os.path.isdir(child_dir):
        for item in os.listdir(child_dir):
            src_path = os.path.join(child_dir, item)
            shutil.move(src_path, parent_dir)
        os.rmdir(child_dir)

    wrong_dirs = [("crandell-s", "crandall-s"), ("merriss-s", "merris-s"), ("rodrique-r", "rodrigue-r"), ("stclair-c", "clair-c"), ("williams-w3", "williams-b")]
    for wrong_name, correct_name in wrong_dirs:
        dir_with_wrong_name = RAW_DATA_DIR / wrong_name
        if os.path.isdir(dir_with_wrong_name):
            new_dir = RAW_DATA_DIR / correct_name
            os.rename(dir_with_wrong_name, new_dir)


def extract_email(text: str) -> Optional[str]:
    """
    Extract a clean email address from text.
    
    Handles formats like:
    - john.doe@enron.com
    - John Doe <john.doe@enron.com>
    - "Doe, John" <john.doe@enron.com>
    """
    text = text.strip()
    match = _EMAIL_REGEX.search(text)
    if match:
        return match.group(0).lower().strip()
    return None


def parse_recipient_list(text: str) -> List[str]:
    """
    Parse a comma-separated list of recipients.

    Handles multi-line continuations and various formats.
    """
    recipients = []

    # Split by comma, but be careful about commas in names
    # Simple approach: split and try to extract emails
    parts = text.split(",")

    for part in parts:
        email = extract_email(part)
        if email:
            recipients.append(email)

    return recipients


def parse_email_file(filepath: str) -> Optional[Dict]:
    """
    Parse a single email file and extract sender, recipients, and timestamp.

    Args:
        filepath: Path to the email file

    Returns:
        Dictionary with 'sender', 'recipients', and 'timestamp' keys,
        or None if parsing fails.
    """
    try:
        with open(filepath, "r", encoding="latin-1") as f:
            content = f.read()
    except (IOError, OSError):
        print("Warning: Failed to read file:", filepath)
        return None

    # Parse headers (headers are separated from body by empty line)
    header_section = content.split("\n\n", 1)[0]

    # Parse headers, handling multi-line continuation
    headers = {}
    current_header = None
    current_value = []

    for line in header_section.split("\n"):
        if line and line[0] in " \t":  # Continuation line
            if current_header:
                current_value.append(line.strip())
        elif ":" in line:
            if current_header:
                headers[current_header] = " ".join(current_value)
            current_header, value = line.split(":", 1)
            current_header = current_header.strip().lower()
            current_value = [value.strip()]
    # Don't forget the last header
    if current_header:
        headers[current_header] = " ".join(current_value)

    # Extract sender (From field)
    sender = headers.get("from", "").strip()
    if not sender:
        print("Warning: No sender in file:", filepath)
        return None

    # Clean sender email
    sender = extract_email(sender)
    if not sender:
        return None

    # Extract recipients (To, Cc, Bcc fields)
    recipients = []
    for field in ["to", "cc", "bcc"]:
        if field in headers:
            field_recipients = parse_recipient_list(headers[field])
            recipients.extend(field_recipients)

    if not recipients:
        return None

    # Extract timestamp (Date field)
    date_str = headers.get("date", "").strip()
    if not date_str:
        print("Warning: No date in file:", filepath)
        return None

    try:
        # Parse the date string
        dt = parsedate_to_datetime(date_str)
        timestamp = dt.timestamp()
    except (ValueError, TypeError):
        return None

    return {
        "sender": sender,
        "recipients": recipients,
        "timestamp": timestamp,
    }


def get_person_directories(data_dir: Path) -> List[str]:
    """
    Get the list of person directories (the 150 people in the corpus).
    
    Args:
        data_dir: Path to the root of the email corpus
        
    Returns:
        List of person directory names
    """
    person_dirs = []
    for item in data_dir.iterdir():
        if item.is_dir():
            person_dirs.append(item.name)
    return sorted(person_dirs)


def extract_email_address_from_sent_mail(person_dir: Path) -> Optional[str]:
    """
    Extract the email address of a person from their sent folders.
    
    The sender's email address appears in the "From" header of emails 
    they sent.
    
    Args:
        person_dir: Path to the person's directory
        
    Returns:
        The person's email address, or None if not found
    """
    last_name, first_name_letter = person_dir.name.split("-", 1)
    if "-" in first_name_letter:
        first_name_letter = first_name_letter.split("-")[-1]

    for subdir in ["_sent_mail", "sent_items", "sent"]:
        sent_mail_dir = person_dir / subdir
        if sent_mail_dir.exists() and sent_mail_dir.is_dir():
            for email_file in list(sent_mail_dir.iterdir()):
                if email_file.is_file():
                    parsed = parse_email_file(str(email_file))
                    if parsed and parsed["sender"]:
                        if ".." not in parsed["sender"]:
                            if parsed["sender"].startswith(first_name_letter):
                                if parsed["sender"].split(".", 1)[1].replace("-", "").replace("'", "").startswith(last_name):
                                    return parsed["sender"]

    return None


def build_person_email_mapping(data_dir: Path) -> Dict[str, str]:
    """
    Build a mapping from person directory names to their email addresses.
    
    Args:
        data_dir: Path to the root of the email corpus
        
    Returns:
        Dictionary mapping person directory names to email addresses
    """
    person_dirs = get_person_directories(data_dir)
    person_to_email = {
        # some people's email addresses are missing from their sent folders, so we hardcode them here
        "harris-s": "steven.harris@enron.com",
        "lucci-p": "paul.lucci@enron.com",
        "steffes-j": "james.steffes@enron.com",
        "white-s": "stacey.white@enron.com",
        "phanis-s": "stephanie.panus@enron.com",
        "whalley-l": "greg.whalley@enron.com",
    }

    print("Building person-to-email mapping...")
    for person_name in person_dirs:
        if person_name in person_to_email:
            continue
        person_dir = data_dir / person_name
        email = extract_email_address_from_sent_mail(person_dir)
        if email:
            person_to_email[person_name] = email
        else:
            print(f"Warning: Could not find email for {person_name}")
    
    print(f"Found email addresses for {len(person_to_email)} of {len(person_dirs)} people")

    return person_to_email


def parse_email(filepath: str, valid_emails: Set[str]) -> List[Dict]:
    """
    Parse an email.

    The sender and recipient are from the email headers.
    We only include edges where the sender and recipient are in the valid_emails set.

    Args:
        filepath: Path to the email file
        valid_emails: Set of valid email addresses (the 150 people)

    Returns:
        List of dictionaries with 'sender', 'recipient', and 'timestamp' keys
    """
    parsed = parse_email_file(filepath)
    if not parsed:
        return []
    
    sender = parsed["sender"]
    
    # Only include if sender is one of the 150 people
    if sender not in valid_emails:
        return []
    
    edges = []
    timestamp = parsed["timestamp"]

    for recipient in parsed["recipients"]:
        if recipient in valid_emails:
            edges.append({
                "sender": sender,
                "recipient": recipient,
                "timestamp": timestamp,
            })

    return edges


def process_person_directory(args: Tuple[str, Set[str]]) -> List[Dict]:
    """
    Process all emails in a person's directory.
    
    Args:
        args: Tuple of (person_dir_path, valid_emails)

    Returns:
        List of edge dictionaries
    """
    person_dir_path, valid_emails = args
    person_dir = Path(person_dir_path)
    edges = []

    for email_file in person_dir.rglob("*"):
        if email_file.is_file():
            file_edges = parse_email(str(email_file), valid_emails)
            edges.extend(file_edges)

    return edges


def build_temporal_graph(edges: List[Dict], valid_emails: Set[str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, int]]:
    """
    Build temporal graph data from parsed edges.

    Creates contiguous node IDs (0 to N-1) for all unique email addresses,
    then creates edges from sender to recipient with timestamps.

    Args:
        edges: List of edge dictionaries with 'sender', 'recipient', 'timestamp'
        valid_emails: Set of valid email addresses (the 150 people)

    Returns:
        Tuple of (src, dst, t, node_mapping) where:
        - src: source node indices (senders)
        - dst: destination node indices (recipients)
        - t: timestamps (as float seconds since epoch)
        - node_mapping: dictionary mapping email addresses to node IDs
    """

    class IdGenerator:
        def __init__(self):
            self.next_id = 0

        def get_next(self) -> int:
            self.next_id += 1
            return self.next_id - 1

    # Build node mapping (email -> contiguous ID)
    node_to_id: Dict[str, int] = {}
    id_generator = IdGenerator()

    def get_or_create_node(email: str, id_generator: IdGenerator) -> int:
        if email not in node_to_id:
            node_to_id[email] = id_generator.get_next()
        return node_to_id[email]

    # Filter edges to only include valid emails and collect edges
    filtered_edges: List[Tuple[int, int, float]] = []

    for edge in edges:
        sender = edge["sender"]
        recipient = edge["recipient"]        
        sender_id = get_or_create_node(sender, id_generator)
        recipient_id = get_or_create_node(recipient, id_generator)
        # Avoid duplicate edges (same sender, recipient, timestamp)
        if (sender_id, recipient_id, edge["timestamp"]) not in filtered_edges:
            filtered_edges.append((sender_id, recipient_id, edge["timestamp"]))

    if not filtered_edges:
        raise ValueError("No valid edges found after filtering")

    # Convert to numpy array for efficient sorting and tensor creation
    edges_np = np.array(filtered_edges, dtype=np.float64)

    # Sort by timestamp (column 2) using numpy argsort
    sort_idx = np.argsort(edges_np[:, 2])
    edges_sorted = edges_np[sort_idx]

    # Create tensors from numpy arrays
    src = torch.from_numpy(edges_sorted[:, 0].astype(np.int64))
    dst = torch.from_numpy(edges_sorted[:, 1].astype(np.int64))
    t = torch.from_numpy(edges_sorted[:, 2])

    print(f"Built graph with {id_generator.next_id} unique nodes and {len(filtered_edges)} edges")
    print(f"Time range: {datetime.fromtimestamp(t.min().item())} to {datetime.fromtimestamp(t.max().item())}")

    return src, dst, t, node_to_id


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess Enron email corpus into TemporalData format"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing even if output file exists",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=cpu_count(),
        help=f"Number of worker processes (default: {cpu_count()})",
    )
    args = parser.parse_args()

    # Check if output already exists
    if OUTPUT_FILE.exists() and not args.force:
        print(f"Output file already exists: {OUTPUT_FILE}")
        print("Use --force to reprocess")
        return

    # Ensure output directory exists
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Fix known corpus issues
    fix_corpus_issues()

    # Step 1: Build mapping from person directory names to email addresses
    person_to_email = build_person_email_mapping(RAW_DATA_DIR)
    
    # Create set of valid email addresses (the 150 people)
    valid_emails = set(person_to_email.values())

    # Step 2: Process each person's directory
    print(f"Processing {len(person_to_email)} person directories...")
    
    # Prepare arguments for parallel processing
    process_args = [
        (str(RAW_DATA_DIR / person_name), valid_emails)
        for person_name in person_to_email.keys()
    ]
    
    all_edges = []
    with Pool(processes=args.workers) as pool:
        results = list(pool.imap(
            process_person_directory,
            process_args,
            chunksize=max(1, len(process_args) // (args.workers * 4))
        ))
        for person_edges in results:
            all_edges.extend(person_edges)

    print(f"Total edges before filtering: {len(all_edges)}")

    # Step 3: Build temporal graph
    print("Building temporal graph...")
    src, dst, t, node_mapping = build_temporal_graph(all_edges, valid_emails)

    # Create TemporalData object
    data = TemporalData(src=src, dst=dst, t=t)

    # Save to file
    print(f"Saving to {OUTPUT_FILE}...")
    torch.save(data, OUTPUT_FILE)

    # Also save the node mapping for reference
    mapping_file = PROCESSED_DATA_DIR / "enron_node_mapping.pt"
    torch.save(node_mapping, mapping_file)

    print("Done!")
    print(f"  TemporalData saved to: {OUTPUT_FILE}")
    print(f"  Node mapping saved to: {mapping_file}")


if __name__ == "__main__":
    main()

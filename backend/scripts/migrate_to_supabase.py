"""
backend/migrate_to_supabase.py
──────────────────────────────
One-time migration script to move existing user accounts and face embeddings
from accounts_v3.pkl and face_database_v3.pkl into Supabase.

Safe to re-run (idempotent): skips existing records.
Does NOT modify or delete the original .pkl backup files.
"""

import os
import sys
import pickle
import numpy as np
import bcrypt

# Ensure project root is in sys.path
PROJECT_ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.database.supabase import get_supabase_client

ACCOUNTS_PKL = os.path.join(PROJECT_ROOT, "accounts_v3.pkl")
FACES_PKL = os.path.join(PROJECT_ROOT, "face_database_v3.pkl")


def is_hashed(password_str: str) -> bool:
    """
    Check whether a string appears to already be a hash:
    - bcrypt hash (starts with $2a$, $2b$, or $2y$)
    - SHA-256 hex string (64 characters of hex digits)
    """
    if not password_str:
        return False
    if password_str.startswith(("$2a$", "$2b$", "$2y$")):
        return True
    if len(password_str) == 64 and all(c in "0123456789abcdefABCDEF" for c in password_str):
        return True
    return False


def hash_with_bcrypt(plain_password: str) -> str:
    """Hash a plaintext password using bcrypt."""
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(plain_password.encode("utf-8"), salt).decode("utf-8")


def migrate():
    print("=" * 60)
    print("Starting Migration: Local Pickle Files -> Supabase")
    print("=" * 60)

    client = get_supabase_client()

    users_migrated = 0
    users_skipped = 0
    faces_migrated = 0
    faces_skipped = 0
    errors = []

    # ──────────────────────────────────────────────────────────
    # 1. Migrate Users from accounts_v3.pkl
    # ──────────────────────────────────────────────────────────
    if not os.path.exists(ACCOUNTS_PKL):
        print(f"Warning: {ACCOUNTS_PKL} not found. Skipping user migration.")
    else:
        print(f"Loading accounts from: {ACCOUNTS_PKL}")
        try:
            with open(ACCOUNTS_PKL, "rb") as f:
                accounts_data = pickle.load(f)
        except Exception as exc:
            print(f"Error reading {ACCOUNTS_PKL}: {exc}")
            accounts_data = {}

        for email, info in accounts_data.items():
            email_clean = email.strip().lower()
            username = info.get("username", email_clean.split("@")[0])
            raw_password = info.get("password", "")

            # Check if password needs bcrypt hashing or is already hashed
            if is_hashed(raw_password):
                password_hash = raw_password
            else:
                password_hash = hash_with_bcrypt(raw_password)

            # Check if user already exists in Supabase
            try:
                res = client.table("users").select("email").eq("email", email_clean).execute()
                if res.data and len(res.data) > 0:
                    print(f"  [User Skipped] {email_clean} already exists in Supabase.")
                    users_skipped += 1
                    continue

                client.table("users").insert({
                    "email": email_clean,
                    "username": username,
                    "password_hash": password_hash,
                }).execute()
                print(f"  [User Migrated] {email_clean} ({username})")
                users_migrated += 1
            except Exception as exc:
                err_msg = f"Failed migrating user {email_clean}: {exc}"
                print(f"  [ERROR] {err_msg}")
                errors.append(err_msg)

    # ──────────────────────────────────────────────────────────
    # 2. Migrate Face Embeddings from face_database_v3.pkl
    # ──────────────────────────────────────────────────────────
    if not os.path.exists(FACES_PKL):
        print(f"Warning: {FACES_PKL} not found. Skipping face embeddings migration.")
    else:
        print(f"\nLoading face embeddings from: {FACES_PKL}")
        try:
            with open(FACES_PKL, "rb") as f:
                faces_data = pickle.load(f)
        except Exception as exc:
            print(f"Error reading {FACES_PKL}: {exc}")
            faces_data = {}

        for email, emb in faces_data.items():
            email_clean = email.strip().lower()

            # Ensure embedding is a JSON-serializable list of floats
            if isinstance(emb, np.ndarray):
                emb_list = [float(x) for x in emb.flatten().tolist()]
            elif isinstance(emb, (list, tuple)):
                emb_list = [float(x) for x in emb]
            else:
                err_msg = f"Invalid embedding type {type(emb)} for {email_clean}"
                print(f"  [ERROR] {err_msg}")
                errors.append(err_msg)
                continue

            try:
                # Check if face embedding already exists for this email
                res = client.table("face_embeddings").select("id").eq("email", email_clean).execute()
                if res.data and len(res.data) > 0:
                    print(f"  [Face Skipped] Face embedding for {email_clean} already exists.")
                    faces_skipped += 1
                    continue

                # Ensure user exists in users table first (foreign key constraint)
                user_res = client.table("users").select("email").eq("email", email_clean).execute()
                if not user_res.data:
                    # Insert user stub if not found
                    client.table("users").insert({
                        "email": email_clean,
                        "username": email_clean.split("@")[0],
                        "password_hash": hash_with_bcrypt("temporary_pass_123"),
                    }).execute()
                    users_migrated += 1

                client.table("face_embeddings").insert({
                    "email": email_clean,
                    "embedding": emb_list,
                }).execute()
                print(f"  [Face Migrated] {email_clean} ({len(emb_list)} dimensions)")
                faces_migrated += 1
            except Exception as exc:
                err_msg = f"Failed migrating face embedding for {email_clean}: {exc}"
                print(f"  [ERROR] {err_msg}")
                errors.append(err_msg)

    # ──────────────────────────────────────────────────────────
    # Summary
    # ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Migration Summary")
    print("=" * 60)
    print(f"Users migrated:         {users_migrated}")
    print(f"Users skipped:          {users_skipped}")
    print(f"Face embeddings migrated: {faces_migrated}")
    print(f"Face embeddings skipped:  {faces_skipped}")
    if errors:
        print(f"Errors encountered ({len(errors)}):")
        for err in errors:
            print(f"  - {err}")
    else:
        print("All records migrated with zero errors! Original .pkl files preserved.")
    print("=" * 60)


if __name__ == "__main__":
    migrate()

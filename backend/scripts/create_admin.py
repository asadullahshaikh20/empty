"""Create an Administrator account (admins cannot self-register through the API).

Usage (from backend/):
  python scripts/create_admin.py --name "Admin" --email admin@example.com --username admin
The password is prompted for (not echoed) unless --password is given.
"""

from __future__ import annotations

import argparse
import getpass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import func, select  # noqa: E402

from app.db import SessionLocal  # noqa: E402
from app.models import User  # noqa: E402
from pydantic import EmailStr, TypeAdapter, ValidationError  # noqa: E402

from app.schemas import USERNAME_RE, validate_password_policy  # noqa: E402
from app.security import hash_password  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--email", required=True)
    ap.add_argument("--username", required=True)
    ap.add_argument("--password", help="omit to be prompted")
    args = ap.parse_args(argv)

    if not USERNAME_RE.fullmatch(args.username):
        print("error: username must be 3-50 characters: letters, digits, '_', '.' or '-'", file=sys.stderr)
        return 2
    try:
        email = TypeAdapter(EmailStr).validate_python(args.email.strip()).lower()
    except ValidationError:
        print("error: invalid email address", file=sys.stderr)
        return 2
    password = args.password
    if password is None:
        password = getpass.getpass("Password: ")
        if getpass.getpass("Confirm password: ") != password:
            print("error: passwords do not match", file=sys.stderr)
            return 2
    try:
        validate_password_policy(password)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    with SessionLocal() as db:
        if db.scalar(select(User.id).where(User.email == email)) or db.scalar(
            select(User.id).where(func.lower(User.username) == args.username.lower())
        ):
            print("error: email or username already exists", file=sys.stderr)
            return 1
        db.add(User(name=args.name, email=email, username=args.username, password_hash=hash_password(password),
                    role="admin"))
        db.commit()
    print(f"Administrator {args.username} <{email}> created.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

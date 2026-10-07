"""
hash_api_key.py -- bcrypt API key হ্যাশ তৈরির utility

Usage:
    python src/hash_api_key.py my-secret-api-key
"""
import sys

def hash_key(plain: str) -> str:
    import bcrypt
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt(rounds=12)).decode("utf-8")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/hash_api_key.py <plain-key>")
        sys.exit(1)
    plain = sys.argv[1]
    hashed = hash_key(plain)
    print(f"\nPlain key  : {plain}")
    print(f"Bcrypt hash: {hashed}")
    print(f"\nAdd to .env: FINGUARD_API_KEYS={hashed}:admin")

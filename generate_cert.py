#!/usr/bin/env python3
"""
generate_cert.py — Self-signed TLS certificate generator for VoiceGuard demo.

Generates cert.pem + key.pem in the project root for local HTTPS serving.
Idempotent: skips if both files already exist. Use --force to regenerate.

    python generate_cert.py
    python generate_cert.py --force

DEMO-ONLY: These certificates are self-signed and intended solely for
development/demo use. Do not deploy to production with self-signed certs.
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys

def generate_self_signed_cert(
    cert_path: str = "cert.pem",
    key_path: str = "key.pem",
    cn: str = "VoiceGuard Demo",
    days: int = 365,
) -> None:
    """Generate a self-signed RSA certificate + private key."""
    try:
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import NameOID
    except ImportError:
        print("ERROR: 'cryptography' package is required. Install via:")
        print("  pip install cryptography")
        sys.exit(1)

    # Generate RSA private key
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    # Build certificate
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, cn),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "VoiceGuard SIH 2026"),
    ])

    now = datetime.datetime.now(datetime.timezone.utc)
    san_list = [
        x509.DNSName("localhost"),
        x509.DNSName("*.ngrok-free.app"),
        x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
        x509.IPAddress(ipaddress.IPv4Address("0.0.0.0")),
    ]
    try:
        import socket
        for ip in socket.gethostbyname_ex(socket.gethostname())[2]:
            try:
                addr = ipaddress.ip_address(ip)
                if addr not in (ipaddress.IPv4Address("127.0.0.1"), ipaddress.IPv4Address("0.0.0.0")):
                    san_list.append(x509.IPAddress(addr))
            except Exception:
                pass
    except Exception:
        pass

    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + datetime.timedelta(days=days))
        .add_extension(
            x509.SubjectAlternativeName(san_list),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )

    # Write private key
    with open(key_path, "wb") as f:
        f.write(key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        ))
    print(f"  Private key written: {key_path}")

    # Write certificate
    with open(cert_path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
    print(f"  Certificate written: {cert_path}")


import ipaddress  # noqa: E402 — needed for SAN IP addresses


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate self-signed TLS cert for VoiceGuard demo."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate even if cert.pem / key.pem already exist.",
    )
    parser.add_argument(
        "--cert", default="cert.pem", help="Output certificate path (default: cert.pem)"
    )
    parser.add_argument(
        "--key", default="key.pem", help="Output private key path (default: key.pem)"
    )
    args = parser.parse_args()

    if not args.force and os.path.exists(args.cert) and os.path.exists(args.key):
        print(f"Certificate files already exist ({args.cert}, {args.key}). Use --force to regenerate.")
        return

    print("Generating self-signed TLS certificate for VoiceGuard demo...")
    generate_self_signed_cert(cert_path=args.cert, key_path=args.key)
    print("Done. Start the backend with:")
    print(f"  python backend.py")


if __name__ == "__main__":
    main()

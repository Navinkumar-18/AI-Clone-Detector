# VoiceGuard — Security Remediation & Certificate-Key Hygiene

**Audit Target**: `Navinkumar-18/AI-Clone-Detector`  
**Branch**: `voiceguard/sih2026-hardening`  
**Security Status**: Remediated on hardening branch  

---

## 1. Secret Remediation & Certificate-Key Hygiene

### Compromised Certificate and Private Key
In earlier commits on the repository prior to base commit `45b9547`, raw certificate and private key files were committed into version control:
- `cert.pem`: Self-signed public certificate
- `key.pem`: RSA private key (2048-bit, unencrypted)

### Security Statement:
> "cert.pem and key.pem were previously committed and must not be reused for public deployment. New certificates and keys must be generated. Git-history cleanup may be required separately."

### Remediation Actions Taken on Branch:
1. **Removed from Tracking**:
   - Removed cert.pem and key.pem from Git tracking on the hardening branch and added certificate/key patterns to .gitignore. Because these files were previously committed, the old private key must be treated as compromised. Historical removal was not automatically performed.
   - Verification command:
     ```bash
     git ls-files | grep -E '(^|/)(cert|key)\.(pem|crt|key)$' || true
     # Output: empty (zero tracked keys)
     ```
2. **Git Exclusion Configuration**:
   - Updated `.gitignore` to prevent newly generated certificates and keys from being accidentally tracked:
     ```gitignore
     *.pem
     *.crt
     *.key
     *.pfx
     *.p12
     ```
   - *Clarification*: Adding files to `.gitignore` prevents future commits from tracking them; it does not remove files from prior Git history.
3. **Dynamic Ephemeral Generation**:
   - `generate_cert.py` is called dynamically on startup if local certificates are missing. Local self-signed certificates are for development-only testing. Public deployment requires new trusted certificate/key material issued by a trusted Certificate Authority.

---

## 2. TLS Certificate Verification & Demo Scoping

### Vulnerability in Earlier Prototype
Earlier prototype code in `lib/main.dart` installed a global `HttpOverrides` class:
```dart
// INSECURE PROTOTYPE PATTERN (REMOVED):
HttpOverrides.global = _DemoHttpOverrides(); // Bypassed TLS for ALL process traffic
```
This disabled certificate validation globally across all packages and network connections in the Dart process.

### Remediated Architecture:
Removed the global HttpOverrides bypass. Restricted the development-only self-signed certificate exception to the VoiceGuardApiClient, scoped to the configured target host and enabled only when kDemoMode && !kReleaseMode.

1. **Global Override Removed**: `HttpOverrides.global` was deleted from `lib/main.dart`. Unrelated network traffic is never subject to certificate validation bypasses.
2. **Target Host Scoped**: In `lib/api_client.dart`, an `IOClient` attaches a `badCertificateCallback` that accepts certificates only when the host strictly matches the configured `BackendConfig.baseUrl`.
3. **Secure Mode by Default**: `kDemoMode` defaults to `false` in `lib/config.dart`. Standard TLS validation is active out-of-the-box.
4. **Forbidden in Release Builds**: Guarded by `!kReleaseMode`. Release builds cannot enable the demo bypass, even if the flag is passed.
5. **Development Scope**: Local self-signed certificates are for development-only testing. Public deployment requires new trusted certificate/key material.

---

## 3. Upload Protection & Denial-of-Service Defense

| Defense Layer | Previous Implementation | Remediated Architecture |
| :--- | :--- | :--- |
| **Upload Size Ceiling** | `await file.read()` (buffered all into RAM) | `_read_upload_limited()` streams in 64 KB chunks, rejecting streams exceeding 10 MB with `HTTP 413` |
| **Audio Duration Bounds** | No upper duration limit | Gated by `max_duration=60.0` in `audio_quality.py` |
| **Worker Concurrency** | Unbounded async coroutines | Bounded `ThreadPoolExecutor` (4 workers) guarded by `asyncio.Semaphore(4)` |
| **Rate Limiting** | No request limits | In-memory sliding-window IP rate limiter (60 req/min, `HTTP 429`) |
| **Disk Cleanup** | Unlinked inconsistently | Unconditionally removed in `finally` block with structured `PRIVACY` cleanup logging |

### Rate Limiter Scope:
The in-memory rate limiter is suitable for a single-process prototype only. It is not sufficient for distributed production deployment.

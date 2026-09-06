# VoiceGuard — Security Remediation Report

**Audit Target**: `Navinkumar-18/AI-Clone-Detector`  
**Branch**: `voiceguard/sih2026-hardening`  
**Security Status**: Remediated on working branch  

---

## 1. Compromised Cryptographic Key Material

### The Issue
In earlier commits on the `main` branch (prior to base commit `45b9547`), raw certificate and private key files were committed directly into version control:
- `cert.pem`: Self-signed TLS public certificate
- `key.pem`: RSA private key (2048-bit, unencrypted)

### Security Statement
> **CRITICAL SECURITY DISCLOSURE**:  
> `cert.pem` and `key.pem` were previously committed and must be treated as compromised. They must not be reused for public deployment. New certificates and keys must be generated, and Git-history cleanup may be required separately.

### Actions Taken on Current Branch
1. **Removed from Tracking**: Executed `git rm --cached cert.pem key.pem`. The files are no longer tracked in the repository index.
2. **Exclusion from Future Commits**: Updated `.gitignore` with strict exclusion rules:
   ```gitignore
   *.pem
   *.crt
   *.key
   *.pfx
   *.p12
   ```
   Verified with `git check-ignore cert.pem key.pem test.key foo.crt`.
3. **On-Demand Ephemeral Generation**: `generate_cert.py` is invoked dynamically at runtime if local certificates are missing. Newly generated files are automatically ignored by git.

### History Retention and Cleanup Procedure
While the files are removed from the working branch, they persist in historical git commits prior to the hardening branch. To clean historical commits prior to public release:
```bash
# Optional history cleanup using git-filter-repo (destructive operation):
pip install git-filter-repo
git filter-repo --invert-paths --path cert.pem --path key.pem

# Requires coordinated force push to all remote branches:
git push origin --force --all
```
> **Notice**: As per security review guidelines, historical git rewrites are not performed automatically without explicit team coordination, to avoid breaking developer clones.

### Key Rotation Requirements for Production
- **Never reuse development keys**: Any key material generated on development machines must never be deployed to production.
- **Production PKI**: Production deployments must provision valid CA-signed certificates (via Let's Encrypt, DigiCert, AWS ACM, etc.) and inject secrets via environment variables or secret managers (e.g., HashiCorp Vault, AWS Secrets Manager).

---

## 2. TLS Certificate Verification & Demo Mode Scoping

### The Previous Vulnerability
Earlier prototype code in `lib/main.dart` installed a global `HttpOverrides` class:
```dart
// INSECURE PROTOTYPE PATTERN (REMOVED):
HttpOverrides.global = _DemoHttpOverrides(); // Bypassed TLS for ALL process traffic
```
This disabled certificate validation globally across all packages and plugins running in the Dart VM.

### The Remediated Architecture
We **restricted the development-only self-signed certificate exception to the VoiceGuard HTTP client and enabled it only when explicit demo mode is active. Secure mode keeps certificate verification enabled.**

1. **Global Override Completely Removed**: `HttpOverrides.global` is deleted from `lib/main.dart`. No global certificate override affects unrelated HTTP traffic.
2. **Scoped to VoiceGuard Client**: In `lib/api_client.dart`, an `IOClient` attaches a `badCertificateCallback` that verifies that the hostname matches `BackendConfig.baseUrl`.
3. **Secure Mode by Default**: `kDemoMode` defaults to `false` in `lib/config.dart`. Standard TLS verification is active out-of-the-box.
4. **Explicit Opt-In for Demo**: Demo mode requires explicit compile-time opt-in:
   ```bash
   flutter run -d windows --dart-define=DEMO_MODE=true
   ```
5. **Forbidden in Release Builds**: Guarded by `!kReleaseMode`. Even if `--dart-define=DEMO_MODE=true` is accidentally specified in a release build, the bypass is strictly ignored.
6. **Visible UI Indicator**: When demo mode is active, the app bar renders a prominent visual `DEMO MODE` badge.

### Backend Default Verification
In `voiceguard_config.py` and `config/model_config.yaml`:
```yaml
security:
  demo_mode: false # Secure default
  allowed_cors_origins:
    - "http://localhost"
    - "https://localhost"
```
Confirmed by automated test `tests/test_security_config.py::test_demo_mode_default_false`:
`VOICEGUARD_DEMO_MODE=false` is the secure default.

---

## 3. Upload Protection & Denial-of-Service Defense

| Vector | Previous Prototype | Remediated Architecture |
| :--- | :--- | :--- |
| **Upload Size** | `await file.read()` (read entire file into RAM) | `_read_upload_limited()` streams in 64 KB chunks, rejecting if bytes $> 10\text{ MB}$ (`HTTP 413`) |
| **Audio Duration** | No length limit | Gated by `max_duration=30.0` in `audio_quality.py` |
| **Concurrency** | Unbounded async coroutines | Bounded `ThreadPoolExecutor(max_workers=4)` with `asyncio.Semaphore` |
| **Abuse / Flooding** | No rate limits | Sliding-window IP rate limiter (60 req/min, `HTTP 429`) |
| **Disk Exhaustion** | Temp files unlinked inconsistently | Unconditionally removed in `finally` block with `PRIVACY` audit logging |

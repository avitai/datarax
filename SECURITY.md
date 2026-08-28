# Security Policy

## Supported Versions

Datarax is a research preview. Security fixes target the latest released version
and the current `main` branch.

## Reporting a Vulnerability

Please report suspected vulnerabilities privately by emailing
security@avitai.bio.

Include:

- Affected Datarax version or commit.
- Environment details, including Python, JAX, and operating system versions.
- A minimal reproduction or proof of impact.
- Any known mitigations.

We will acknowledge reports as soon as practical, investigate privately, and
coordinate disclosure once a fix or mitigation is available.

## Scope

Security-sensitive areas include pipeline deserialization, dataset and checkpoint loading, cache and file-system storage, source connectors that fetch remote data, CI release automation, and any code path that processes untrusted input data.

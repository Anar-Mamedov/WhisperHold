# Security Policy

## Reporting a vulnerability

Please **do not open a public issue** for security problems.

Report them privately through GitHub instead: open the repository's
**Security** tab and choose **Report a vulnerability**. Only the maintainer can
see the report, and we can discuss and fix it there before anything is made
public.

Helpful details to include:

- what the problem is and what an attacker could do with it
- steps to reproduce, or a proof of concept
- the macOS version and the commit you tested

## Supported versions

Only the latest commit on `main` receives security fixes.

## Scope

WhisperHold runs entirely on your Mac. Of particular interest are issues that
could make it:

- send audio or text off the machine
- paste text somewhere the user did not intend
- run untrusted code through its Python worker or runtime installer

Vulnerabilities in PyTorch, Transformers or other dependencies should be
reported to those projects; tell us too if WhisperHold needs a version bump.

# Priority and timestamps

## What this document claims

This project makes a **narrow, checkable** priority claim and states its evidence boundary explicitly.

**There is no verifiable evidence of any date before 2026-09-17.** Work carried out before that date has internal
records only, with no third-party timestamp. The verifiable public record of this project begins with the GitHub
repository creation (2026-09-17 16:10:19 UTC) and GitHub's record of receiving its content, at
2026-09-17 16:10:52 UTC.

## Why git history is not the evidence

A git commit carries two self-asserted timestamps (author date and committer date). Both are taken from
environment variables and can be set to any value; `--amend`, `rebase` and `filter-branch` rewrite them freely.
**A commit claiming an early date proves nothing about when its content existed.** The commit dates GitHub shows on
its web pages are read from the commit object, so they are equally self-asserted.

This repository's root commit illustrates the distinction rather than hiding it:

| Field | Value | Who asserts it |
|---|---|---|
| commit `8f951866…` author date | `2026-09-17T16:10:07Z` | the committer (forgeable) |
| commit `8f951866…` committer date | `2026-09-17T16:10:07Z` | the committer (forgeable) |
| commit signature | none (unsigned) | — |
| tag `v0.1.0-preview` | points at `8f951866229d5f4fbe050b9469c82957e691ff43` | — |
| repository `created_at` | **`2026-09-17T16:10:19Z`** | **GitHub** |
| repository activity: `main` created at `8f951866…` | **`2026-09-17T16:10:52Z`** | **GitHub** |

The last two rows are asserted by GitHub rather than by the author. Anyone can query them:

```
gh api repos/Ashley-QwQ/GlassBox-NR --jq .created_at
gh api repos/Ashley-QwQ/GlassBox-NR/activity --jq '.[] | select(.activity_type == "branch_creation")'
```

The repository's `pushed_at` field is not used as evidence here: it records the most recent push and moves forward
with every later update.

**The claim is therefore anchored at two GitHub-asserted times**, not at the commit's own date and not at any earlier
internal milestone:

1. **2026-09-17 16:10:52 UTC**: the repository activity record of `main` being created at commit `8f951866…`, i.e.
   the moment GitHub recorded receiving this content;
2. **2026-09-17 16:10:19 UTC**: the repository's `created_at`, a fixed field of the repository itself. It is listed
   as a second anchor because activity records may be subject to a retention period.

## Unpublished work: commitment, not disclosure

Work that is not yet public is anchored by publishing a **hash** rather than the files. A commitment manifest lists
every file's path, size and SHA-256, and reduces to a single root hash under a declared convention. Timestamping
that root proves the files existed at that moment without revealing their contents.

The manifest for the current unpublished work (the native rewrite of the pipeline's tail block and its audit trail)
was built on 2026-09-18 and timestamped by two independent RFC 3161 authorities. Snapshots v2 to v4 are later
snapshots of the same scope, built and timestamped the same way on the same day:

| Snapshot | Manifest root (SHA-256) | Files covered | FreeTSA token | DigiCert token |
|---|---|---|---|---|
| v1 | `f43ab5f39927b4cc5cb76afc39019a3231ae731cf5d9460b3eefea86b7b1b733` | 2,263 | `2026-09-18T00:18:32Z`, serial `0x08373BC8` | `2026-09-18T00:18:41Z`, serial `0xB423531AF2AC4CB9676B8384D1CBC7D7` |
| v2 | `48431130e8210c224b59ccb004437c9144cc34a5e10a7b0a9612a81be07fd4b7` | 2,347 | `2026-09-18T01:41:56Z`, serial `0x08378387` | `2026-09-18T01:41:56Z`, serial `0x1288DEA02372E343B3F4016A5372931D` |
| v3 | `1edb271e9fa77bddfc62d0548ceaddd9ea67f2a4dacaec304acc7d8ce6e596a4` | 2,481 | `2026-09-18T02:37:31Z`, serial `0x0837B36C` | `2026-09-18T02:37:32Z`, serial `0x56209E1F5EC3C66164ACF2EA8651460C` |
| v4 | `0d50afccaf7fb456ab23f14b0cbeb6f1f8720ce0383034f95d5c15b2867baaa4` | 2,531 | `2026-09-18T13:25:29Z`, serial `0x083A5A0E` | `2026-09-18T13:25:29Z`, serial `0x03D009C974F9E09501E09AFC89208E8E` |

All four roots use one convention: entries sorted by path; per entry `<path>` `0x00` `<sha256hex>` `\n`;
root = SHA-256 of the concatenation.

Two independent authorities are used deliberately: if either one's key is later compromised or distrusted, the other
still stands.

Anyone holding a token can verify it offline against the root:

```
openssl ts -verify -digest <root> -in <token>.tsr -CAfile <tsa-cacert>.pem -untrusted <tsa>.crt
```

and confirm the check is not vacuous by repeating it with a different digest, which must fail with
`message imprint mismatch`.

**The manifest and the tokens are not published.** They are retained; if the priority claim is ever challenged, the
manifest and the tokens are produced together and the challenger recomputes the root themselves.

## Ongoing practice

From this point on, each milestone is anchored before it is announced:

1. build a commitment manifest and compute its root under the declared convention;
2. obtain RFC 3161 timestamps from two independent authorities;
3. create an annotated tag (signed once a signing key is in place);
4. publish a GitHub Release;
5. optionally archive to Zenodo (DOI) and Software Heritage.

A cryptographic signature establishes **who**, not **when**; it is not a substitute for a timestamp. Both are kept.

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
was built on 2026-09-18 and timestamped by two independent RFC 3161 authorities:

| Item | Value |
|---|---|
| Manifest root (SHA-256) | `f43ab5f39927b4cc5cb76afc39019a3231ae731cf5d9460b3eefea86b7b1b733` |
| Files covered | 2,263 |
| Root convention | entries sorted by path; per entry `<path>` `0x00` `<sha256hex>` `\n`; root = SHA-256 of the concatenation |
| FreeTSA token | `2026-09-18T00:18:32Z`, serial `0x08373BC8` |
| DigiCert token | `2026-09-18T00:18:41Z`, serial `0xB423531AF2AC4CB9676B8384D1CBC7D7` |

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
3. create a signed annotated tag;
4. publish a GitHub Release;
5. optionally archive to Zenodo (DOI) and Software Heritage.

A cryptographic signature establishes **who**, not **when**; it is not a substitute for a timestamp. Both are kept.

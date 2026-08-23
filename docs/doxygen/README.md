# Building the documentation as HTML

The architecture documentation lives as Markdown in [`docs/architecture`](../architecture/README.md)
so it stays readable in the repository and renders on GitLab and GitHub. This directory holds
what is needed to turn those same sources into HTML.

## Quick start

```bash
make docs           # Doxygen site: API reference + architecture pages -> docs/html/index.html
make docs_single    # one self-contained page            -> docs/architecture.html
```

Both outputs are generated, so neither is committed.

## How it works

```
docs/architecture/*.md          the sources — the single point of truth
        |
        |  scripts/build_docs.py
        v
docs/doxygen/generated/*.md     preprocessed for Doxygen
        |
        |  doxygen Doxyfile
        v
docs/html/                      browsable site
```

`scripts/build_docs.py` needs nothing beyond the Python standard library, so it runs inside the
pinned `jpeg_ai_vm` environment. It has three targets:

| Target | Output | Purpose |
| --- | --- | --- |
| `doxygen` | `docs/doxygen/generated/` | Markdown rewritten for Doxygen |
| `single` | `docs/architecture.html` | One page, opens with a double click |
| `artifact` | wherever `-o` points | Body-only fragment for a host that supplies the page shell |

Two rewrites happen in the `doxygen` target:

- **Mermaid fences become `\htmlonly` blocks.** Doxygen turns an ordinary fenced block into one
  `<div>` per line, which no diagram renderer can read back, so the diagram source is passed
  through verbatim inside a `<pre class="mermaid">` element instead.
- **Cross-document links become `@ref` page references,** because the `.md` files do not exist in
  the generated output. Each page gets a stable `{#arch_…}` label for those references to resolve
  against.

## Diagrams

Doxygen has no Mermaid renderer, so [`assets/mermaid-init.js`](assets/mermaid-init.js) draws the
diagrams in the browser. It looks for a copy of Mermaid next to the HTML output first and falls
back to a CDN. If neither is reachable the blocks stay on screen as readable diagram source, so
an offline build is degraded rather than broken.

To make an offline build draw the diagrams properly, vendor Mermaid once:

```bash
python scripts/build_docs.py --vendor-mermaid     # -> docs/doxygen/assets/mermaid.min.js
```

`HTML_EXTRA_FILES` in the `Doxyfile` copies the assets into `docs/html/` at build time.

Alternatively the diagrams can be baked into the page as SVG, which removes the runtime
dependency entirely. This is the one part that needs more than the standard library — Playwright
and a local `mermaid.min.js`:

```bash
python scripts/build_docs.py single --prerender path/to/mermaid.min.js
```

## Doxyfile settings this relies on

| Setting | Value |
| --- | --- |
| `INPUT` | `src` and `docs/doxygen/generated` |
| `FILE_PATTERNS` | adds `*.md` |
| `USE_MDFILE_AS_MAINPAGE` | `docs/doxygen/generated/README.md` |
| `HTML_EXTRA_STYLESHEET` | `docs/doxygen/assets/extra.css` |
| `HTML_EXTRA_FILES` | `docs/doxygen/assets/mermaid-init.js` |
| `GENERATE_TREEVIEW` | `YES`, for the navigation sidebar |
| `TOC_INCLUDE_HEADINGS` | `3`, for a per-page table of contents |

## Editing the documentation

Edit the Markdown in `docs/architecture/` and rebuild. Two things to avoid, both of which break
Mermaid in the browser as well as in the generated pages:

- **No HTML entities inside a Mermaid block.** Write the characters directly, or avoid angle
  brackets in labels — `&lt;img&gt;` is a parse error in a sequence-diagram note.
- **No dots inside a dotted-link label.** `A -.text with a.dot.-> B` does not parse; use
  `A -.->|"text with a.dot"| B`.

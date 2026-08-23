#!/usr/bin/env python3
# The copyright in this software is being made available under the BSD
# License, included below. This software may be subject to other third party
# and contributor rights, including patent rights, and no such rights are
# granted under this license.
#
# Copyright (c) 2010-2022, ITU/ISO/IEC
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# * Neither the name of the ITU/ISO/IEC nor the names of its contributors may
# be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
# BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
# THE POSSIBILITY OF SUCH DAMAGE.

"""Build the architecture documentation in docs/architecture into HTML.

Three targets, all driven from the same Markdown sources so the sources stay
the single point of truth and keep rendering on GitLab/GitHub unchanged:

  doxygen   Preprocess the Markdown into docs/doxygen/generated so that
            `doxygen Doxyfile` picks the pages up alongside the API
            documentation.  Mermaid fences become \\htmlonly blocks that a
            small companion script renders in the browser.

  single    Emit docs/architecture.html: one self-contained page holding every
            chapter, openable with a double click.

  artifact  Emit the same page as a body-only fragment with no external
            scripts, for hosts that render Mermaid themselves.

Standard library only, Python 3.6+, so it runs inside the pinned jpeg_ai_vm
environment without adding a dependency.
"""

import argparse
import html
import os
import re
import sys

DOC_DIR = os.path.join('docs', 'architecture')
DOXY_GENERATED = os.path.join('docs', 'doxygen', 'generated')
DOXY_ASSETS = os.path.join('docs', 'doxygen', 'assets')
SINGLE_OUT = os.path.join('docs', 'architecture.html')

PAGE_TITLE = 'JPEG AI Codec Internals'
MERMAID_CDN = 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js'


# ######################################################################################################################
#  Source documents
# ######################################################################################################################
class Chapter:
    """One Markdown source file and the identifiers derived from its name."""

    def __init__(self, path):
        self.path = path
        self.filename = os.path.basename(path)
        stem = os.path.splitext(self.filename)[0]

        m = re.match(r'^(\d+)-(.*)$', stem)
        if m:
            self.number = m.group(1)
            self.slug = m.group(2)
        else:
            self.number = None
            self.slug = 'overview'

        self.anchor = 'ch-' + self.slug
        self.doxygen_label = 'arch_' + self.slug.replace('-', '_')

        with open(path, encoding='utf-8') as f:
            self.text = f.read()

        # The index page introduces the whole set, which the page masthead
        # already does, so it is listed as plain "Overview" rather than
        # repeating its own long heading.
        self.title = 'Overview' if self.number is None else self._read_title()

    def _read_title(self):
        for line in self.text.split('\n'):
            if line.startswith('# '):
                title = line[2:].strip()
                # "04 — Encoding Pipeline" -> "Encoding Pipeline"; the number is
                # carried separately so the page can set it as an eyebrow.
                return re.sub(r'^\d+\s*[—-]\s*', '', title)
        return self.slug.replace('-', ' ').title()


def read_chapters(doc_dir):
    """README first, then the numbered chapters in order."""
    if not os.path.isdir(doc_dir):
        sys.exit('error: no such directory: {}'.format(doc_dir))

    names = [n for n in os.listdir(doc_dir) if n.endswith('.md')]
    numbered = sorted(n for n in names if re.match(r'^\d', n))
    readme = [n for n in names if n.lower() == 'readme.md']
    ordered = readme + numbered

    if not ordered:
        sys.exit('error: no Markdown files found in {}'.format(doc_dir))

    return [Chapter(os.path.join(doc_dir, n)) for n in ordered]


# ######################################################################################################################
#  Markdown -> HTML
# ######################################################################################################################
SENTINEL = '\x00'


class Inline:
    """Inline Markdown, with code spans protected from every other rule.

    Code spans are lifted out before escaping so that their contents survive
    verbatim; everything else is escaped, then links and emphasis are applied.
    The store is shared across a call so a table row can be protected once and
    then split on the pipe character without cutting a span in half.
    """

    def __init__(self, link_resolver=None):
        self.store = []
        self.link_resolver = link_resolver or (lambda target: target)

    def protect(self, text):
        def take(m):
            self.store.append(m.group(1))
            return '{0}{1}{0}'.format(SENTINEL, len(self.store) - 1)

        return re.sub(r'`([^`]+)`', take, text)

    def finish(self, text):
        text = html.escape(text, quote=False)

        def link(m):
            label, target = m.group(1), m.group(2)
            target = self.link_resolver(target)
            external = target.startswith('http')
            attrs = ' target="_blank" rel="noopener"' if external else ''
            return '<a href="{}"{}>{}</a>'.format(html.escape(target, quote=True), attrs, label)

        text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', link, text)
        text = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', text)
        text = re.sub(r'(?<![\w*])\*([^*\s][^*]*?)\*(?![\w*])', r'<em>\1</em>', text)

        def restore(m):
            return '<code>{}</code>'.format(html.escape(self.store[int(m.group(1))], quote=False))

        return re.sub(SENTINEL + r'(\d+)' + SENTINEL, restore, text)

    def __call__(self, text):
        return self.finish(self.protect(text))


def slugify(text):
    text = re.sub(r'`([^`]*)`', r'\1', text)
    text = re.sub(r'[^\w\s-]', '', text.lower())
    return re.sub(r'[\s_]+', '-', text).strip('-') or 'section'


class MarkdownRenderer:
    """A renderer for the subset of Markdown these documents use.

    Headings, fenced code, Mermaid fences, pipe tables, ordered and unordered
    lists with indented continuations, block quotes, horizontal rules and
    paragraphs.  Nothing more is needed and nothing more is attempted.
    """

    def __init__(self, chapter, link_resolver=None, heading_offset=1):
        self.chapter = chapter
        self.link_resolver = link_resolver
        self.heading_offset = heading_offset
        self.headings = []

    def inline(self, text):
        return Inline(self.link_resolver)(text)

    def render(self, text):
        self.out = []
        lines = text.split('\n')
        i = 0
        n = len(lines)

        while i < n:
            line = lines[i]

            if line.startswith('```'):
                i = self._fence(lines, i)
                continue

            if re.match(r'^#{1,6} ', line):
                self._heading(line)
                i += 1
                continue

            if line.strip() in ('---', '***', '___'):
                self.out.append('<hr>')
                i += 1
                continue

            if line.startswith('|') and i + 1 < n and re.match(r'^\s*\|[\s:|-]+\|\s*$', lines[i + 1]):
                i = self._table(lines, i)
                continue

            if re.match(r'^(-|\d+\.) ', line):
                i = self._list(lines, i)
                continue

            if line.startswith('> '):
                i = self._quote(lines, i)
                continue

            if not line.strip():
                i += 1
                continue

            i = self._paragraph(lines, i)

        return '\n'.join(self.out)

    # ------------------------------------------------------------------ blocks
    def _fence(self, lines, i):
        lang = lines[i][3:].strip().lower()
        body = []
        i += 1
        while i < len(lines) and not lines[i].startswith('```'):
            body.append(lines[i])
            i += 1
        i += 1  # closing fence
        source = '\n'.join(body)

        if lang == 'mermaid':
            self.out.append(
                '<figure class="diagram"><pre class="mermaid">{}</pre></figure>'.format(
                    html.escape(source, quote=False)))
        else:
            cls = ' class="lang-{}"'.format(re.sub(r'[^\w-]', '', lang)) if lang else ''
            self.out.append('<div class="codeblock"><pre><code{}>{}</code></pre></div>'.format(
                cls, html.escape(source, quote=False)))
        return i

    def _heading(self, line):
        level = len(line) - len(line.lstrip('#'))
        text = line[level:].strip()
        # The chapter's own H1 is rendered by the page shell, not here.
        if level == 1:
            return
        anchor = '{}--{}'.format(self.chapter.anchor, slugify(text))
        if level == 2:
            self.headings.append((anchor, re.sub(r'^\d+\.\s*', '', text)))
        tag = 'h{}'.format(min(level + self.heading_offset - 1, 6))
        self.out.append('<{tag} id="{a}">{t}<a class="anchor" href="#{a}" aria-label="Link to this section">#</a></{tag}>'.format(
            tag=tag, a=anchor, t=self.inline(text)))

    def _table(self, lines, i):
        inline = Inline(self.link_resolver)

        def cells(row):
            row = inline.protect(row).strip()
            row = row.strip('|')
            return [inline.finish(c.strip()) for c in row.split('|')]

        head = cells(lines[i])
        i += 2
        body = []
        while i < len(lines) and lines[i].startswith('|'):
            body.append(cells(lines[i]))
            i += 1

        parts = ['<div class="tablewrap"><table>', '<thead><tr>']
        parts += ['<th>{}</th>'.format(c) for c in head]
        parts.append('</tr></thead><tbody>')
        for row in body:
            parts.append('<tr>' + ''.join('<td>{}</td>'.format(c) for c in row) + '</tr>')
        parts.append('</tbody></table></div>')
        self.out.append(''.join(parts))
        return i

    def _list(self, lines, i):
        ordered = bool(re.match(r'^\d+\. ', lines[i]))
        pattern = r'^\d+\.\s+' if ordered else r'^-\s+'
        items = []

        while i < len(lines) and re.match(pattern, lines[i]):
            items.append([re.sub(pattern, '', lines[i])])
            i += 1
            # Continuation lines are indented; blank lines end the item only if
            # the following line is not indented too.
            while i < len(lines):
                if lines[i].startswith('  ') and lines[i].strip():
                    items[-1].append(lines[i].strip())
                    i += 1
                elif not lines[i].strip() and i + 1 < len(lines) and lines[i + 1].startswith('  ') \
                        and lines[i + 1].strip():
                    items[-1].append('')
                    i += 1
                else:
                    break

        tag = 'ol' if ordered else 'ul'
        rendered = ''.join('<li>{}</li>'.format(self.inline(' '.join(p for p in parts if p)))
                           for parts in items)
        self.out.append('<{0}>{1}</{0}>'.format(tag, rendered))
        return i

    def _quote(self, lines, i):
        body = []
        while i < len(lines) and lines[i].startswith('>'):
            body.append(lines[i].lstrip('>').strip())
            i += 1
        self.out.append('<blockquote><p>{}</p></blockquote>'.format(self.inline(' '.join(body))))
        return i

    def _paragraph(self, lines, i):
        body = []
        while i < len(lines) and lines[i].strip() \
                and not lines[i].startswith(('```', '|', '> ')) \
                and not re.match(r'^#{1,6} ', lines[i]) \
                and not re.match(r'^(-|\d+\.) ', lines[i]) \
                and lines[i].strip() not in ('---', '***', '___'):
            body.append(lines[i].strip())
            i += 1
        if body:
            self.out.append('<p>{}</p>'.format(self.inline(' '.join(body))))
        return i


# ######################################################################################################################
#  Target: doxygen
# ######################################################################################################################
def emit_doxygen(chapters, out_dir):
    """Rewrite the Markdown so Doxygen can consume it.

    Two transformations.  Mermaid fences become \\htmlonly blocks holding a
    <pre class="mermaid"> element, because Doxygen turns an ordinary fence into
    per-line <div>s that no diagram renderer can read back.  Cross-document
    links become @ref page references, because the .md files do not exist in
    the generated output.
    """
    ensure_dir(out_dir)
    by_filename = {c.filename: c for c in chapters}

    for chapter in chapters:
        text = chapter.text
        used_mermaid = ['```mermaid' in text]

        def mermaid(m):
            used_mermaid[0] = True
            return ('\\htmlonly\n<pre class="mermaid">\n{}\n</pre>\n\\endhtmlonly'
                    .format(html.escape(m.group(1), quote=False)))

        text = re.sub(r'```mermaid\n(.*?)\n```', mermaid, text, flags=re.S)

        def ref(m):
            label, target = m.group(1), m.group(2)
            name = target.split('#')[0]
            if name in by_filename:
                return '[{}](@ref {})'.format(label, by_filename[name].doxygen_label)
            # A Markdown file outside this set has no page in the generated
            # output, and Doxygen would warn about the dangling reference, so
            # the link becomes plain text.
            return label

        text = re.sub(r'\[([^\]]+)\]\(([^)]+\.md(?:#[^)]*)?)\)', ref, text)

        # Give the page a stable identifier so the @ref links above resolve.
        text = re.sub(r'^# (.+)$', r'# \1 {#' + chapter.doxygen_label + '}', text, count=1,
                      flags=re.M)

        if used_mermaid[0]:
            text += ('\n\n\\htmlonly\n<script src="mermaid-init.js"></script>\n\\endhtmlonly\n')

        write(os.path.join(out_dir, chapter.filename), text)

    print('doxygen : {} pages -> {}'.format(len(chapters), out_dir))


# ######################################################################################################################
#  Target: single page
# ######################################################################################################################
def build_body(chapters):
    """Render every chapter into one document body."""
    by_filename = {c.filename: c for c in chapters}

    def resolve(target):
        name = target.split('#')[0]
        if name in by_filename:
            return '#' + by_filename[name].anchor
        return target

    sections = []
    nav = []

    for chapter in chapters:
        renderer = MarkdownRenderer(chapter, link_resolver=resolve)
        content = renderer.render(chapter.text)

        eyebrow = ('<p class="eyebrow">Chapter {}</p>'.format(chapter.number)
                   if chapter.number else '<p class="eyebrow">Start here</p>')
        sections.append(
            '<section class="chapter" id="{anchor}">\n'
            '<header class="chapter-head">{eyebrow}<h2>{title}</h2></header>\n'
            '{content}\n</section>'.format(
                anchor=chapter.anchor, eyebrow=eyebrow,
                title=html.escape(chapter.title, quote=False), content=content))

        subnav = ''.join(
            '<li><a href="#{}">{}</a></li>'.format(a, html.escape(t.replace('`', ''), quote=False))
            for a, t in renderer.headings)
        nav.append(
            '<li class="nav-chapter" data-target="{anchor}">'
            '<a class="nav-top" href="#{anchor}">'
            '<span class="nav-num">{num}</span><span class="nav-title">{title}</span></a>'
            '<ul class="nav-sub">{subnav}</ul></li>'.format(
                anchor=chapter.anchor, num=chapter.number or '—',
                title=html.escape(chapter.title, quote=False), subnav=subnav))

    return '\n'.join(sections), '\n'.join(nav)


STYLE = """
/* ---------------------------------------------------------------------------
   Palette derived from the YCbCr chrominance plane the codec works in:
   the Cb axis gives the indigo-violet accent, the Cr axis the amber
   secondary, and the neutrals carry a slight violet bias so they read as
   chosen rather than inherited.
   --------------------------------------------------------------------------- */
:root {
  --ground:       #f6f6fa;
  --surface:      #ffffff;
  --surface-sunk: #f0f0f6;
  --ink:          #17171f;
  --ink-muted:    #5b5c6e;
  --ink-faint:    #8a8b9c;
  --rule:         #e2e2ec;
  --rule-strong:  #cfcfdd;
  --accent:       #4a47b5;
  --accent-ink:   #3a3796;
  --accent-soft:  #ecebf9;
  --amber:        #a85f18;
  --amber-soft:   #fbf0e2;
  --shadow:       0 1px 2px rgba(23, 23, 31, .06), 0 8px 24px rgba(23, 23, 31, .05);
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --ground:       #101017;
    --surface:      #191922;
    --surface-sunk: #14141c;
    --ink:          #e7e7f0;
    --ink-muted:    #9d9db2;
    --ink-faint:    #74748a;
    --rule:         #2a2a37;
    --rule-strong:  #3a3a4b;
    --accent:       #a09df2;
    --accent-ink:   #b9b7f7;
    --accent-soft:  #232145;
    --amber:        #e0a061;
    --amber-soft:   #2e2314;
    --shadow:       0 1px 2px rgba(0, 0, 0, .4), 0 8px 24px rgba(0, 0, 0, .3);
  }
}
:root[data-theme="dark"] {
  --ground:       #101017;
  --surface:      #191922;
  --surface-sunk: #14141c;
  --ink:          #e7e7f0;
  --ink-muted:    #9d9db2;
  --ink-faint:    #74748a;
  --rule:         #2a2a37;
  --rule-strong:  #3a3a4b;
  --accent:       #a09df2;
  --accent-ink:   #b9b7f7;
  --accent-soft:  #232145;
  --amber:        #e0a061;
  --amber-soft:   #2e2314;
  --shadow:       0 1px 2px rgba(0, 0, 0, .4), 0 8px 24px rgba(0, 0, 0, .3);
}

*, *::before, *::after { box-sizing: border-box; }

body {
  margin: 0;
  background: var(--ground);
  color: var(--ink);
  font-family: "IBM Plex Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  font-size: 16px;
  line-height: 1.65;
  -webkit-font-smoothing: antialiased;
}

/* The masthead sits in the content column rather than spanning the page so its
   text aligns with the reading measure below it; the sidebar spans both rows. */
.layout { display: grid; grid-template-columns: 19rem minmax(0, 1fr); grid-template-rows: auto 1fr; }
.sidebar { grid-column: 1; grid-row: 1 / span 2; }
.masthead { grid-column: 2; grid-row: 1; }
main { grid-column: 2; grid-row: 2; }

/* ------------------------------------------------------------------ masthead */
.masthead {
  border-bottom: 1px solid var(--rule);
  background: var(--surface);
  padding: 2.9rem clamp(1.25rem, 4vw, 3.5rem) 2.3rem;
}
.masthead-inner { max-width: 46rem; margin: 0 auto; display: flex; flex-direction: column; gap: .9rem; }
.standard {
  font-family: "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: .74rem; letter-spacing: .11em; text-transform: uppercase;
  color: var(--accent); margin: 0;
}
.masthead h1 {
  font-family: "IBM Plex Serif", Georgia, "Times New Roman", serif;
  font-weight: 600; font-size: clamp(2rem, 4.5vw, 3rem); line-height: 1.1;
  margin: 0; letter-spacing: -.018em; text-wrap: balance;
}
.masthead p.lede {
  margin: 0; max-width: 62ch; color: var(--ink-muted); font-size: 1.05rem;
}
.factbar {
  display: flex; flex-wrap: wrap; gap: .5rem 1.75rem; margin: .5rem 0 0; padding: 0; list-style: none;
}
.factbar div { display: flex; flex-direction: column; gap: .1rem; }
.factbar dt {
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: .68rem; letter-spacing: .1em; text-transform: uppercase; color: var(--ink-faint);
}
.factbar dd {
  margin: 0; font-size: .95rem; font-weight: 500;
  font-variant-numeric: tabular-nums;
}

/* ----------------------------------------------------------------------- nav */
.sidebar {
  position: sticky; top: 0; align-self: start;
  min-height: 100vh; max-height: 100vh; overflow-y: auto;
  border-right: 1px solid var(--rule);
  background: var(--surface);
  padding: 2.9rem 1rem 4rem 1.5rem;
}
.sidebar h2 {
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: .68rem; letter-spacing: .12em; text-transform: uppercase;
  color: var(--ink-faint); margin: 0 0 .9rem;
}
.sidebar ul { list-style: none; margin: 0; padding: 0; }
.nav-chapter + .nav-chapter { margin-top: .1rem; }
.nav-top {
  display: grid; grid-template-columns: 1.9rem 1fr; align-items: baseline;
  gap: .25rem; padding: .3rem .4rem; border-radius: 5px;
  text-decoration: none; color: var(--ink-muted); font-size: .875rem; line-height: 1.35;
}
.nav-top:hover { background: var(--surface-sunk); color: var(--ink); }
.nav-num {
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: .72rem; color: var(--ink-faint); font-variant-numeric: tabular-nums;
}
.nav-chapter.active > .nav-top { color: var(--accent-ink); background: var(--accent-soft); font-weight: 500; }
.nav-chapter.active .nav-num { color: var(--accent); }
.nav-sub { display: none; margin: .15rem 0 .5rem 1.9rem; padding: 0 0 0 .7rem; border-left: 1px solid var(--rule); }
.nav-chapter.active .nav-sub { display: block; }
.nav-sub a {
  display: block; padding: .16rem 0; font-size: .8rem; color: var(--ink-faint); text-decoration: none;
}
.nav-sub a:hover { color: var(--accent); }

/* -------------------------------------------------------------------- content */
main { padding: 2.75rem clamp(1.25rem, 4vw, 3.5rem) 6rem; min-width: 0; }
.reading { max-width: 46rem; margin: 0 auto; }

.chapter { scroll-margin-top: 1.5rem; }
.chapter + .chapter { margin-top: 4.5rem; padding-top: 3rem; border-top: 1px solid var(--rule); }
.chapter-head { margin-bottom: 1.75rem; }
.eyebrow {
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: .7rem; letter-spacing: .12em; text-transform: uppercase;
  color: var(--accent); margin: 0 0 .35rem;
}
.chapter-head h2 {
  font-family: "IBM Plex Serif", Georgia, serif;
  font-size: clamp(1.6rem, 3vw, 2.15rem); font-weight: 600; line-height: 1.15;
  margin: 0; letter-spacing: -.015em; text-wrap: balance;
}

main h3, main h4, main h5 {
  font-family: "IBM Plex Serif", Georgia, serif;
  font-weight: 600; line-height: 1.25; text-wrap: balance; scroll-margin-top: 1.5rem;
}
main h3 { font-size: 1.28rem; margin: 2.6rem 0 .8rem; }
main h4 { font-size: 1.06rem; margin: 2rem 0 .6rem; }
main h5 { font-size: .95rem; margin: 1.6rem 0 .5rem; color: var(--ink-muted); }

.anchor {
  margin-left: .4rem; color: var(--rule-strong); text-decoration: none;
  opacity: 0; transition: opacity .15s ease; font-weight: 400;
}
h3:hover .anchor, h4:hover .anchor, h5:hover .anchor, .anchor:focus { opacity: 1; }

main p { margin: 0 0 1.05rem; }
main ul, main ol { margin: 0 0 1.15rem; padding-left: 1.35rem; }
main li { margin-bottom: .4rem; }
main li::marker { color: var(--ink-faint); }

a { color: var(--accent-ink); text-decoration-color: var(--rule-strong); text-underline-offset: 2px; }
a:hover { text-decoration-color: currentColor; }
:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; border-radius: 3px; }

strong { font-weight: 600; }
hr { border: 0; border-top: 1px solid var(--rule); margin: 2.5rem 0; }

blockquote {
  margin: 1.4rem 0; padding: .85rem 1.15rem;
  border-left: 3px solid var(--amber); background: var(--amber-soft); border-radius: 0 5px 5px 0;
}
blockquote p { margin: 0; }

/* ------------------------------------------------------------------ monospace */
code, pre, .lang-text {
  font-family: "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}
:not(pre) > code {
  background: var(--surface-sunk); border: 1px solid var(--rule);
  padding: .08em .34em; border-radius: 4px; font-size: .855em; word-break: break-word;
}
.codeblock {
  margin: 0 0 1.3rem; overflow-x: auto;
  background: var(--surface-sunk); border: 1px solid var(--rule); border-radius: 7px;
}
.codeblock pre { margin: 0; padding: .95rem 1.1rem; }
.codeblock code { font-size: .845rem; line-height: 1.6; white-space: pre; }

/* --------------------------------------------------------------------- tables */
.tablewrap {
  margin: 0 0 1.4rem; overflow-x: auto;
  border: 1px solid var(--rule); border-radius: 7px; background: var(--surface);
}
table { border-collapse: collapse; width: 100%; font-size: .875rem; }
th, td { text-align: left; padding: .55rem .8rem; border-bottom: 1px solid var(--rule); vertical-align: top; }
th {
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: .7rem; letter-spacing: .07em; text-transform: uppercase;
  color: var(--ink-muted); font-weight: 500; white-space: nowrap;
  background: var(--surface-sunk);
}
tbody tr:last-child td { border-bottom: 0; }
td code { font-size: .82em; }

/* ------------------------------------------------------------------- diagrams */
.diagram {
  margin: 0 0 1.5rem; padding: 1.1rem;
  background: #f7f7fb; border: 1px solid #dfdfe9; border-radius: 7px;
  box-shadow: var(--shadow); overflow-x: auto;
}
.diagram pre.mermaid {
  margin: 0; background: none; border: 0; text-align: center;
  font-size: .8rem; line-height: 1.5; color: #4a4b5c;
}
.diagram svg { max-width: 100%; height: auto; }

/* ------------------------------------------------------------------ responsive */
@media (max-width: 62rem) {
  .layout { grid-template-columns: minmax(0, 1fr); grid-template-rows: auto auto auto; }
  .masthead { grid-column: 1; grid-row: 1; }
  .sidebar {
    grid-column: 1; grid-row: 2;
    position: static; max-height: none; border-right: 0; border-bottom: 1px solid var(--rule);
    padding: 1.25rem clamp(1.25rem, 4vw, 3.5rem);
  }
  main { grid-column: 1; grid-row: 3; }
  .nav-sub { display: none !important; }
  .nav-top { grid-template-columns: 1.7rem 1fr; }
}

@media (prefers-reduced-motion: reduce) {
  * { animation: none !important; transition: none !important; scroll-behavior: auto !important; }
}
"""

SCROLLSPY = """
(function () {
  var chapters = [].slice.call(document.querySelectorAll('.chapter'));
  var navItems = {};
  [].forEach.call(document.querySelectorAll('.nav-chapter'), function (li) {
    navItems[li.getAttribute('data-target')] = li;
  });
  if (!chapters.length) { return; }

  var current = null;
  function activate(id) {
    if (id === current) { return; }
    if (current && navItems[current]) { navItems[current].classList.remove('active'); }
    current = id;
    if (navItems[id]) {
      navItems[id].classList.add('active');
      var top = navItems[id].getBoundingClientRect().top;
      var bar = navItems[id].parentNode.parentNode;
      if (bar.scrollHeight > bar.clientHeight && (top < 80 || top > bar.clientHeight - 80)) {
        navItems[id].scrollIntoView({ block: 'nearest' });
      }
    }
  }

  function update() {
    var best = chapters[0].id;
    for (var i = 0; i < chapters.length; i++) {
      if (chapters[i].getBoundingClientRect().top <= 120) { best = chapters[i].id; }
    }
    activate(best);
  }

  var ticking = false;
  window.addEventListener('scroll', function () {
    if (ticking) { return; }
    ticking = true;
    window.requestAnimationFrame(function () { update(); ticking = false; });
  }, { passive: true });
  update();
})();
"""


def masthead(chapters, stats):
    facts = ''.join(
        '<div><dt>{}</dt><dd>{}</dd></div>'.format(html.escape(k), html.escape(v))
        for k, v in stats)
    return (
        '<header class="masthead"><div class="masthead-inner">'
        '<p class="standard">Rec. ITU-T T.840.1 &nbsp;|&nbsp; ISO/IEC 6048-1</p>'
        '<h1>{title}</h1>'
        '<p class="lede">How the JPEG AI reference software is put together: the engine and tool '
        'framework, the encoding and decoding pipelines, the bitstream, the entropy coder, the '
        'neural transforms, and every coding tool.</p>'
        '<dl class="factbar">{facts}</dl>'
        '</div></header>'.format(title=html.escape(PAGE_TITLE), facts=facts))


# Prefer a vendored copy so an offline build still draws diagrams; fall back to
# the CDN; if neither arrives the <pre class="mermaid"> blocks stay on screen as
# readable diagram source, which is a usable degradation rather than a blank.
MERMAID_LOADER = """
(function () {
  function init() {
    if (window.mermaid) {
      window.mermaid.initialize({ startOnLoad: true, securityLevel: 'loose', theme: 'neutral' });
    }
  }
  function load(src, onerror) {
    var s = document.createElement('script');
    s.src = src;
    s.onload = init;
    s.onerror = onerror;
    document.body.appendChild(s);
  }
  load('mermaid.min.js', function () { load('@CDN@', null); });
})();
"""

FONTS = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">\n'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n'
    '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
    'family=IBM+Plex+Mono:wght@400;500&'
    'family=IBM+Plex+Sans:wght@400;500;600&'
    'family=IBM+Plex+Serif:wght@600&display=swap">'
)


def emit_single(chapters, out_path, standalone=True, prerender=None):
    """Render every chapter into one page.

    A standalone page is a complete document that pulls Mermaid from a CDN (or
    a vendored copy beside it).  The artifact variant is the same content with
    no document scaffolding and no external script, for a host that supplies
    the <head> and renders <pre class="mermaid"> itself.
    """
    body, nav = build_body(chapters)
    diagrams = sum(c.text.count('```mermaid') for c in chapters)
    stats = [
        ('Chapters', str(len(chapters))),
        ('Diagrams', str(diagrams)),
        ('Source', 'docs/architecture'),
        ('Release', release_version()),
    ]

    head = '<title>{title}</title>\n{fonts}\n<style>{style}</style>'.format(
        title=html.escape(PAGE_TITLE), fonts=FONTS, style=STYLE)

    content = (
        '{masthead}\n'
        '<div class="layout">\n'
        '<nav class="sidebar" aria-label="Contents"><h2>Contents</h2><ul>{nav}</ul></nav>\n'
        '<main><div class="reading">{body}</div></main>\n'
        '</div>\n'
        '<script>{spy}</script>'
    ).format(masthead=masthead(chapters, stats), nav=nav, body=body, spy=SCROLLSPY)

    if standalone:
        page = (
            '<!doctype html>\n<html lang="en">\n<head>\n'
            '<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            '{head}\n'
            '</head>\n<body>\n{content}\n'
            '<script>{loader}</script>\n'
            '</body>\n</html>\n'
        ).format(head=head, content=content, loader=MERMAID_LOADER.replace('@CDN@', MERMAID_CDN))
    else:
        page = head + '\n' + content + '\n'

    if prerender:
        page = prerender_mermaid(page, prerender['mermaid_js'], prerender.get('chromium'))

    write(out_path, page)
    print('{:9}: {} chapters, {} diagrams -> {} ({:.0f} KB)'.format(
        'single' if standalone else 'artifact', len(chapters), diagrams, out_path,
        os.path.getsize(out_path) / 1024.0))


# ######################################################################################################################
#  Optional: bake the diagrams into the page
# ######################################################################################################################
def prerender_mermaid(page, mermaid_js, chromium=None):
    """Replace every <pre class="mermaid"> with the SVG Mermaid renders for it.

    Optional, and the only part of this script that needs anything beyond the
    standard library: Playwright and a local copy of mermaid.min.js.  Worth it
    for a page that will be read somewhere the diagrams cannot be rendered on
    the fly, since it makes the result self-contained and removes any
    dependency on what the host does with a Mermaid block.
    """
    from playwright.sync_api import sync_playwright  # noqa: I001  (optional dependency)

    blocks = list(re.finditer(r'<pre class="mermaid">(.*?)</pre>', page, re.S))
    if not blocks:
        return page

    launch = {'args': ['--no-sandbox']}
    if chromium:
        launch['executable_path'] = chromium

    svgs = []
    with sync_playwright() as p:
        browser = p.chromium.launch(**launch)
        tab = browser.new_page()
        tab.goto('about:blank')
        tab.add_script_tag(path=mermaid_js)
        tab.wait_for_function('() => !!window.mermaid', timeout=60000)
        tab.evaluate("window.mermaid.initialize("
                     "{startOnLoad:false, securityLevel:'loose', theme:'neutral'})")
        for i, m in enumerate(blocks):
            source = html.unescape(m.group(1))
            svgs.append(tab.evaluate(
                "async (a) => (await window.mermaid.render(a[0], a[1])).svg",
                ['prerendered-{}'.format(i), source]))
        browser.close()

    out, last = [], 0
    for m, svg in zip(blocks, svgs):
        out.append(page[last:m.start()])
        out.append(svg)
        last = m.end()
    out.append(page[last:])
    print('prerender: {} diagrams baked in as SVG'.format(len(svgs)))
    return ''.join(out)


def release_version():
    try:
        import json
        with open(os.path.join('cfg', 'info.json'), encoding='utf-8') as f:
            return json.load(f).get('version', 'unknown')
    except Exception:
        return 'unknown'


# ######################################################################################################################
#  Helpers
# ######################################################################################################################
def ensure_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def write(path, text):
    ensure_dir(os.path.dirname(path) or '.')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)


def vendor_mermaid(dest):
    """Download Mermaid next to the generated pages so the build works offline."""
    try:
        from urllib.request import urlopen
    except ImportError:
        from urllib2 import urlopen  # noqa: F401  (Python 2 is not supported, but be explicit)
    ensure_dir(os.path.dirname(dest) or '.')
    print('fetching {}'.format(MERMAID_CDN))
    data = urlopen(MERMAID_CDN, timeout=60).read()
    with open(dest, 'wb') as f:
        f.write(data)
    print('vendored : {} ({:.0f} KB)'.format(dest, len(data) / 1024.0))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('target', nargs='?', default='all',
                        choices=['all', 'doxygen', 'single', 'artifact'],
                        help='which output to produce (default: all)')
    parser.add_argument('--doc-dir', default=DOC_DIR, help='directory holding the Markdown sources')
    parser.add_argument('-o', '--output', default=None, help='override the output path')
    parser.add_argument('--vendor-mermaid', action='store_true',
                        help='download Mermaid into docs/doxygen/assets for offline builds')
    parser.add_argument('--prerender', metavar='MERMAID_JS', default=None,
                        help='bake the diagrams in as SVG using this mermaid.min.js '
                             '(needs Playwright; applies to the single and artifact targets)')
    parser.add_argument('--chromium', default=os.environ.get('PLAYWRIGHT_CHROMIUM'),
                        help='browser executable for --prerender')
    args = parser.parse_args()

    prerender = ({'mermaid_js': args.prerender, 'chromium': args.chromium}
                 if args.prerender else None)

    chapters = read_chapters(args.doc_dir)

    if args.target in ('all', 'doxygen'):
        emit_doxygen(chapters, args.output if args.target == 'doxygen' and args.output
                     else DOXY_GENERATED)
    if args.target in ('all', 'single'):
        emit_single(chapters, args.output if args.target == 'single' and args.output
                    else SINGLE_OUT, standalone=True, prerender=prerender)
    if args.target == 'artifact':
        emit_single(chapters, args.output or 'architecture-artifact.html', standalone=False,
                    prerender=prerender)

    if args.vendor_mermaid:
        vendor_mermaid(os.path.join(DOXY_ASSETS, 'mermaid.min.js'))


if __name__ == '__main__':
    main()

/* The copyright in this software is being made available under the BSD
 License, included below. This software may be subject to other third party
 and contributor rights, including patent rights, and no such rights are
 granted under this license.

 Copyright (c) 2010-2022, ITU/ISO/IEC
 All rights reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice,
 this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright notice,
 this list of conditions and the following disclaimer in the documentation
 and/or other materials provided with the distribution.
 * Neither the name of the ITU/ISO/IEC nor the names of its contributors may
 be used to endorse or promote products derived from this software without
 specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
 BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
 THE POSSIBILITY OF SUCH DAMAGE. */

/* Renders the <pre class="mermaid"> blocks that scripts/build_docs.py leaves in
   the generated Doxygen pages.

   Doxygen has no diagram renderer of its own for Mermaid, so the pages carry
   the diagram source and this script turns it into SVG in the browser.  It
   prefers a copy vendored next to the HTML output and falls back to the CDN;
   when neither is reachable the blocks stay on screen as readable diagram
   source, so an offline build is degraded rather than broken. */

(function () {
    var CDN = 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js';

    function hasDiagrams() {
        return document.querySelector('pre.mermaid') !== null;
    }

    function pinNaturalWidth() {
        // Mermaid emits width="100%", which shrinks a diagram wider than the
        // text column until its labels are unreadable.  Pin each diagram to its
        // natural width instead and let pre.mermaid scroll.
        var svgs = document.querySelectorAll('pre.mermaid svg');
        for (var i = 0; i < svgs.length; i++) {
            if (svgs[i].style.maxWidth) {
                svgs[i].style.width = svgs[i].style.maxWidth;
            }
        }
    }

    function init() {
        if (!window.mermaid) {
            return;
        }
        window.mermaid.initialize({
            startOnLoad: false,
            securityLevel: 'loose',
            theme: 'neutral'
        });
        window.mermaid
            .run({ querySelector: 'pre.mermaid' })
            .then(pinNaturalWidth, pinNaturalWidth);
    }

    function note() {
        var blocks = document.querySelectorAll('pre.mermaid');
        for (var i = 0; i < blocks.length; i++) {
            if (blocks[i].previousElementSibling &&
                blocks[i].previousElementSibling.className === 'mermaid-note') {
                continue;
            }
            var p = document.createElement('p');
            p.className = 'mermaid-note';
            p.textContent = 'Diagram source (Mermaid could not be loaded).';
            blocks[i].parentNode.insertBefore(p, blocks[i]);
        }
    }

    function load(src, onerror) {
        var s = document.createElement('script');
        s.src = src;
        s.onload = init;
        s.onerror = onerror;
        document.body.appendChild(s);
    }

    function boot() {
        if (!hasDiagrams()) {
            return;
        }
        load('mermaid.min.js', function () {
            load(CDN, note);
        });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', boot);
    } else {
        boot();
    }
})();

// renderer/waterfall.js
//
// Waterfall — a source-agnostic, GPU-accelerated spectrum waterfall.
//
// Consumes a stream of FFT magnitude frames via pushFrame() from ANY source
// (audio-passband FFT for conventional radios, wideband panadapter FFT for a
// Flex) and renders them as a scrolling heat-map on a WebGL canvas.
//
// Rendering technique — borrowed in concept from AetherSDR's QRhi waterfall,
// reimplemented in WebGL: the scrollback history is a fixed ring-buffer
// texture. Each pushFrame() writes ONE row via texSubImage2D and advances a
// rowOffset uniform; the fragment shader wraps the vertical UV with
// fract(uv.y + rowOffset), so the whole display scrolls with no per-frame
// redraw and no pixel copying.
//
// WebGL1, not WebGL2 — deliberately. Confirmed via a direct GPU diagnostic
// (app.getGPUInfo() + a real getContext('webgl2') probe) that this project's
// own ARM SBC target rejects WebGL2 outright ("ES3 is blocklisted/disabled/
// unsupported by driver", glImplementationParts "gl=none,angle=none") while
// WebGL1 works fine there. A prior WebGL2 version of this file rendered
// completely blank on that hardware (2026-09-03). WebGL1 has no real
// downside for what this component actually needs (a textured quad + a
// couple of lines), so there's no reason to special-case WebGL2 on machines
// that do support it — always requesting 'webgl' is simplest and most
// portable. One real constraint this brings: WebGL1 requires power-of-two
// texture dimensions for REPEAT wrapping (used on the ring buffer's row
// axis) — historyRows must stay a power of two (256, 512, ...); bins (the
// other axis, CLAMP_TO_EDGE) has no such restriction.
//
// Loaded via <script> (POTACAT's renderer has no ES modules) — defines the
// global `Waterfall` class. See docs/waterfall-plan.md.

(function (global) {
  'use strict';

  const WF_VERT = `
attribute vec2 a_pos;
varying vec2 v_uv;
void main() {
  v_uv = a_pos * 0.5 + 0.5;
  gl_Position = vec4(a_pos, 0.0, 1.0);
}`;

  // Newest row sits near the top and the display scrolls down: as a frame is
  // pushed, rowOffset grows, so a fixed screen pixel samples a row 1/H lower.
  const WF_FRAG = `
precision highp float;
varying vec2 v_uv;
uniform sampler2D u_tex;
uniform float u_rowOffset;
uniform int u_colormap;

vec3 cmClassic(float t) {
  t = clamp(t, 0.0, 1.0);
  if (t < 0.2) return mix(vec3(0.0),              vec3(0.0,0.0,0.5), t/0.2);
  if (t < 0.4) return mix(vec3(0.0,0.0,0.5),      vec3(0.0,0.8,1.0), (t-0.2)/0.2);
  if (t < 0.6) return mix(vec3(0.0,0.8,1.0),      vec3(0.0,1.0,0.0), (t-0.4)/0.2);
  if (t < 0.8) return mix(vec3(0.0,1.0,0.0),      vec3(1.0,1.0,0.0), (t-0.6)/0.2);
  return            mix(vec3(1.0,1.0,0.0),        vec3(1.0,1.0,1.0), (t-0.8)/0.2);
}
vec3 cmTurbo(float t) {
  t = clamp(t, 0.0, 1.0);
  if (t < 0.25) return mix(vec3(0.12,0.06,0.28), vec3(0.0,0.42,0.86),  t/0.25);
  if (t < 0.50) return mix(vec3(0.0,0.42,0.86),  vec3(0.0,0.86,0.70),  (t-0.25)/0.25);
  if (t < 0.70) return mix(vec3(0.0,0.86,0.70),  vec3(0.62,0.95,0.22), (t-0.50)/0.20);
  if (t < 0.88) return mix(vec3(0.62,0.95,0.22), vec3(0.98,0.73,0.05), (t-0.70)/0.18);
  return            mix(vec3(0.98,0.73,0.05),    vec3(0.96,0.16,0.10), (t-0.88)/0.12);
}
void main() {
  float tv = fract(v_uv.y + u_rowOffset);
  // LUMINANCE texture (WebGL1 has no R8) — value lands in .r (and is
  // replicated across .g/.b by the format; only .r is read here).
  float m = texture2D(u_tex, vec2(v_uv.x, tv)).r;
  vec3 c = (u_colormap == 1) ? cmTurbo(m) : cmClassic(m);
  gl_FragColor = vec4(c, 1.0);
}`;

  const LINE_VERT = `
attribute vec2 a_pos;
void main() { gl_Position = vec4(a_pos, 0.0, 1.0); }`;

  const LINE_FRAG = `
precision highp float;
uniform vec4 u_color;
void main() { gl_FragColor = u_color; }`;

  function compile(gl, type, src) {
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
      const log = gl.getShaderInfoLog(s);
      gl.deleteShader(s);
      throw new Error('shader compile: ' + log);
    }
    return s;
  }
  function buildProgram(gl, vsSrc, fsSrc) {
    const vs = compile(gl, gl.VERTEX_SHADER, vsSrc);
    const fs = compile(gl, gl.FRAGMENT_SHADER, fsSrc);
    const p = gl.createProgram();
    gl.attachShader(p, vs);
    gl.attachShader(p, fs);
    gl.linkProgram(p);
    gl.deleteShader(vs);
    gl.deleteShader(fs);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
      const log = gl.getProgramInfoLog(p);
      gl.deleteProgram(p);
      throw new Error('program link: ' + log);
    }
    return p;
  }

  class Waterfall {
    /**
     * @param {HTMLCanvasElement} canvas - a canvas with no prior 2D context.
     * @param {object} opts - { bins, historyRows, colormap, gamma,
     *   floorSmoothing, peakDecay, peakAttack }
     *   historyRows MUST be a power of two (WebGL1 REPEAT constraint).
     *
     *   floorSmoothing/peakDecay/peakAttack default to values tuned for a
     *   ~60fps pushFrame() rate (the desktop popout's real rate). A caller
     *   pushing much slower (e.g. ECHOCAT's ~10fps spectrum broadcast)
     *   needs LARGER per-push floorSmoothing/peakAttack (and a SMALLER
     *   peakDecay, i.e. more per-push release) to reach the same real-time
     *   convergence speed — otherwise the adaptive range takes ~6x longer
     *   in wall-clock time to settle, which reads as a background that
     *   never quite goes dark even with real gain changes (confirmed live:
     *   bad2.png/bad3.png, 2026-09-03 — background stayed a solid mid-blue
     *   regardless of RX gain because the floor simply hadn't caught up
     *   yet). See renderer/remote.js's ft8Wf construction for the
     *   10fps-matched constants (derived from an exponential decay
     *   time-constant match, not guessed).
     */
    constructor(canvas, opts = {}) {
      this.canvas = canvas;
      this.bins = Math.max(16, opts.bins || 1024);
      this.historyRows = Math.max(16, opts.historyRows || 512);
      this.gamma = opts.gamma || 0.5;
      this._floorSmoothing = opts.floorSmoothing != null ? opts.floorSmoothing : 0.05;
      this._peakDecay = opts.peakDecay != null ? opts.peakDecay : 0.97;
      this._peakAttack = opts.peakAttack != null ? opts.peakAttack : 0.03;
      this._cmIndex = opts.colormap === 'turbo' ? 1 : 0;
      this._markers = [];
      this._clickCb = null;

      // Adaptive range — slow noise floor, fast-attack / slow-decay peak.
      this._floor = 0;
      this._peak = 1;
      this._rangeInit = false;

      // Ring buffer.
      this._writeRow = 0;
      this._rowU8 = new Uint8Array(this.bins);

      const gl = canvas.getContext('webgl', {
        antialias: false, depth: false, premultipliedAlpha: false,
      });
      this.supported = !!gl;
      if (!gl) {
        console.warn('[Waterfall] WebGL unavailable — waterfall disabled');
        return;
      }
      this.gl = gl;
      try {
        this._initGL();
      } catch (e) {
        console.warn('[Waterfall] init failed:', e.message);
        this.supported = false;
        this.gl = null;
        return;
      }
      this._onClick = (e) => this._handleClick(e);
      canvas.addEventListener('click', this._onClick);
      if (typeof ResizeObserver !== 'undefined') {
        this._ro = new ResizeObserver(() => this.resize());
        this._ro.observe(canvas);
      }
      this.resize();
    }

    _initGL() {
      const gl = this.gl;
      this._wfProg = buildProgram(gl, WF_VERT, WF_FRAG);
      this._lineProg = buildProgram(gl, LINE_VERT, LINE_FRAG);

      // Fullscreen quad (two triangles).
      this._quad = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, this._quad);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
        -1, -1,  1, -1,  -1, 1,
        -1,  1,  1, -1,   1, 1,
      ]), gl.STATIC_DRAW);
      this._lineBuf = gl.createBuffer();

      // Ring-buffer history texture: LUMINANCE (WebGL1 has no sized R8
      // internal format — LUMINANCE is the single-channel 8-bit equivalent
      // core WebGL1 actually supports), bins wide x historyRows tall.
      this._tex = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, this._tex);
      gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.LUMINANCE, this.bins, this.historyRows, 0,
                    gl.LUMINANCE, gl.UNSIGNED_BYTE, null);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT); // wraps the ring — needs historyRows POT

      this._uTex = gl.getUniformLocation(this._wfProg, 'u_tex');
      this._uRowOffset = gl.getUniformLocation(this._wfProg, 'u_rowOffset');
      this._uColormap = gl.getUniformLocation(this._wfProg, 'u_colormap');
      this._wfPos = gl.getAttribLocation(this._wfProg, 'a_pos');
      this._uLineColor = gl.getUniformLocation(this._lineProg, 'u_color');
      this._linePos = gl.getAttribLocation(this._lineProg, 'a_pos');
    }

    /** Feed one FFT frame. `mags` — Float32Array of bin magnitudes, any scale.
     *  Frames of any length are resampled to `bins`; the component auto-ranges. */
    pushFrame(mags) {
      if (!this.supported || !mags || !mags.length) return;
      const n = this.bins;
      const step = mags.length / n;
      const tmp = this._frameTmp || (this._frameTmp = new Float32Array(n));
      let lo = Infinity, hi = -Infinity;
      for (let i = 0; i < n; i++) {
        const v = mags[(i * step) | 0];
        tmp[i] = v;
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
      if (!isFinite(lo) || !isFinite(hi)) return;

      if (!this._rangeInit) {
        this._floor = lo; this._peak = hi; this._rangeInit = true;
      } else {
        this._floor += this._floorSmoothing * (lo - this._floor);
        if (hi >= this._peak) {
          // Fast but SMOOTH attack, not an instant snap. The instant
          // Math.max(...) this replaces gave the row that first caught a
          // new high value a discontinuously different effective range
          // than its immediate neighbors — confirmed live as the cause of
          // a "narrow in the middle, fans out at the edges" artifact on
          // every trace's onset/offset (bad4.png/bad5.png, 2026-09-03,
          // reproduced at normal RX gain so it wasn't a saturation
          // artifact): the same underlying signal's box-averaged
          // bleed-over into neighboring bins crossed the visibility
          // threshold inconsistently row-to-row because the range itself
          // wasn't stable row-to-row. 0.5 still converges within ~1-2
          // pushes (not sluggish), just without the single-frame jump.
          this._peak += 0.5 * (hi - this._peak);
        } else {
          this._peak = this._peak * this._peakDecay + hi * this._peakAttack;
        }
      }
      const range = Math.max(1e-6, this._peak - this._floor);
      const g = this.gamma;
      for (let i = 0; i < n; i++) {
        let t = (tmp[i] - this._floor) / range;
        t = t <= 0 ? 0 : t >= 1 ? 1 : Math.pow(t, g);
        this._rowU8[i] = (t * 255) | 0;
      }

      const gl = this.gl;
      gl.bindTexture(gl.TEXTURE_2D, this._tex);
      gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, this._writeRow, n, 1,
                       gl.LUMINANCE, gl.UNSIGNED_BYTE, this._rowU8);
      this._writeRow = (this._writeRow + 1) % this.historyRows;
      this._render();
    }

    _render() {
      const gl = this.gl;
      if (!gl) return;
      gl.viewport(0, 0, this.canvas.width, this.canvas.height);
      gl.clearColor(0, 0, 0, 1);
      gl.clear(gl.COLOR_BUFFER_BIT);

      // Waterfall.
      gl.useProgram(this._wfProg);
      gl.bindBuffer(gl.ARRAY_BUFFER, this._quad);
      gl.enableVertexAttribArray(this._wfPos);
      gl.vertexAttribPointer(this._wfPos, 2, gl.FLOAT, false, 0, 0);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this._tex);
      gl.uniform1i(this._uTex, 0);
      gl.uniform1i(this._uColormap, this._cmIndex);
      gl.uniform1f(this._uRowOffset, this._writeRow / this.historyRows);
      gl.drawArrays(gl.TRIANGLES, 0, 6);

      // RX/TX marker lines.
      if (this._markers.length) {
        gl.useProgram(this._lineProg);
        gl.bindBuffer(gl.ARRAY_BUFFER, this._lineBuf);
        gl.enableVertexAttribArray(this._linePos);
        gl.vertexAttribPointer(this._linePos, 2, gl.FLOAT, false, 0, 0);
        for (const m of this._markers) {
          const x = (m.pos || 0) * 2 - 1;
          gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([x, -1, x, 1]), gl.DYNAMIC_DRAW);
          const c = m.color || [1, 1, 1, 1];
          gl.uniform4f(this._uLineColor, c[0], c[1], c[2], c[3] == null ? 1 : c[3]);
          gl.drawArrays(gl.LINES, 0, 2);
        }
      }
    }

    /** markers: [{ pos: 0..1, color: [r,g,b,a] }] — e.g. RX/TX cursors. */
    setMarkers(markers) {
      this._markers = Array.isArray(markers) ? markers : [];
      this._render();
    }

    setColormap(name) {
      this._cmIndex = name === 'turbo' ? 1 : 0;
      this._render();
    }

    /** Register a click-to-tune callback. cb receives the click x as 0..1. */
    onClick(cb) { this._clickCb = cb; }

    _handleClick(e) {
      if (!this._clickCb) return;
      const r = this.canvas.getBoundingClientRect();
      if (r.width <= 0) return;
      let p = (e.clientX - r.left) / r.width;
      p = p < 0 ? 0 : p > 1 ? 1 : p;
      this._clickCb(p);
    }

    /** Match the canvas backing store to its CSS size x devicePixelRatio. */
    resize() {
      if (!this.supported) return;
      const dpr = window.devicePixelRatio || 1;
      const w = Math.max(1, Math.round(this.canvas.clientWidth * dpr));
      const h = Math.max(1, Math.round(this.canvas.clientHeight * dpr));
      if (this.canvas.width !== w || this.canvas.height !== h) {
        this.canvas.width = w;
        this.canvas.height = h;
      }
      this._render();
    }

    destroy() {
      if (this._onClick) this.canvas.removeEventListener('click', this._onClick);
      if (this._ro) { this._ro.disconnect(); this._ro = null; }
      const gl = this.gl;
      if (gl) {
        gl.deleteTexture(this._tex);
        gl.deleteBuffer(this._quad);
        gl.deleteBuffer(this._lineBuf);
        gl.deleteProgram(this._wfProg);
        gl.deleteProgram(this._lineProg);
      }
      this.gl = null;
      this.supported = false;
    }
  }

  global.Waterfall = Waterfall;
})(typeof window !== 'undefined' ? window : this);

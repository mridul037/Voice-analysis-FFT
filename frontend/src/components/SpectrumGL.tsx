import { useCallback, useEffect, useRef, useState, type CSSProperties } from "react";

type FftMessage = {
  type: "fft";
  sampleRate: number;
  frameSize: number;
  bins: number[];
};

const VS = `#version 300 es
in vec2 a_pos;
void main() {
  gl_Position = vec4(a_pos, 0.0, 1.0);
}`;

const FS = `#version 300 es
precision highp float;
uniform sampler2D u_bins;
uniform vec2 u_resolution;
uniform float u_binCount;
out vec4 oColor;

void main() {
  vec2 p = gl_FragCoord.xy;
  float xN = p.x / u_resolution.x;
  float mag = texture(u_bins, vec2(xN, 0.5)).r;
  float h = mag * (u_resolution.y - 8.0);
  float y = p.y;
  float barBottom = 4.0;
  float t = smoothstep(h + barBottom, h + barBottom - 2.0, y);
  vec3 topCol = vec3(0.35, 0.85, 1.0);
  vec3 botCol = vec3(0.05, 0.12, 0.22);
  vec3 fill = mix(botCol, topCol, sqrt(mag));
  vec3 bg = vec3(0.06, 0.07, 0.09);
  vec3 col = mix(bg, fill, t);
  oColor = vec4(col, 1.0);
}`;

function compile(gl: WebGL2RenderingContext, type: number, src: string): WebGLShader {
  const sh = gl.createShader(type)!;
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(sh) ?? "compile error";
    gl.deleteShader(sh);
    throw new Error(log);
  }
  return sh;
}

function linkProgram(gl: WebGL2RenderingContext, vs: WebGLShader, fs: WebGLShader): WebGLProgram {
  const prog = gl.createProgram()!;
  gl.attachShader(prog, vs);
  gl.attachShader(prog, fs);
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    const log = gl.getProgramInfoLog(prog) ?? "link error";
    gl.deleteProgram(prog);
    throw new Error(log);
  }
  return prog;
}

function setupGl(canvas: HTMLCanvasElement): {
  gl: WebGL2RenderingContext;
  prog: WebGLProgram;
  vao: WebGLVertexArrayObject;
  tex: WebGLTexture;
  loc: { bins: WebGLUniformLocation; resolution: WebGLUniformLocation; binCount: WebGLUniformLocation };
} {
  const gl = canvas.getContext("webgl2", { antialias: true, alpha: false });
  if (!gl) throw new Error("WebGL2 not available");

  const vShader = compile(gl, gl.VERTEX_SHADER, VS);
  const fShader = compile(gl, gl.FRAGMENT_SHADER, FS);
  const prog = linkProgram(gl, vShader, fShader);
  gl.deleteShader(vShader);
  gl.deleteShader(fShader);

  const vao = gl.createVertexArray()!;
  gl.bindVertexArray(vao);
  const buf = gl.createBuffer()!;
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  const tri = new Float32Array([-1, -1, 3, -1, -1, 3]);
  gl.bufferData(gl.ARRAY_BUFFER, tri, gl.STATIC_DRAW);
  const locPos = gl.getAttribLocation(prog, "a_pos");
  gl.enableVertexAttribArray(locPos);
  gl.vertexAttribPointer(locPos, 2, gl.FLOAT, false, 0, 0);
  gl.bindVertexArray(null);

  const tex = gl.createTexture()!;
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.bindTexture(gl.TEXTURE_2D, null);

  const bins = gl.getUniformLocation(prog, "u_bins")!;
  const resolution = gl.getUniformLocation(prog, "u_resolution")!;
  const binCount = gl.getUniformLocation(prog, "u_binCount")!;

  return { gl, prog, vao, tex, loc: { bins: bins, resolution, binCount: binCount } };
}

const BIN_COUNT = 256;
/** `fftSize` 512 → `frequencyBinCount` 256, matches `BIN_COUNT` and the WebGL texture width. */
const ANALYSER_FFT_SIZE = 512;

type MicSession = {
  ctx: AudioContext;
  stream: MediaStream;
  analyser: AnalyserNode;
  freqBytes: Uint8Array<ArrayBuffer>;
};

function fillBinsFromFrequencyBytes(
  bins: Float32Array,
  freqBytes: ArrayLike<number>,
  binCount: number,
): void {
  const n = Math.min(binCount, freqBytes.length);
  for (let i = 0; i < n; i++) {
    const v = freqBytes[i]! / 255;
    bins[i] = Math.pow(v, 0.55);
  }
  for (let i = n; i < binCount; i++) bins[i] = 0;
}

export function SpectrumGL() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const glState = useRef<ReturnType<typeof setupGl> | null>(null);
  const binsRef = useRef<Float32Array>(new Float32Array(BIN_COUNT));
  const rafRef = useRef<number>(0);
  const micActiveRef = useRef(false);
  const micSessionRef = useRef<MicSession | null>(null);
  const [status, setStatus] = useState<"connecting" | "open" | "closed" | "error">("connecting");
  const [streamMeta, setStreamMeta] = useState<string>("");
  const [glError, setGlError] = useState<string>("");
  const [wsError, setWsError] = useState<string>("");
  const [micOn, setMicOn] = useState(false);
  const [micError, setMicError] = useState<string>("");

  const resize = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = Math.min(window.devicePixelRatio ?? 1, 2);
    const w = Math.floor(canvas.clientWidth * dpr);
    const h = Math.floor(320 * dpr);
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    resize();
    try {
      glState.current = setupGl(canvas);
    } catch (e) {
      setStatus("error");
      setGlError(e instanceof Error ? e.message : String(e));
      return;
    }

    const draw = () => {
      const state = glState.current;
      if (!state) return;
      const { gl, prog, vao, tex, loc } = state;
      resize();
      const mic = micSessionRef.current;
      if (micActiveRef.current && mic) {
        mic.analyser.getByteFrequencyData(mic.freqBytes);
        fillBinsFromFrequencyBytes(binsRef.current, mic.freqBytes, BIN_COUNT);
      }
      const cw = canvas.width;
      const ch = canvas.height;
      gl.viewport(0, 0, cw, ch);
      gl.useProgram(prog);
      gl.bindVertexArray(vao);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, tex);
      const bytes = new Uint8Array(binsRef.current.length);
      for (let i = 0; i < binsRef.current.length; i++) {
        bytes[i] = Math.min(255, Math.max(0, Math.round(binsRef.current[i]! * 255)));
      }
      gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
      gl.texImage2D(
        gl.TEXTURE_2D,
        0,
        gl.R8,
        bytes.length,
        1,
        0,
        gl.RED,
        gl.UNSIGNED_BYTE,
        bytes,
      );
      gl.uniform1i(loc.bins, 0);
      gl.uniform2f(loc.resolution, cw, ch);
      gl.uniform1f(loc.binCount, bytes.length);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      gl.bindVertexArray(null);
      rafRef.current = requestAnimationFrame(draw);
    };
    rafRef.current = requestAnimationFrame(draw);

    const ro = new ResizeObserver(() => resize());
    ro.observe(canvas);

    return () => {
      cancelAnimationFrame(rafRef.current);
      ro.disconnect();
      const state = glState.current;
      if (state) {
        state.gl.deleteProgram(state.prog);
        state.gl.deleteVertexArray(state.vao);
        state.gl.deleteTexture(state.tex);
      }
      glState.current = null;
    };
  }, [resize]);

  useEffect(() => {
    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    const url = `${proto}//${window.location.host}/ws`;
    const ws = new WebSocket(url);

    ws.onopen = () => {
      setStatus("open");
      setWsError("");
    };
    ws.onclose = () => setStatus("closed");
    ws.onerror = () => {
      setStatus("error");
      setWsError("WebSocket failed. Start the API (uvicorn on port 8000) and refresh.");
    };

    ws.onmessage = (ev) => {
      if (micActiveRef.current) return;
      try {
        const msg = JSON.parse(String(ev.data)) as FftMessage;
        if (msg.type !== "fft" || !Array.isArray(msg.bins)) return;
        const arr = binsRef.current;
        const n = Math.min(arr.length, msg.bins.length);
        for (let i = 0; i < n; i++) arr[i] = msg.bins[i] ?? 0;
        setStreamMeta(`${msg.sampleRate} Hz · frame ${msg.frameSize} · server`);
      } catch {
        /* ignore */
      }
    };

    return () => ws.close();
  }, []);

  const stopMic = useCallback(() => {
    micActiveRef.current = false;
    setMicOn(false);
    setMicError("");
    const s = micSessionRef.current;
    if (s) {
      s.stream.getTracks().forEach((t) => t.stop());
      void s.ctx.close();
      micSessionRef.current = null;
    }
    setStreamMeta("");
  }, []);

  const startMic = useCallback(async () => {
    setMicError("");
    if (!navigator.mediaDevices?.getUserMedia) {
      setMicError("This browser does not support getUserMedia.");
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
      const Ctx = window.AudioContext ?? (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
      if (!Ctx) {
        stream.getTracks().forEach((t) => t.stop());
        setMicError("Web Audio API not available.");
        return;
      }
      const ctx = new Ctx();
      await ctx.resume();
      const analyser = ctx.createAnalyser();
      analyser.fftSize = ANALYSER_FFT_SIZE;
      analyser.smoothingTimeConstant = 0.35;
      const source = ctx.createMediaStreamSource(stream);
      source.connect(analyser);
      const buf = new ArrayBuffer(analyser.frequencyBinCount);
      const freqBytes = new Uint8Array(buf) as Uint8Array<ArrayBuffer>;
      micSessionRef.current = {
        ctx,
        stream,
        analyser,
        freqBytes,
      };
      micActiveRef.current = true;
      setMicOn(true);
      setStreamMeta(`Mic · ${Math.round(ctx.sampleRate)} Hz · FFT ${analyser.fftSize} · browser`);
    } catch (e) {
      const name = e instanceof DOMException ? e.name : "";
      if (name === "NotAllowedError" || name === "SecurityError") {
        setMicError("Microphone permission denied.");
      } else if (name === "NotFoundError") {
        setMicError("No microphone found.");
      } else {
        setMicError(e instanceof Error ? e.message : String(e));
      }
    }
  }, []);

  useEffect(() => () => stopMic(), [stopMic]);

  const statusLabel =
    status === "open" ? "Connected" : status === "connecting" ? "Connecting…" : status === "closed" ? "Disconnected" : "Error";

  return (
    <div>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: "0.75rem",
          marginBottom: "0.5rem",
          fontSize: "0.85rem",
          color: "#9aa0a6",
        }}
      >
        <span
          style={{
            width: 8,
            height: 8,
            borderRadius: "50%",
            background:
              status === "open" ? "#34a853" : status === "connecting" ? "#fbbc04" : "#ea4335",
          }}
        />
        <span>{statusLabel}</span>
        {streamMeta ? <span>· {streamMeta}</span> : null}
        <span style={{ flex: 1 }} />
        {micOn ? (
          <button
            type="button"
            onClick={stopMic}
            style={micButtonStyle}
          >
            Stop microphone
          </button>
        ) : (
          <button
            type="button"
            onClick={() => void startMic()}
            style={micButtonStyle}
          >
            Use microphone
          </button>
        )}
      </div>
      <canvas
        ref={canvasRef}
        style={{
          width: "100%",
          height: 320,
          display: "block",
          borderRadius: 8,
          border: "1px solid #2a2f3a",
          background: "#0f1115",
        }}
      />
      {micError ? (
        <p style={{ color: "#f28b82", fontSize: "0.85rem", marginTop: "0.5rem" }}>{micError}</p>
      ) : null}
      {glError || (status === "error" && wsError) ? (
        <p style={{ color: "#f28b82", fontSize: "0.85rem", marginTop: "0.5rem" }}>
          {glError || wsError}
        </p>
      ) : null}
    </div>
  );
}

const micButtonStyle: CSSProperties = {
  font: "inherit",
  fontSize: "0.85rem",
  padding: "0.35rem 0.65rem",
  borderRadius: 6,
  border: "1px solid #3c4043",
  background: "#1a1d24",
  color: "#e8eaed",
  cursor: "pointer",
};

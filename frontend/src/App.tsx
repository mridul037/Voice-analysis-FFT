import { SpectrumGL } from "./components/SpectrumGL";
import "./App.css";

export default function App() {
  return (
    <main
      style={{
        maxWidth: 960,
        margin: "0 auto",
        padding: "1.5rem",
      }}
    >
      <h1 style={{ fontSize: "1.25rem", fontWeight: 600, margin: "0 0 0.5rem" }}>
        Voice Analyser
      </h1>
      <p style={{ margin: "0 0 1rem", color: "#9aa0a6", fontSize: "0.9rem" }}>
        WebSocket FFT stream (FastAPI + NumPy) → WebGL2 spectrum
      </p>
      <SpectrumGL />
    </main>
  );
}

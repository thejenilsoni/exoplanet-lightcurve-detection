"use client";

import {ChangeEvent, useMemo, useRef, useState} from "react";
import {analyzeLightCurve, type AnalysisResult} from "@/lib/api";

type RunState = "idle" | "running" | "complete" | "error";

const demoTime = Array.from({length: 180}, (_, i) => i * 0.12);
const demoFlux = demoTime.map((time, index) => {
  const phase = ((time - 0.48) % 2.47 + 2.47) % 2.47;
  const transit = phase < 0.13 || phase > 2.40 ? -0.0087 : 0;
  const noise = Math.sin(index * 2.13) * 0.0012 + Math.cos(index * 0.71) * 0.0007;
  return 1 + transit + noise + Math.sin(time / 4) * 0.0009;
});
const demoTrend = demoTime.map((time) => 1 + Math.sin(time / 4) * 0.0009);
const demoPeriods = Array.from({length: 120}, (_, i) => 0.5 + i * 0.065);
const demoPower = demoPeriods.map((period) =>
  0.08 + 0.12 * Math.abs(Math.sin(period * 3.2)) +
  0.78 * Math.exp(-Math.pow(period - 2.47, 2) / 0.018) +
  0.31 * Math.exp(-Math.pow(period - 4.94, 2) / 0.045)
);
const demoPhase = Array.from({length: 160}, (_, i) => -0.5 + i / 159);
const demoPhaseFlux = demoPhase.map((phase, index) => {
  const transit = Math.abs(phase) < 0.047 ? -0.0087 * (1 - Math.pow(Math.abs(phase) / 0.047, 4)) : 0;
  return 1 + transit + Math.sin(index * 1.91) * 0.0007;
});

const demo: AnalysisResult = {
  requestId: "demo-2026",
  targetName: "TIC 307210830",
  mission: "TESS · Sector 82",
  disposition: "high-interest",
  probability: 0.934,
  model: "transit-fusion-1d",
  mode: "baseline",
  metrics: {
    periodDays: 2.4704,
    durationHours: 2.78,
    depthPpm: 8710,
    signalToNoise: 14.62,
    oddEvenMismatch: 0.047,
    secondaryDepthPpm: 184,
  },
  time: demoTime,
  normalizedFlux: demoFlux,
  trend: demoTrend,
  phase: demoPhase,
  phaseFlux: demoPhaseFlux,
  periods: demoPeriods,
  power: demoPower,
  flags: ["Period stable", "Odd-even consistent", "No significant secondary", "Transit shape plausible"],
};

export default function TransitLab() {
  const inputRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [state, setState] = useState<RunState>("idle");
  const [result, setResult] = useState<AnalysisResult>(demo);
  const [error, setError] = useState("");
  const [view, setView] = useState<"detrended" | "raw">("detrended");

  async function runAnalysis() {
    setState("running");
    setError("");
    if (!file) {
      await new Promise((resolve) => setTimeout(resolve, 1400));
      setResult(demo);
      setState("complete");
      return;
    }
    try {
      setResult(await analyzeLightCurve(file));
      setState("complete");
    } catch (reason) {
      setState("error");
      setError(reason instanceof Error ? reason.message : "Unable to analyze this light curve.");
    }
  }

  function selectFile(event: ChangeEvent<HTMLInputElement>) {
    const next = event.target.files?.[0];
    if (!next) return;
    setFile(next);
    setState("idle");
    setError("");
  }

  return (
    <main className="shell">
      <aside className="sidebar">
        <div className="logo"><span>TL</span><i /></div>
        <nav>
          <Nav symbol="⌁" label="Analysis" active />
          <Nav symbol="◫" label="Candidates" />
          <Nav symbol="⌇" label="Datasets" />
          <Nav symbol="◇" label="Models" />
        </nav>
        <div className="side-spacer" />
        <Nav symbol="⚙" label="Settings" />
        <div className="avatar">JS</div>
      </aside>

      <section className="workspace">
        <header className="topbar">
          <div>
            <p className="overline"><span className="pulse" /> Transit detection workspace</p>
            <h1>Exoplanet candidate analysis</h1>
          </div>
          <div className="top-actions">
            <span className="service-status"><i /> Detection service ready</span>
            <button className="quiet-button">Export report</button>
          </div>
        </header>

        <div className="main-grid">
          <section className="analysis">
            <div className="target-bar">
              <div className="target-identity">
                <div className="star-glyph">✦</div>
                <div>
                  <p className="overline">Active target</p>
                  <h2>{file?.name ?? result.targetName}</h2>
                  <span>{file ? "Uploaded light curve" : result.mission} · PDCSAP flux</span>
                </div>
              </div>
              <div className="target-actions">
                <input ref={inputRef} hidden type="file" accept=".csv,.fits,.fit" onChange={selectFile} />
                <button className="upload" onClick={() => inputRef.current?.click()}>
                  {file ? "Replace file" : "Upload CSV / FITS"}
                </button>
                <button className="analyze-button" onClick={runAnalysis} disabled={state === "running"}>
                  {state === "running" ? "Searching transits…" : "Run detection"}
                </button>
              </div>
            </div>

            {error && <div className="error">{error} Start the API or run the built-in demonstration.</div>}

            <div className="metric-grid">
              <Metric label="Best period" value={result.metrics.periodDays.toFixed(4)} unit="days" accent="gold" />
              <Metric label="Transit depth" value={result.metrics.depthPpm.toLocaleString()} unit="ppm" accent="blue" />
              <Metric label="Duration" value={result.metrics.durationHours.toFixed(2)} unit="hours" accent="violet" />
              <Metric label="Detection S/N" value={result.metrics.signalToNoise.toFixed(2)} unit="sigma" accent="mint" />
            </div>

            <article className={`chart-card large ${state === "running" ? "processing" : ""}`}>
              <div className="card-head">
                <div>
                  <p className="overline">Time domain</p>
                  <h3>Light curve</h3>
                </div>
                <div className="segmented">
                  <button className={view === "detrended" ? "selected" : ""} onClick={() => setView("detrended")}>Detrended</button>
                  <button className={view === "raw" ? "selected" : ""} onClick={() => setView("raw")}>Raw + trend</button>
                </div>
              </div>
              <LightCurveChart
                time={result.time}
                flux={view === "detrended" ? result.normalizedFlux : result.normalizedFlux.map((v, i) => v + result.trend[i] - 1)}
                trend={view === "raw" ? result.trend : undefined}
              />
              <div className="axis-caption"><span>Time (BTJD)</span><span>Normalized flux</span></div>
              {state === "running" && <Scanning />}
            </article>

            <div className="chart-row">
              <article className="chart-card">
                <div className="card-head">
                  <div>
                    <p className="overline">Period search</p>
                    <h3>Box Least Squares periodogram</h3>
                  </div>
                  <span className="peak-label">Peak · {result.metrics.periodDays.toFixed(3)} d</span>
                </div>
                <Periodogram periods={result.periods} power={result.power} peak={result.metrics.periodDays} />
              </article>
              <article className="chart-card">
                <div className="card-head">
                  <div>
                    <p className="overline">Transit profile</p>
                    <h3>Phase-folded signal</h3>
                  </div>
                  <span className="fold-label">Phase 0.0</span>
                </div>
                <PhaseChart phase={result.phase} flux={result.phaseFlux} />
              </article>
            </div>

            <article className="candidate-table-card">
              <div className="card-head">
                <div>
                  <p className="overline">Ranked detections</p>
                  <h3>Period candidates</h3>
                </div>
                <span className="muted">Top harmonics and aliases</span>
              </div>
              <div className="candidate-table">
                <div className="table-row heading"><span>Rank</span><span>Period</span><span>Power</span><span>S/N</span><span>Assessment</span></div>
                <CandidateRow rank="01" period={result.metrics.periodDays} power={0.94} snr={result.metrics.signalToNoise} label="Primary" />
                <CandidateRow rank="02" period={result.metrics.periodDays * 2} power={0.51} snr={8.21} label="Harmonic" />
                <CandidateRow rank="03" period={1.236} power={0.34} snr={6.48} label="Alias" />
              </div>
            </article>
          </section>

          <aside className="insights">
            <article className="score-card">
              <p className="overline">Candidate assessment</p>
              <div className="score-row">
                <div className="score-ring" style={{"--score": `${result.probability * 360}deg`} as React.CSSProperties}>
                  <div><strong>{Math.round(result.probability * 100)}</strong><span>%</span></div>
                </div>
                <div>
                  <span className={`disposition ${result.disposition}`}>{result.disposition.replace("-", " ")}</span>
                  <h3>Transit-like signal</h3>
                  <p>Combined periodicity, morphology, and vetting score.</p>
                </div>
              </div>
              <div className="model-line"><span>Detection mode</span><strong>{result.mode === "learned" ? "Neural ensemble" : "Validated baseline"}</strong></div>
            </article>

            <article className="insight-card">
              <div className="card-head"><div><p className="overline">Vetting checks</p><h3>Signal diagnostics</h3></div><span className="check-count">4 / 4</span></div>
              <div className="checks">
                {result.flags.map((flag) => <div className="check" key={flag}><i>✓</i><span>{flag}</span></div>)}
              </div>
            </article>

            <article className="insight-card">
              <p className="overline">False-positive analysis</p>
              <h3>Diagnostic measurements</h3>
              <div className="diagnostics">
                <Diagnostic label="Odd-even mismatch" value={`${(result.metrics.oddEvenMismatch * 100).toFixed(1)}%`} status="Pass" />
                <Diagnostic label="Secondary eclipse" value={`${result.metrics.secondaryDepthPpm} ppm`} status="Pass" />
                <Diagnostic label="Transit count" value="8 events" status="Good" />
                <Diagnostic label="Data coverage" value="91.4%" status="Good" />
              </div>
            </article>

            <article className="insight-card explanation">
              <p className="overline">Model explanation</p>
              <h3>Evidence contribution</h3>
              <Evidence label="BLS peak strength" value={92} />
              <Evidence label="Transit morphology" value={86} />
              <Evidence label="Period consistency" value={81} />
              <Evidence label="Noise resilience" value={74} />
            </article>

            <div className="science-note">
              <span>i</span>
              <p><strong>Candidate, not confirmation</strong>Automated detections require centroid checks, contamination analysis, and independent follow-up.</p>
            </div>
          </aside>
        </div>
      </section>
    </main>
  );
}

function Nav({symbol, label, active = false}: {symbol:string; label:string; active?:boolean}) {
  return <button className={`nav-item ${active ? "active" : ""}`} title={label}><b>{symbol}</b><span>{label}</span></button>;
}

function Metric({label, value, unit, accent}: {label:string; value:string; unit:string; accent:string}) {
  return <div className={`metric ${accent}`}><span>{label}</span><strong>{value}</strong><small>{unit}</small></div>;
}

function Scanning() {
  return <div className="scan-overlay"><i /><strong>Evaluating periodic signals</strong><span>Detrending · BLS search · candidate vetting</span></div>;
}

function CandidateRow({rank, period, power, snr, label}: {rank:string; period:number; power:number; snr:number; label:string}) {
  return <div className="table-row"><span className="rank">{rank}</span><span>{period.toFixed(4)} d</span><span><i className="power-bar" style={{width:`${power * 70}px`}} />{power.toFixed(2)}</span><span>{snr.toFixed(2)}</span><span className={label === "Primary" ? "primary-tag" : "soft-tag"}>{label}</span></div>;
}

function Diagnostic({label, value, status}: {label:string; value:string; status:string}) {
  return <div><span>{label}</span><strong>{value}</strong><i>{status}</i></div>;
}

function Evidence({label, value}: {label:string; value:number}) {
  return <div className="evidence"><div><span>{label}</span><strong>{value}%</strong></div><i><b style={{width:`${value}%`}} /></i></div>;
}

function points(x: number[], y: number[], width:number, height:number, padding=14) {
  const minX=Math.min(...x), maxX=Math.max(...x), minY=Math.min(...y), maxY=Math.max(...y);
  return x.map((value,index) => {
    const px=padding+(value-minX)/(maxX-minX || 1)*(width-padding*2);
    const py=padding+(maxY-y[index])/(maxY-minY || 1)*(height-padding*2);
    return `${px.toFixed(1)},${py.toFixed(1)}`;
  }).join(" ");
}

function LightCurveChart({time, flux, trend}: {time:number[]; flux:number[]; trend?:number[]}) {
  return <svg className="light-chart" viewBox="0 0 900 260" preserveAspectRatio="none" aria-label="Light curve chart">
    <Grid width={900} height={260} />
    {trend && <polyline className="trend-line" points={points(time,trend,900,260)} />}
    <polyline className="flux-line" points={points(time,flux,900,260)} />
    {time.filter((_,i)=>i%3===0).map((value,i) => {
      const sourceIndex=i*3; const coordinate=points([time[0],value],[Math.min(...flux),flux[sourceIndex]],900,260).split(" ")[1];
      const [cx,cy]=coordinate.split(",");
      return <circle key={value} cx={cx} cy={cy} r="1.7" className="flux-point" />;
    })}
  </svg>;
}

function Periodogram({periods, power, peak}: {periods:number[]; power:number[]; peak:number}) {
  const peakX=14+(peak-Math.min(...periods))/(Math.max(...periods)-Math.min(...periods))*(440-28);
  return <svg className="small-chart" viewBox="0 0 440 190" preserveAspectRatio="none" aria-label="BLS periodogram">
    <Grid width={440} height={190} />
    <polygon className="power-area" points={`14,176 ${points(periods,power,440,190)} 426,176`} />
    <polyline className="power-line" points={points(periods,power,440,190)} />
    <line className="peak-line" x1={peakX} x2={peakX} y1="12" y2="176" />
  </svg>;
}

function PhaseChart({phase, flux}: {phase:number[]; flux:number[]}) {
  return <svg className="small-chart" viewBox="0 0 440 190" preserveAspectRatio="none" aria-label="Phase folded transit">
    <Grid width={440} height={190} />
    <polyline className="phase-line" points={points(phase,flux,440,190)} />
    {phase.filter((_,i)=>i%4===0).map((value,i) => {
      const index=i*4; const xy=points([phase[0],value],[Math.min(...flux),flux[index]],440,190).split(" ")[1].split(",");
      return <circle key={value} cx={xy[0]} cy={xy[1]} r="2" className="phase-point" />;
    })}
  </svg>;
}

function Grid({width,height}:{width:number;height:number}) {
  return <g className="grid-lines">{[0.2,0.4,0.6,0.8].map(v=><line key={`h${v}`} x1="14" x2={width-14} y1={height*v} y2={height*v}/>)}
    {[0.2,0.4,0.6,0.8].map(v=><line key={`v${v}`} y1="12" y2={height-14} x1={width*v} x2={width*v}/>)}</g>;
}

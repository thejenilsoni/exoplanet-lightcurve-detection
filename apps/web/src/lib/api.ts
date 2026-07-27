export type CandidateMetrics = {
  periodDays: number;
  durationHours: number;
  depthPpm: number;
  signalToNoise: number;
  oddEvenMismatch: number;
  secondaryDepthPpm: number;
};

export type AnalysisResult = {
  requestId: string;
  targetName: string;
  mission: string;
  disposition: "high-interest" | "review" | "low-interest";
  probability: number;
  model: string;
  mode: "learned" | "baseline";
  metrics: CandidateMetrics;
  time: number[];
  normalizedFlux: number[];
  trend: number[];
  phase: number[];
  phaseFlux: number[];
  periods: number[];
  power: number[];
  flags: string[];
};

const API_URL = process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") ?? "http://localhost:8000";

export async function analyzeLightCurve(file: File): Promise<AnalysisResult> {
  const form = new FormData();
  form.append("light_curve", file);
  const response = await fetch(`${API_URL}/v1/analyze`, {method: "POST", body: form});
  if (!response.ok) throw new Error((await response.text()) || "Analysis failed");
  return (await response.json()) as AnalysisResult;
}

import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Transit Lab — Exoplanet Candidate Detection",
  description: "Detect and inspect transit candidates in noisy astronomical light curves.",
};

export default function RootLayout({children}: Readonly<{children: React.ReactNode}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

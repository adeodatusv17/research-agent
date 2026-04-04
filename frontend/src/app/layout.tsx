import type { Metadata } from "next";
import { Inter, Space_Grotesk } from "next/font/google";
import "katex/dist/katex.min.css";
import "../styles/globals.css";
import { Toaster } from "react-hot-toast";

const inter = Inter({ subsets: ["latin"], variable: "--font-sans" });
const spaceGrotesk = Space_Grotesk({ subsets: ["latin"], variable: "--font-label" });

export const metadata: Metadata = {
  title: "Digital Archivist | Research Replication",
  description: "Upload research papers, inspect structured analysis, review reproducibility, and generate experiment scaffolds.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`dark ${inter.variable} ${spaceGrotesk.variable}`}>
      <body className="min-h-screen bg-bg-base text-text-primary font-sans antialiased">
        {children}
        <Toaster
          position="bottom-right"
          toastOptions={{
            style: {
              background: "#131B2B",
              color: "#F0F4FF",
              border: "1px solid #1E2D45",
              borderRadius: "10px",
              fontSize: "14px",
            },
            success: { iconTheme: { primary: "#10B981", secondary: "#131B2B" } },
            error: { iconTheme: { primary: "#EF4444", secondary: "#131B2B" } },
          }}
        />
      </body>
    </html>
  );
}

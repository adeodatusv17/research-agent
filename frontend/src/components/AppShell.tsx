"use client";

import Sidebar from "@/components/Sidebar";

interface AppShellProps {
  children: React.ReactNode;
}

export default function AppShell({ children }: AppShellProps) {
  return (
    <div className="flex min-h-screen">
      <Sidebar />
      <main className="flex-1 min-h-screen min-w-0">
        {children}
      </main>
    </div>
  );
}

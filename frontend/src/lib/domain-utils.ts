import type { PaperDomain } from "@/lib/types";

export const DOMAIN_ORDER: PaperDomain[] = [
  "ml",
  "theory",
  "systems",
  "security",
  "networks",
  "general",
];

export function normalizeDomain(domain?: string | null): PaperDomain {
  const safeDomain = (domain ?? "general").toLowerCase();
  return DOMAIN_ORDER.includes(safeDomain as PaperDomain)
    ? (safeDomain as PaperDomain)
    : "general";
}

export function formatDomainLabel(domain?: string | null): string {
  const safeDomain = normalizeDomain(domain);
  if (safeDomain === "ml") return "Machine Learning";
  return safeDomain.charAt(0).toUpperCase() + safeDomain.slice(1);
}

export function formatDomainShortLabel(domain?: string | null): string {
  const safeDomain = normalizeDomain(domain);
  if (safeDomain === "ml") return "ML";
  return safeDomain.charAt(0).toUpperCase() + safeDomain.slice(1);
}

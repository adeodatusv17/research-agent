"use client";

interface DomainBlockProps {
  title: string;
  items: Array<{ label: string; value: string | string[] | null | undefined }>;
}

export default function DomainBlock({ title, items }: DomainBlockProps) {
  const visibleItems = items.filter(({ value }) => {
    if (Array.isArray(value)) return value.length > 0;
    return Boolean(value);
  });

  if (visibleItems.length === 0) {
    return null;
  }

  return (
    <section className="mt-8">
      <h2 className="text-base font-semibold text-white">{title}</h2>
      <div className="mt-4 grid grid-cols-1 gap-3 md:grid-cols-2">
        {visibleItems.map(({ label, value }) => (
          <article
            key={label}
            className="rounded-xl border border-white/10 bg-bg-surface p-4 shadow-lg shadow-black/20 transition-transform duration-150 hover:-translate-y-[2px] hover:border-white/20"
          >
            <p className="text-sm text-gray-500">{label}</p>
            {Array.isArray(value) ? (
              <div className="mt-2 flex flex-wrap gap-2">
                {value.map((item) => (
                  <span
                    key={item}
                    className="rounded-full border border-white/10 bg-bg-hover px-2.5 py-1 text-sm text-gray-300"
                  >
                    {item}
                  </span>
                ))}
              </div>
            ) : (
              <p className="mt-2 text-sm leading-relaxed text-gray-300">{value}</p>
            )}
          </article>
        ))}
      </div>
    </section>
  );
}

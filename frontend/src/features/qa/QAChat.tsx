"use client";

import { useState, useRef, useEffect, useMemo } from "react";
import { Send, User, Bot, BookOpen, ChevronDown, ChevronUp, Loader2, Copy, Check, Sigma } from "lucide-react";
import katex from "katex";
import ReactMarkdown from "react-markdown";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { vscDarkPlus } from "react-syntax-highlighter/dist/cjs/styles/prism";
import clsx from "clsx";
import { askQuestion } from "@/lib/api-client";
import type { ChatMessage, ConversationTurn, EquationCollection, PaperDomain, QASource, TableArtifact } from "@/lib/types";
import toast from "react-hot-toast";

interface QAChatProps {
  paperId: string;
  domain?: PaperDomain | null;
}

function SourceCard({ source }: { source: QASource }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="rounded-xl border border-white/10 bg-bg-hover p-2.5 text-xs shadow-lg shadow-black/20">
      <button
        onClick={() => setExpanded((prev) => !prev)}
        className="flex w-full cursor-pointer items-center justify-between"
      >
        <div className="flex items-center gap-2 text-gray-500">
          <BookOpen className="h-3 w-3" />
          <span className="font-medium text-gray-300">
            {source.section_name ?? "Unknown section"}
            {source.subsection_name ? ` > ${source.subsection_name}` : ""}
            {source.page_number ? ` • p.${source.page_number}` : ""}
          </span>
          <span className="font-mono text-accent-cyan">{Math.round(source.score * 100)}%</span>
        </div>
        {expanded ? <ChevronUp className="h-3 w-3 text-gray-500" /> : <ChevronDown className="h-3 w-3 text-gray-500" />}
      </button>
      {expanded && (
        <p className="mt-2 border-t border-white/10 pt-2 pl-5 leading-relaxed text-gray-500">
          {source.content_snippet}
        </p>
      )}
    </div>
  );
}

function CodeBlock({ inline, className, children, ...props }: any) {
  const match = /language-(\w+)/.exec(className || "");
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(String(children).replace(/\n$/, ""));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  if (!inline && match) {
    return (
      <div className="relative my-4 overflow-hidden rounded-xl border border-white/10">
        <div className="flex items-center justify-between border-b border-white/10 bg-bg-hover px-4 py-2">
          <span className="font-mono text-xs uppercase tracking-wider text-gray-500">{match[1]}</span>
          <button
            onClick={handleCopy}
            className="flex items-center gap-1.5 text-xs text-gray-500 transition-colors cursor-pointer hover:text-text-primary"
          >
            {copied ? <Check className="h-3.5 w-3.5 text-status-success" /> : <Copy className="h-3.5 w-3.5" />}
            {copied ? "Copied" : "Copy code"}
          </button>
        </div>
        <SyntaxHighlighter
          style={vscDarkPlus as any}
          language={match[1]}
          PreTag="div"
          customStyle={{ margin: 0, padding: "1rem", background: "#0A0A0A", fontSize: "0.8125rem" }}
          {...props}
        >
          {String(children).replace(/\n$/, "")}
        </SyntaxHighlighter>
      </div>
    );
  }

  return (
    <code className="rounded border border-white/10 bg-bg-hover px-1.5 py-0.5 font-mono text-[11px] text-text-primary" {...props}>
      {children}
    </code>
  );
}

function splitEquationSteps(latex: string): string[] {
  const cleaned = latex.replace(/\s+/g, " ").trim();
  if (!cleaned) {
    return [];
  }
  const pieces = cleaned
    .split(/(?=(?:[A-Za-zxy~Ëœ][A-Za-z0-9Ëœâ€²'`\s]{0,14})\s*=)/g)
    .map((part) => part.replace(/^\(\d+\)\s*/, "").trim())
    .filter(Boolean);
  if (pieces.length <= 1) {
    return [cleaned];
  }
  return pieces.filter((part) => /=/.test(part));
}

function renderEquationLatex(latex: string): string {
  try {
    return katex.renderToString(latex, {
      throwOnError: false,
      displayMode: true,
    });
  } catch {
    return katex.renderToString(`\\text{${latex.replace(/[{}]/g, "")}}`, {
      throwOnError: false,
      displayMode: true,
    });
  }
}

function EquationBlock({
  latex,
  description,
}: {
  latex: string;
  description: string;
}) {
  const equationSteps = useMemo(() => splitEquationSteps(latex), [latex]);
  const renderedEquations = useMemo(
    () => equationSteps.map((step) => renderEquationLatex(step)),
    [equationSteps]
  );

  return (
    <div className="rounded-xl border border-white/10 bg-black/20 p-4">
      <div className="rounded-lg border border-white/10 bg-black/30 p-4">
        <div className="space-y-3">
          {renderedEquations.map((equation, index) => (
            <div key={`${equationSteps[index]}-${index}`} className="overflow-x-auto text-white">
              {renderedEquations.length > 1 && (
                <p className="mb-2 text-[11px] font-medium uppercase tracking-[0.14em] text-gray-500">
                  Step {index + 1}
                </p>
              )}
              <div dangerouslySetInnerHTML={{ __html: equation }} />
            </div>
          ))}
        </div>
      </div>
      {description && (
        <div className="mt-3 rounded-lg border border-white/5 bg-white/[0.02] px-3 py-2">
          <p className="text-sm leading-relaxed text-gray-400">{description}</p>
        </div>
      )}
    </div>
  );
}

function EquationPanel({ equations }: { equations: EquationCollection | null | undefined }) {
  const equationItems = equations?.items?.filter((item) => (item.latex ?? "").trim().length > 0) ?? [];
  if (equationItems.length === 0) {
    return null;
  }
  return (
    <div className="mb-4 rounded-xl border border-white/10 bg-bg-hover/40 p-4">
      <div className="mb-3 flex items-center gap-2 text-sm text-gray-300">
        <Sigma className="h-4 w-4" />
        <span>Equations from the paper</span>
      </div>
      {equations?.source === "llm_generated" && (
        <div className="mb-4 rounded-lg border border-amber-500/20 bg-amber-500/10 p-3 text-sm text-amber-200">
          These equations were inferred by AI because no direct equations were extracted. Verify against the original paper.
        </div>
      )}
      <div className="space-y-4">
        {equationItems.map((item, index) => (
          <EquationBlock
            key={item.id ?? `qa-equation-${index}`}
            latex={(item.latex ?? "").trim()}
            description={(item.description ?? "").trim()}
          />
        ))}
      </div>
    </div>
  );
}

function parseTableRows(normalizedTableText: string): string[][] {
  return normalizedTableText
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => line.split("|").map((cell) => cell.trim()));
}

function TableCard({ table }: { table: TableArtifact }) {
  const rows = parseTableRows(table.normalized_table_text ?? "").slice(0, 8);
  const header = table.table_label ?? "Table";
  const meta = [
    table.table_type ? table.table_type.replace(/_/g, " ") : null,
    table.page_number ? `p.${table.page_number}` : null,
  ].filter(Boolean);

  return (
    <div className="rounded-xl border border-white/10 bg-black/20 p-4">
      <div className="mb-3 flex items-start justify-between gap-3">
        <div>
          <p className="text-sm font-medium text-white">{header}</p>
          {table.caption && <p className="mt-1 text-sm leading-relaxed text-gray-400">{table.caption}</p>}
        </div>
        {meta.length > 0 && (
          <div className="rounded-full border border-white/10 px-2 py-1 text-[10px] uppercase tracking-[0.12em] text-gray-500">
            {meta.join(" • ")}
          </div>
        )}
      </div>
      {rows.length > 0 ? (
        <div className="overflow-x-auto rounded-lg border border-white/10">
          <table className="min-w-full border-collapse text-left text-xs">
            <tbody>
              {rows.map((row, rowIndex) => (
                <tr key={`${header}-${rowIndex}`} className={clsx(rowIndex === 0 && "bg-white/[0.04]")}>
                  {row.map((cell, cellIndex) => (
                    <td key={`${header}-${rowIndex}-${cellIndex}`} className="border-t border-white/10 px-3 py-2 text-gray-300 first:border-l-0">
                      {cell || "—"}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="rounded-lg border border-white/10 bg-black/30 p-3 text-sm text-gray-400">
          {table.normalized_table_text ?? "No table rows available."}
        </div>
      )}
    </div>
  );
}

function TablePanel({ tables }: { tables: TableArtifact[] | null | undefined }) {
  const tableItems = tables?.filter((table) => (table.normalized_table_text ?? "").trim().length > 0) ?? [];
  if (tableItems.length === 0) {
    return null;
  }
  return (
    <div className="mb-4 rounded-xl border border-white/10 bg-bg-hover/40 p-4">
      <div className="mb-3 flex items-center gap-2 text-sm text-gray-300">
        <BookOpen className="h-4 w-4" />
        <span>Tables from the paper</span>
      </div>
      <div className="space-y-4">
        {tableItems.map((table, index) => (
          <TableCard key={table.table_id ?? `qa-table-${index}`} table={table} />
        ))}
      </div>
    </div>
  );
}

export default function QAChat({ paperId, domain }: QAChatProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const stored = sessionStorage.getItem(`qa-${paperId}`);
    if (stored) {
      try {
        setMessages(JSON.parse(stored));
      } catch {
        // ignore malformed cache
      }
    }
  }, [paperId]);

  useEffect(() => {
    sessionStorage.setItem(`qa-${paperId}`, JSON.stringify(messages));
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, paperId]);

  function applySuggestion(prompt: string) {
    setInput?.(prompt);
  }

  function buildRecentTurns(history: ChatMessage[]): ConversationTurn[] {
    return history
      .slice(-4)
      .map((message) => ({
        role: message.role,
        content: message.content,
      }))
      .filter((turn) => turn.content.trim().length > 0);
  }

  async function handleSend() {
    const query = input.trim();
    if (!query || loading) return;
    const requestId = crypto.randomUUID();
    const recentTurns = buildRecentTurns(messages);

    const userMsg: ChatMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content: query,
      timestamp: Date.now(),
    };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setLoading(true);
    console.info("[qa] request_started", { paperId, requestId, query });

    try {
      const data = await askQuestion(paperId, query, { requestId }, recentTurns);
      console.info("[qa] request_completed", {
        paperId,
        requestId,
        responseRequestId: data.request_id,
        sourceCount: data.sources?.length ?? 0,
      });
      const assistantMsg: ChatMessage = {
        id: crypto.randomUUID(),
        role: "assistant",
        content: data.answer,
        sources: data.sources,
        equations: data.equations,
        tables: data.tables,
        timestamp: Date.now(),
      };
      setMessages((prev) => [...prev, assistantMsg]);
    } catch (err: unknown) {
      console.error("[qa] request_failed", {
        paperId,
        requestId,
        error: err instanceof Error ? err.message : String(err),
        rawError: err,
      });
      toast.error(err instanceof Error ? err.message : "Failed to get answer");
      setMessages((prev) => prev.slice(0, -1));
      setInput(query);
    } finally {
      setLoading(false);
    }
  }

  const suggestions = [
    "What is the main contribution?",
    "How is the approach evaluated?",
    "What are the limitations?",
    ...(domain === "ml" ? ["What dataset and optimizer are used?"] : []),
  ];

  return (
    <div className="flex h-full flex-col">
      <div className="flex-1 overflow-y-auto px-6 py-6 space-y-6">
        {messages.length === 0 && (
          <div className="flex min-h-[60vh] flex-col items-center justify-center text-center">
            <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-full border border-white/10 bg-bg-hover">
              <Bot className="h-5 w-5 text-gray-500" />
            </div>
            <h2 className="text-xl font-semibold text-white">Ask anything about this paper</h2>
            <div className="mt-5 flex flex-wrap justify-center gap-2">
              {suggestions.map((prompt) => (
                <button
                  key={prompt}
                  onClick={() => applySuggestion(prompt)}
                  className="rounded-full border border-white/10 bg-bg-card px-3 py-1.5 text-xs text-gray-300 transition-colors cursor-pointer hover:border-white/20 hover:bg-bg-hover"
                >
                  {prompt}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((msg) => (
          <div key={msg.id} className={clsx("flex gap-3 animate-slide-up", msg.role === "user" ? "justify-end" : "justify-start")}>
            {msg.role === "assistant" && (
              <div className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-full border border-white/10 bg-bg-hover">
                <Bot className="h-3.5 w-3.5 text-gray-500" />
              </div>
            )}
            <div className={clsx("max-w-[75%] space-y-2", msg.role === "user" ? "items-end" : "items-start")}>
              <div
                className={clsx(
                  "rounded-2xl px-4 py-3 text-sm leading-relaxed",
                  msg.role === "user"
                    ? "rounded-tr-sm border border-white/10 bg-bg-hover text-text-primary"
                    : "w-full overflow-x-auto rounded-tl-sm border border-white/10 bg-bg-card text-text-primary"
                )}
              >
                {msg.role === "assistant" ? (
                  <div className="prose prose-invert max-w-none prose-p:leading-relaxed prose-pre:p-0 prose-pre:bg-transparent prose-headings:text-text-primary prose-a:text-text-primary prose-strong:text-text-primary">
                    <EquationPanel equations={msg.equations} />
                    <TablePanel tables={msg.tables} />
                    <ReactMarkdown components={{ code: CodeBlock as any }}>{msg.content}</ReactMarkdown>
                  </div>
                ) : (
                  msg.content
                )}
              </div>
              {msg.sources && msg.sources.length > 0 && (
                <div className="space-y-1.5">
                  <p className="px-1 text-[10px] text-gray-500">Sources ({msg.sources.length})</p>
                  {msg.sources.slice(0, 3).map((src, i) => (
                    <SourceCard key={i} source={src} />
                  ))}
                </div>
              )}
            </div>
            {msg.role === "user" && (
              <div className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-full border border-white/10 bg-bg-hover">
                <User className="h-3.5 w-3.5 text-gray-500" />
              </div>
            )}
          </div>
        ))}

        {loading && (
          <div className="flex gap-3 animate-fade-in">
            <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full border border-white/10 bg-bg-hover">
              <Bot className="h-3.5 w-3.5 text-gray-500" />
            </div>
            <div className="flex items-center gap-2 rounded-2xl rounded-tl-sm border border-white/10 bg-bg-card px-4 py-3">
              <Loader2 className="h-3.5 w-3.5 animate-spin text-gray-500" />
              <span className="text-sm text-gray-500">Thinking...</span>
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      <div className="border-t border-bg-border bg-bg-surface/50 px-6 py-4 backdrop-blur-sm">
        <div className="flex items-end gap-2">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                handleSend();
              }
            }}
            placeholder="Ask a question about this paper... (Enter to send)"
            rows={1}
            className="input-field min-h-[40px] max-h-[120px] flex-1 resize-none overflow-y-auto leading-relaxed"
          />
          <button
            onClick={handleSend}
            disabled={!input.trim() || loading}
            className={clsx("btn-primary shrink-0 px-3 py-2.5", (!input.trim() || loading) && "cursor-not-allowed opacity-40")}
          >
            <Send className="h-4 w-4" />
          </button>
        </div>
        <p className="mt-1.5 text-[10px] text-gray-500">Shift+Enter for new line</p>
      </div>
    </div>
  );
}

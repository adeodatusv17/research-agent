"use client";

import { useState, useCallback, useRef } from "react";
import { X, Upload, FileText, Loader2, CheckCircle, AlertCircle } from "lucide-react";
import clsx from "clsx";
import { uploadPaper } from "@/lib/api-client";
import toast from "react-hot-toast";

interface UploadModalProps {
  open: boolean;
  onClose: () => void;
  onUploaded: (paperId: string) => void;
}

type UploadStatus = "idle" | "uploading" | "success" | "error";

export default function UploadModal({ open, onClose, onUploaded }: UploadModalProps) {
  const [status, setStatus] = useState<UploadStatus>("idle");
  const [dragOver, setDragOver] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [errorMsg, setErrorMsg] = useState("");
  const fileRef = useRef<HTMLInputElement>(null);

  function reset() {
    setStatus("idle");
    setSelectedFile(null);
    setErrorMsg("");
    setDragOver(false);
  }

  function handleClose() {
    if (status === "uploading") return;
    reset();
    onClose();
  }

  function selectFile(file: File) {
    if (file.type !== "application/pdf") {
      setErrorMsg("Only PDF files are supported.");
      return;
    }
    setSelectedFile(file);
    setErrorMsg("");
  }

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) selectFile(file);
  }, []);

  async function handleUpload() {
    if (!selectedFile) return;
    setStatus("uploading");
    try {
      const result = await uploadPaper(selectedFile);
      setStatus("success");
      toast.success("Paper uploaded successfully!");
      setTimeout(() => {
        reset();
        onUploaded(result.paper_id);
      }, 800);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Upload failed";
      setStatus("error");
      setErrorMsg(msg);
      toast.error(msg);
    }
  }

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/60 backdrop-blur-sm"
        onClick={handleClose}
      />

      {/* Modal */}
      <div className="relative w-full max-w-md card p-6 shadow-2xl animate-slide-up border-gradient">
        {/* Header */}
        <div className="flex items-center justify-between mb-5">
          <div>
            <h2 className="text-base font-semibold text-text-primary">Upload Research Paper</h2>
            <p className="text-xs text-text-muted mt-0.5">PDF only · Auto-parsed and embedded</p>
          </div>
          <button onClick={handleClose} className="btn-ghost p-1.5" disabled={status === "uploading"}>
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Drop Zone */}
        {status !== "success" && (
          <div
            onDrop={onDrop}
            onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onClick={() => fileRef.current?.click()}
            className={clsx(
              "relative border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all duration-200",
              dragOver
                ? "border-accent-indigo bg-accent-indigo/10 shadow-glow-indigo"
                : selectedFile
                ? "border-status-success/40 bg-status-success/5"
                : "border-bg-border hover:border-accent-indigo/40 hover:bg-bg-hover"
            )}
          >
            <input
              ref={fileRef}
              type="file"
              accept="application/pdf"
              className="hidden"
              onChange={(e) => e.target.files?.[0] && selectFile(e.target.files[0])}
            />

            {selectedFile ? (
              <div className="space-y-2">
                <FileText className="w-8 h-8 text-status-success mx-auto" />
                <div className="text-sm font-medium text-text-primary truncate max-w-[280px] mx-auto">
                  {selectedFile.name}
                </div>
                <div className="text-xs text-text-muted">
                  {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                </div>
              </div>
            ) : (
              <div className="space-y-2">
                <Upload className="w-8 h-8 text-text-muted mx-auto" />
                <div className="text-sm text-text-secondary">
                  Drop a PDF here or <span className="text-accent-indigo">browse</span>
                </div>
                <div className="text-xs text-text-muted">PDF up to 50MB</div>
              </div>
            )}
          </div>
        )}

        {/* Success state */}
        {status === "success" && (
          <div className="py-8 text-center animate-fade-in">
            <CheckCircle className="w-12 h-12 text-status-success mx-auto mb-3" />
            <p className="text-sm font-medium text-text-primary">Upload complete!</p>
            <p className="text-xs text-text-muted mt-1">Redirecting to paper…</p>
          </div>
        )}

        {/* Error */}
        {errorMsg && (
          <div className="mt-3 flex items-center gap-2 text-xs text-status-error bg-status-error/10 border border-status-error/20 rounded-lg px-3 py-2">
            <AlertCircle className="w-3.5 h-3.5 flex-shrink-0" />
            {errorMsg}
          </div>
        )}

        {/* Actions */}
        {status !== "success" && (
          <div className="mt-5 flex gap-2">
            <button onClick={handleClose} className="btn-secondary flex-1" disabled={status === "uploading"}>
              Cancel
            </button>
            <button
              onClick={handleUpload}
              disabled={!selectedFile || status === "uploading"}
              className={clsx(
                "btn-primary flex-1 justify-center",
                (!selectedFile || status === "uploading") && "opacity-50 cursor-not-allowed"
              )}
            >
              {status === "uploading" ? (
                <>
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                  Uploading…
                </>
              ) : (
                <>
                  <Upload className="w-3.5 h-3.5" />
                  Upload
                </>
              )}
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

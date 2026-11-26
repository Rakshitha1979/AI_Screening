import type { ReactNode } from "react";
import { AlertTriangle, Info } from "lucide-react";

export function InfoCallout({ children }: { children: ReactNode }) {
  return (
    <div className="flex items-start gap-2.5 bg-[#EFF5FD] border-l-4 border-[#6392D5] rounded-lg px-4 py-3 text-[0.9rem] text-[#3B5D91] mb-4">
      <Info className="w-4 h-4 mt-0.5 shrink-0 text-[#2563EB]" />
      <div>{children}</div>
    </div>
  );
}

export function AlertCallout({ children }: { children: ReactNode }) {
  return (
    <div className="flex items-start gap-2.5 bg-[#FFF7ED] border-l-4 border-[#E9AD58] rounded-lg px-4 py-3 text-[0.9rem] text-[#98652B] mb-4">
      <AlertTriangle className="w-4 h-4 mt-0.5 shrink-0 text-[#D97706]" />
      <div>{children}</div>
    </div>
  );
}

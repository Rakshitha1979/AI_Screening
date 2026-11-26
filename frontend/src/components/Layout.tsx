import { useState, type ReactNode } from "react";
import { NavLink } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import {
  BarChart3,
  Briefcase,
  FileUp,
  GitCompare,
  LayoutDashboard,
  Menu,
  Search,
  Sparkles,
  X,
} from "lucide-react";
import { api } from "../api/client";

const NAV_ITEMS = [
  { to: "/", label: "Project Overview", icon: LayoutDashboard },
  { to: "/screening", label: "AI Resume Screening", icon: Search },
  { to: "/upload", label: "Upload Resume", icon: FileUp },
  { to: "/analytics", label: "Recruitment Analytics", icon: BarChart3 },
  { to: "/models", label: "Model Comparison", icon: GitCompare },
];

export function Layout({ children }: { children: ReactNode }) {
  const [mobileOpen, setMobileOpen] = useState(false);

  const { data: overview } = useQuery({
    queryKey: ["overview"],
    queryFn: api.overview,
    staleTime: 5 * 60 * 1000,
  });

  const sidebarContent = (
    <div className="flex flex-col h-full px-4 py-5 overflow-y-auto">
      {/* Brand Tile */}
      <div className="brand-tile-gradient border border-[#DCE6F5] rounded-[15px] px-3 py-4 text-center mb-4">
        <div className="relative w-20 h-14 mx-auto mb-2.5 flex items-center justify-center rounded-2xl bg-gradient-to-br from-[#D6E5FC] to-[#E9E0FC]">
          <Briefcase className="w-7 h-7 text-[#29466F]" />
          <Sparkles className="w-4 h-4 text-[#8A75C8] absolute top-1.5 right-2" />
        </div>
        <div className="text-[#29466F] text-[1.15rem] font-extrabold tracking-tight">
          Recruitment Suite
        </div>
        <div className="text-[#7788A2] text-[0.76rem] mt-1 leading-snug">
          AI Screening &amp; Pipeline Analytics
        </div>
      </div>

      {/* Workspace Label */}
      <div className="text-[#8795AA] text-[0.7rem] font-bold tracking-[1.1px] uppercase ml-2 mb-2">
        WORKSPACE
      </div>

      {/* Navigation Links */}
      <nav className="flex flex-col gap-1.5" aria-label="Main navigation">
        {NAV_ITEMS.map(({ to, label, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            end={to === "/"}
            onClick={() => setMobileOpen(false)}
            className={({ isActive }) =>
              `flex items-center gap-2.5 px-3 py-2.5 rounded-[10px] text-[0.875rem] border transition-all duration-150 ${
                isActive
                  ? "bg-[#E2ECFC] border-[#C9DAF5] text-[#315FA8] font-bold shadow-[0_2px_7px_rgba(65,105,170,0.08)]"
                  : "bg-white/65 border-[#E6ECF5] text-[#334155] font-medium hover:bg-[#E8F0FC] hover:border-[#D2E0F5]"
              }`
            }
          >
            <Icon className="w-4 h-4 shrink-0" />
            <span>{label}</span>
          </NavLink>
        ))}
      </nav>

      {/* Data Status Panel */}
      <div className="mt-6 p-3.5 bg-white border border-[#E1E8F2] rounded-xl shadow-[0_3px_10px_rgba(45,70,110,0.045)]">
        <div className="flex items-center text-[#405D85] text-[0.82rem] font-bold pb-2.5 mb-1 border-b border-[#EDF0F6]">
          <span className="inline-block w-2 h-2 bg-[#36B58A] rounded-full mr-2" />
          Data Status
        </div>
        <div className="divide-y divide-[#F0F2F7] text-[0.75rem] text-[#78869A]">
          <div className="flex justify-between py-1.5">
            <span>Resumes</span>
            <strong className="text-[#3E5D87] font-bold">
              {overview ? overview.counts.resumes.toLocaleString() : "—"}
            </strong>
          </div>
          <div className="flex justify-between py-1.5">
            <span>Jobs</span>
            <strong className="text-[#3E5D87] font-bold">
              {overview ? overview.counts.jobs.toLocaleString() : "—"}
            </strong>
          </div>
          <div className="flex justify-between py-1.5">
            <span>Applications</span>
            <strong className="text-[#3E5D87] font-bold">
              {overview ? overview.counts.applications.toLocaleString() : "—"}
            </strong>
          </div>
          <div className="flex justify-between py-1.5">
            <span>Offers</span>
            <strong className="text-[#3E5D87] font-bold">
              {overview ? overview.counts.offers.toLocaleString() : "—"}
            </strong>
          </div>
          <div className="flex justify-between py-1.5">
            <span>Anomalies</span>
            <strong className="text-[#3E5D87] font-bold">
              {overview ? overview.counts.anomalies.toLocaleString() : "—"}
            </strong>
          </div>
        </div>
      </div>

      {/* Footer */}
      <div className="mt-auto pt-5 text-center text-[#929FB1] text-[0.68rem] leading-relaxed">
        Final College Demonstration Project
      </div>
    </div>
  );

  return (
    <div className="min-h-screen flex bg-[#F5F7FC] text-[#263449]">
      {/* Desktop Sidebar */}
      <aside className="hidden lg:block w-[270px] shrink-0 sidebar-gradient border-r border-[#E1E8F2] sticky top-0 h-screen">
        {sidebarContent}
      </aside>

      {/* Mobile Drawer */}
      {mobileOpen && (
        <div className="fixed inset-0 z-50 lg:hidden flex">
          <div
            className="fixed inset-0 bg-slate-900/30"
            onClick={() => setMobileOpen(false)}
            aria-hidden="true"
          />
          <aside className="relative w-[270px] max-w-[85vw] sidebar-gradient border-r border-[#E1E8F2] h-full z-10">
            <button
              type="button"
              onClick={() => setMobileOpen(false)}
              aria-label="Close navigation"
              className="absolute top-3 right-3 p-1.5 rounded-lg bg-white border border-[#E1E8F2] text-slate-600"
            >
              <X className="w-4 h-4" />
            </button>
            {sidebarContent}
          </aside>
        </div>
      )}

      {/* Main Content */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Mobile Top Bar */}
        <header className="lg:hidden flex items-center justify-between px-4 py-3 bg-white border-b border-[#E1E8F2] sticky top-0 z-30">
          <button
            type="button"
            onClick={() => setMobileOpen(true)}
            aria-label="Open navigation"
            className="p-2 rounded-lg border border-[#E1E8F2] text-[#243B60]"
          >
            <Menu className="w-5 h-5" />
          </button>
          <span className="font-bold text-[#243B60] text-sm">Recruitment Suite</span>
          <div className="w-9" />
        </header>

        <main className="flex-1 w-full max-w-[1450px] mx-auto px-4 sm:px-6 lg:px-8 py-6 lg:py-8">
          {children}
        </main>
      </div>
    </div>
  );
}

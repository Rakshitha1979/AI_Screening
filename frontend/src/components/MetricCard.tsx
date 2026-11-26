import type { ReactNode } from "react";

interface MetricCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  valueClassName?: string;
  leftAccentColor?: string;
  children?: ReactNode;
  className?: string;
}

export function MetricCard({
  title,
  value,
  subtitle,
  valueClassName = "text-[#293F60]",
  leftAccentColor,
  children,
  className = "",
}: MetricCardProps) {
  return (
    <div
      className={`relative overflow-hidden bg-white border border-[#E3EAF4] rounded-[13px] p-5 shadow-[0_3px_12px_rgba(45,70,110,0.055)] transition-all duration-200 hover:-translate-y-0.5 hover:shadow-[0_7px_18px_rgba(45,70,110,0.09)] ${
        leftAccentColor ? "" : "metric-card-accent"
      } ${className}`}
      style={leftAccentColor ? { borderLeft: `5px solid ${leftAccentColor}` } : undefined}
    >
      <div className="text-[0.75rem] font-bold uppercase tracking-[0.6px] text-[#718096] mb-2">
        {title}
      </div>
      {value !== "" && (
        <div className={`text-[1.85rem] font-extrabold leading-tight ${valueClassName}`}>
          {value}
        </div>
      )}
      {subtitle && <div className="text-[0.82rem] text-[#78869A] mt-1.5">{subtitle}</div>}
      {children}
    </div>
  );
}

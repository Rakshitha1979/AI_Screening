interface SkillBadgeProps {
  label: string;
  variant?: "pill" | "match" | "missing";
}

export function SkillBadge({ label, variant = "pill" }: SkillBadgeProps) {
  const styles =
    variant === "match"
      ? "bg-[#E0F5EB] text-[#16805B] rounded-[7px] px-2.5 py-1 text-[0.78rem] font-semibold"
      : variant === "missing"
      ? "bg-[#FCE9E9] text-[#B94B4B] rounded-[7px] px-2.5 py-1 text-[0.78rem] font-semibold"
      : "bg-[#EAF0FC] text-[#496BA6] rounded-full px-3 py-1 text-[0.8rem] font-semibold";

  return <span className={`inline-block mr-1.5 mb-1.5 ${styles}`}>{label}</span>;
}

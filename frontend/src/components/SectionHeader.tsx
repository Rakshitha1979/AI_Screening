interface SectionHeaderProps {
  title: string;
}

export function SectionHeader({ title }: SectionHeaderProps) {
  return (
    <div className="section-header-bar mt-6 mb-4 px-3.5 py-2.5 rounded-r-lg text-[#31527F] font-bold text-[1rem]">
      {title}
    </div>
  );
}

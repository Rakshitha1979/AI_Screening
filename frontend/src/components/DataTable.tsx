import { useMemo, useState } from "react";
import { ArrowDown, ArrowUp, ArrowUpDown } from "lucide-react";

export interface ColumnDef<T> {
  key: string;
  header: string;
  sortable?: boolean;
  align?: "left" | "right" | "center";
  sortValue?: (row: T) => string | number;
  render: (row: T) => React.ReactNode;
}

interface DataTableProps<T> {
  data: T[];
  columns: ColumnDef<T>[];
  rowKey: (row: T, index: number) => string;
  emptyMessage?: string;
  onRowClick?: (row: T) => void;
  selectedKey?: string;
}

export function DataTable<T>({
  data,
  columns,
  rowKey,
  emptyMessage = "No records available.",
  onRowClick,
  selectedKey,
}: DataTableProps<T>) {
  const [sortCol, setSortCol] = useState<string | null>(null);
  const [sortDir, setSortDir] = useState<"asc" | "desc">("asc");

  const sortedData = useMemo(() => {
    if (!sortCol) return data;
    const col = columns.find((c) => c.key === sortCol);
    if (!col || !col.sortValue) return data;
    const getter = col.sortValue;
    return [...data].sort((a, b) => {
      const va = getter(a);
      const vb = getter(b);
      if (typeof va === "number" && typeof vb === "number") {
        return sortDir === "asc" ? va - vb : vb - va;
      }
      return sortDir === "asc"
        ? String(va).localeCompare(String(vb))
        : String(vb).localeCompare(String(va));
    });
  }, [data, columns, sortCol, sortDir]);

  const handleSort = (col: ColumnDef<T>) => {
    if (!col.sortable) return;
    if (sortCol === col.key) {
      setSortDir((prev) => (prev === "asc" ? "desc" : "asc"));
    } else {
      setSortCol(col.key);
      setSortDir("desc");
    }
  };

  if (data.length === 0) {
    return (
      <div className="bg-white border border-[#E3EAF4] rounded-[11px] p-6 text-center text-sm text-[#64748B]">
        {emptyMessage}
      </div>
    );
  }

  return (
    <div className="bg-white border border-[#E3EAF4] rounded-[11px] overflow-x-auto shadow-[0_2px_9px_rgba(45,70,110,0.04)]">
      <table className="w-full text-left border-collapse text-sm">
        <thead>
          <tr className="bg-[#F8FAFD] border-b border-[#E3EAF4] text-[#31527F]">
            {columns.map((col) => (
              <th
                key={col.key}
                onClick={() => handleSort(col)}
                className={`px-3.5 py-2.5 font-semibold whitespace-nowrap select-none ${
                  col.sortable ? "cursor-pointer hover:bg-[#EDF3FC]" : ""
                } ${
                  col.align === "right"
                    ? "text-right"
                    : col.align === "center"
                    ? "text-center"
                    : "text-left"
                }`}
              >
                <span className="inline-flex items-center gap-1">
                  {col.header}
                  {col.sortable &&
                    (sortCol === col.key ? (
                      sortDir === "asc" ? (
                        <ArrowUp className="w-3.5 h-3.5 text-[#2563EB]" />
                      ) : (
                        <ArrowDown className="w-3.5 h-3.5 text-[#2563EB]" />
                      )
                    ) : (
                      <ArrowUpDown className="w-3.5 h-3.5 text-[#94A3B8]" />
                    ))}
                </span>
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-[#F0F2F7]">
          {sortedData.map((row, idx) => {
            const key = rowKey(row, idx);
            const isSelected = selectedKey && key === selectedKey;
            return (
              <tr
                key={key}
                onClick={() => onRowClick?.(row)}
                className={`${onRowClick ? "cursor-pointer hover:bg-[#F5F8FE]" : "hover:bg-[#FAFBFD]"} ${
                  isSelected ? "bg-[#EBF2FE]" : ""
                }`}
              >
                {columns.map((col) => (
                  <td
                    key={col.key}
                    className={`px-3.5 py-2.5 text-[#263449] align-middle ${
                      col.align === "right"
                        ? "text-right"
                        : col.align === "center"
                        ? "text-center"
                        : "text-left"
                    }`}
                  >
                    {col.render(row)}
                  </td>
                ))}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

import { Component, type ErrorInfo, type ReactNode } from "react";

interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
  message: string;
}

export class ErrorBoundary extends Component<Props, State> {
  state: State = { hasError: false, message: "" };

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, message: error.message };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("UI ErrorBoundary caught:", error, info);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="bg-white border border-red-200 rounded-xl p-6 max-w-xl mx-auto mt-10">
          <h2 className="text-lg font-bold text-red-700 mb-2">Something went wrong</h2>
          <p className="text-sm text-slate-600 mb-4">{this.state.message}</p>
          <button
            type="button"
            onClick={() => this.setState({ hasError: false, message: "" })}
            className="px-4 py-2 bg-[#2563EB] text-white rounded-lg text-sm font-semibold hover:bg-blue-700"
          >
            Try Again
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

export function PageSkeleton() {
  return (
    <div className="space-y-6 animate-pulse">
      <div className="h-8 w-80 bg-slate-200 rounded-lg" />
      <div className="h-4 w-96 bg-slate-200 rounded" />
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {[0, 1, 2, 3].map((i) => (
          <div key={i} className="h-28 bg-white border border-[#E3EAF4] rounded-[13px]" />
        ))}
      </div>
      <div className="h-64 bg-white border border-[#E3EAF4] rounded-[13px]" />
    </div>
  );
}

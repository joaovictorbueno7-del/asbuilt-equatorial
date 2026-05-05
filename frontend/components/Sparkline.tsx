type Point = { week: string; avg_score: number | null; count: number };

export default function Sparkline({ data, height = 64 }: { data: Point[]; height?: number }) {
  const filled = data.map((d) => d.avg_score ?? 0);
  if (filled.length === 0) return <div className="text-xs text-slate-500">Sem dados</div>;
  const max = Math.max(100, ...filled);
  const min = 0;
  const w = 100;
  const h = 100;
  const step = w / Math.max(1, filled.length - 1);
  const path = filled
    .map((v, i) => {
      const y = h - ((v - min) / (max - min)) * h;
      return `${i === 0 ? "M" : "L"}${(i * step).toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
  const area = `${path} L${w},${h} L0,${h} Z`;

  return (
    <svg viewBox="0 0 100 100" preserveAspectRatio="none" width="100%" height={height}>
      <defs>
        <linearGradient id="spark-grad" x1="0" x2="0" y1="0" y2="1">
          <stop offset="0%" stopColor="#22d3ee" stopOpacity="0.4" />
          <stop offset="100%" stopColor="#22d3ee" stopOpacity="0" />
        </linearGradient>
      </defs>
      <path d={area} fill="url(#spark-grad)" />
      <path d={path} fill="none" stroke="#22d3ee" strokeWidth="2" vectorEffect="non-scaling-stroke" />
      {filled.map((v, i) => (
        <circle
          key={i}
          cx={(i * step).toFixed(1)}
          cy={(h - ((v - min) / (max - min)) * h).toFixed(1)}
          r="1.5"
          fill={v > 0 ? "#22d3ee" : "#475569"}
          vectorEffect="non-scaling-stroke"
        />
      ))}
    </svg>
  );
}

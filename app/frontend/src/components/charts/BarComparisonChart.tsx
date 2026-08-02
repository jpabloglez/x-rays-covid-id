import React from "react";

export interface BarComparisonDatum {
  /** React key. */
  key: string;
  /** Display label, usually the track name. */
  label: string;
  /** null renders as `unmeasuredLabel` with no foreground bar. */
  value: number | null;
  /** A literal Tailwind `fill-*` class. Defaults to neutral slate. */
  barClassName?: string;
  /** The full sentence behind this number, shown as a hover tooltip. */
  detail?: string;
}

export interface BarComparisonChartProps {
  title: string;
  data: BarComparisonDatum[];
  /** The chart's scale. Supplied by the caller rather than auto-fit, so an
   * axis never silently rescales between renders. */
  domainMax: number;
  formatValue?: (value: number) => string;
  unmeasuredLabel?: string;
  width?: number;
  barHeight?: number;
  gap?: number;
}

const DEFAULT_FORMAT = (value: number) => value.toFixed(3);

/**
 * One horizontal bar per datum, hand-built rather than pulled from a charting
 * library -- this project shows two models and a handful of numbers, and a
 * dependency earns its keep at a different scale than that.
 *
 * No parallel data table for screen readers: every number this chart shows is
 * already rendered as plain text elsewhere on the same page (the `<dl>`, the
 * retention paragraph), so the chart is a supplementary view of numbers that
 * are never chart-only. The accessible name carries the same information the
 * bars do.
 */
const BarComparisonChart: React.FC<BarComparisonChartProps> = ({
  title,
  data,
  domainMax,
  formatValue = DEFAULT_FORMAT,
  unmeasuredLabel = "not measured",
  width = 480,
  barHeight = 28,
  gap = 12,
}) => {
  const labelColumn = 96;
  const valueColumn = 64;
  const trackWidth = Math.max(1, width - labelColumn - valueColumn);
  const rowHeight = barHeight + gap;
  const height = data.length * rowHeight;

  const describe = (datum: BarComparisonDatum) =>
    datum.value === null ? unmeasuredLabel : formatValue(datum.value);

  const summary = `${title}: ${data.map((datum) => `${datum.label} ${describe(datum)}`).join(", ")}`;

  return (
    <figure>
      <h4 className="text-sm font-semibold text-slate-900">{title}</h4>
      <svg
        role="img"
        aria-label={summary}
        viewBox={`0 0 ${width} ${height}`}
        width="100%"
        height={height}
        className="mt-2"
      >
        {data.map((datum, index) => {
          const y = index * rowHeight;
          const fraction =
            datum.value === null ? 0 : Math.max(0, Math.min(1, datum.value / domainMax));
          return (
            <g key={datum.key}>
              <title>{datum.detail ?? `${datum.label}: ${describe(datum)}`}</title>
              <text
                x={0}
                y={y + barHeight / 2}
                dominantBaseline="middle"
                className="fill-slate-700 font-sans text-xs"
              >
                {datum.label}
              </text>
              <rect
                x={labelColumn}
                y={y}
                width={trackWidth}
                height={barHeight}
                rx={3}
                className="fill-slate-200"
              />
              {datum.value !== null && (
                <rect
                  x={labelColumn}
                  y={y}
                  width={trackWidth * fraction}
                  height={barHeight}
                  rx={3}
                  className={datum.barClassName ?? "fill-slate-600"}
                />
              )}
              <text
                x={width}
                y={y + barHeight / 2}
                textAnchor="end"
                dominantBaseline="middle"
                className="fill-slate-900 font-mono text-xs tabular-nums"
              >
                {describe(datum)}
              </text>
            </g>
          );
        })}
      </svg>
    </figure>
  );
};

export default BarComparisonChart;

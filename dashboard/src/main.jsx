import { StrictMode, useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import Chart from 'chart.js/auto';
import { getEngine, getEngines, getPrediction, hostedDemo } from './data.js';
import './styles.css';

const formatEstimate = new Intl.NumberFormat('en', { maximumFractionDigits: 2 });
const formatReading = new Intl.NumberFormat('en', { maximumFractionDigits: 5 });
const chartInk = '#526076';
const chartGrid = '#e7edf1';
const chartLine = '#087c70';

function SensorChart({ readings, selectedIndex, sensor }) {
  const canvas = useRef(null);

  useEffect(() => {
    const visible = readings.slice(0, selectedIndex + 1);
    const chart = new Chart(canvas.current, {
      type: 'line',
      data: {
        datasets: [{
          data: visible.map((reading) => ({ x: reading.cycle, y: reading.sensors[sensor] })),
          borderColor: chartLine,
          backgroundColor: '#087c7010',
          borderWidth: 2,
          clip: 6,
          fill: 'start',
          pointRadius: ({ dataIndex }) => dataIndex === selectedIndex ? 5 : 0,
          pointBackgroundColor: chartLine,
          pointHitRadius: 8,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        layout: { padding: { right: 7 } },
        interaction: { mode: 'nearest', axis: 'x', intersect: false },
        scales: {
          x: {
            type: 'linear',
            min: readings[0].cycle,
            max: Math.max(readings.at(-1).cycle, readings[0].cycle + 1),
            grid: { color: chartGrid },
            ticks: { color: chartInk, precision: 0, maxTicksLimit: 8 },
            title: { display: true, text: 'Operating cycle', color: chartInk },
          },
          y: {
            grid: { color: chartGrid },
            ticks: { color: chartInk, maxTicksLimit: 6 },
          },
        },
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              title: (items) => `Cycle ${items[0].parsed.x}`,
              label: (item) => `Sensor ${sensor + 1} · ${formatReading.format(item.parsed.y)}`,
            },
          },
        },
      },
    });
    return () => chart.destroy();
  }, [readings, selectedIndex, sensor]);

  return (
    <div className="plot">
      <canvas
        ref={canvas}
        role="img"
        aria-label={`Sensor ${sensor + 1} readings from cycle ${readings[0].cycle} to ${readings[selectedIndex].cycle}`}
      />
    </div>
  );
}

function Prediction({ reading }) {
  const [estimate, setEstimate] = useState(null);
  const [error, setError] = useState(null);
  const [pending, setPending] = useState(false);
  const activeRequest = useRef(null);

  useEffect(() => () => activeRequest.current?.abort(), []);

  async function predict() {
    const controller = new AbortController();
    activeRequest.current = controller;
    setPending(true);
    setError(null);
    try {
      const result = await getPrediction(reading, controller.signal);
      if (!controller.signal.aborted) setEstimate(result);
    } catch (error) {
      if (!controller.signal.aborted) setError(error.message);
    } finally {
      if (!controller.signal.aborted) setPending(false);
    }
  }

  return (
    <>
      <button type="button" className="primary" onClick={predict} disabled={pending}>
        {pending ? 'Estimating…' : hostedDemo ? 'Show estimate' : 'Get estimate'}
      </button>
      {error && <p className="error" role="alert">{error}</p>}
      <div className="forecast" aria-live="polite" aria-busy={pending}>
        <p className="forecast-label">Estimated life left</p>
        {estimate === null ? (
          <p className="forecast-empty">Your estimate will appear here.</p>
        ) : (
          <p className="forecast-value"><strong>{formatEstimate.format(estimate)}</strong><span>cycles</span></p>
        )}
      </div>
      <p className="model-note">
        Random Forest · uses all {reading.sensors.length} sensor readings at cycle {reading.cycle}.
      </p>
    </>
  );
}

function CycleControl({ readings, index, onChange }) {
  const [expanded, setExpanded] = useState(false);
  const cycle = readings[index].cycle;
  const latestIndex = readings.length - 1;
  const latest = index === latestIndex;

  return (
    <div className="cycle-control">
      <div className="cycle-summary">
        <div>
          <p className="control-label">Reading used for estimate</p>
          <p className="cycle-current"><strong>Cycle {cycle}</strong> · {latest ? 'Latest reading' : 'Earlier reading'}</p>
        </div>
        {latestIndex > 0 && (
          <button
            type="button"
            className="text-button"
            aria-expanded={expanded}
            aria-controls="cycle-options"
            onClick={() => setExpanded(!expanded)}
          >
            {expanded ? 'Done' : 'Change cycle'}
          </button>
        )}
      </div>
      {expanded && (
        <div id="cycle-options" className="cycle-options">
          <label htmlFor="cycle">Prediction cycle</label>
          <input
            id="cycle"
            type="range"
            min="0"
            max={latestIndex}
            value={index}
            aria-valuetext={`Cycle ${cycle}`}
            onChange={(event) => onChange(Number(event.target.value))}
          />
          <div className="range-labels">
            <span>Cycle {readings[0].cycle}</span>
            <span>Cycle {readings[latestIndex].cycle}</span>
          </div>
        </div>
      )}
      {!latest && (
        <button type="button" className="text-button reset-cycle" onClick={() => onChange(latestIndex)}>
          Use latest reading
        </button>
      )}
    </div>
  );
}

function SensorPanel({ engine, readings, index, sensor, onSensorChange }) {
  const reading = readings[index];

  return (
    <section className="panel sensor-panel" aria-labelledby="sensor-heading">
      <div className="panel-header">
        <div>
          <h2 id="sensor-heading">Sensor history</h2>
          <p className="muted">Engine {engine} through cycle {reading.cycle}</p>
        </div>
        <label className="sensor-control">
          <span className="sr-only">Sensor</span>
          <select value={sensor} onChange={(event) => onSensorChange(Number(event.target.value))}>
            {reading.sensors.map((_, index) => <option key={index} value={index}>Sensor {index + 1}</option>)}
          </select>
        </label>
      </div>
      <dl className="reading">
        <dt>Sensor {sensor + 1} at cycle {reading.cycle}</dt>
        <dd>{formatReading.format(reading.sensors[sensor])}</dd>
      </dl>
      <SensorChart readings={readings} selectedIndex={index} sensor={sensor} />
      <p className="chart-caption"><span aria-hidden="true" />The dot marks the reading used for the estimate.</p>
    </section>
  );
}

function EngineWorkspace({ engines, engine, onEngineChange }) {
  const [load, setLoad] = useState(null);
  const [attempt, setAttempt] = useState(0);
  const [selectedIndex, setSelectedIndex] = useState(null);
  const [sensor, setSensor] = useState(10);

  useEffect(() => {
    const controller = new AbortController();
    getEngine(engine, controller.signal)
      .then((data) => {
        if (!controller.signal.aborted) setLoad({ engine, readings: data.readings });
      })
      .catch((error) => {
        if (!controller.signal.aborted) setLoad({ engine, error: error.message });
      });
    return () => controller.abort();
  }, [engine, attempt]);

  const current = load?.engine === engine ? load : null;
  const readings = current?.readings;
  const index = readings ? selectedIndex ?? readings.length - 1 : null;
  const reading = readings?.[index];

  function selectEngine(event) {
    setLoad(null);
    setSelectedIndex(null);
    onEngineChange(Number(event.target.value));
  }

  return (
    <div className="workspace">
      <section className="panel controls" aria-labelledby="estimate-heading">
        <h2 id="estimate-heading">Get a life estimate</h2>
        <label className="engine-control">
          <span className="control-label">Demo engine</span>
          <select value={engine} onChange={selectEngine}>
            {engines.map(({ engine_id }) => <option key={engine_id} value={engine_id}>Engine {engine_id}</option>)}
          </select>
        </label>
        {current?.error && <LoadError message={current.error} retry={() => { setLoad(null); setAttempt(attempt + 1); }} />}
        {!current && <p className="muted" role="status">Loading engine {engine}…</p>}
        {reading && (
          <>
            <CycleControl key={engine} readings={readings} index={index} onChange={setSelectedIndex} />
            <Prediction key={`${engine}-${reading.cycle}`} reading={reading} />
          </>
        )}
      </section>
      {reading && (
        <SensorPanel engine={engine} readings={readings} index={index} sensor={sensor} onSensorChange={setSensor} />
      )}
    </div>
  );
}

function LoadError({ message, retry }) {
  return (
    <div role="alert" className="load-error">
      <p className="error">{message}</p>
      <button type="button" className="secondary" onClick={retry}>Try again</button>
    </div>
  );
}

function App() {
  const [engines, setEngines] = useState(null);
  const [engine, setEngine] = useState(null);
  const [error, setError] = useState(null);
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    const controller = new AbortController();
    getEngines(controller.signal)
      .then((data) => {
        if (controller.signal.aborted) return;
        setEngines(data);
        setEngine(data[0]?.engine_id ?? null);
      })
      .catch((error) => {
        if (!controller.signal.aborted) setError(error.message);
      });
    return () => controller.abort();
  }, [attempt]);

  return (
    <>
      <header className="site-header">
        <div className="shell header-content">
          <div className="wordmark"><span className="brand-mark" aria-hidden="true">PX</span>Predict‑X</div>
          <nav aria-label="Project links">
            <a href="https://github.com/jattu8602/Predict-X#readme">Project details</a>
            {!hostedDemo && <a href="/docs">API docs</a>}
          </nav>
        </div>
      </header>
      <main className="shell">
        <div className="page-header">
          <div className="source-label">
            <span>NASA C-MAPSS · FD001</span>
            {engines?.length > 0 && <span className="source-count">{engines.length} simulated engines</span>}
          </div>
          <h1>How long can this engine run?</h1>
          <p className="intro">
            {hostedDemo
              ? 'Pick an engine to explore its sensor history and see how many operating cycles it was estimated to have left.'
              : 'Pick a simulated engine and estimate how many operating steps (cycles) it has left.'}
          </p>
        </div>
        {error && <div className="panel"><LoadError message={error} retry={() => { setError(null); setAttempt(attempt + 1); }} /></div>}
        {!engines && !error && <p className="panel muted" role="status">Loading demo engines…</p>}
        {engines?.length === 0 && <p className="panel muted">There are no demo engines to display.</p>}
        {engine !== null && <EngineWorkspace engines={engines} engine={engine} onEngineChange={setEngine} />}
      </main>
    </>
  );
}

createRoot(document.getElementById('root')).render(<StrictMode><App /></StrictMode>);

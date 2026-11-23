import { StrictMode, useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import Chart from 'chart.js/auto';
import { getEngine, getEngines, getPrediction, hostedDemo } from './data.js';
import './styles.css';

const formatEstimate = new Intl.NumberFormat('en', { maximumFractionDigits: 2 });
const formatReading = new Intl.NumberFormat('en', { maximumFractionDigits: 5 });

function SensorChart({ readings, selectedIndex, sensor }) {
  const canvas = useRef(null);

  useEffect(() => {
    const visible = readings.slice(0, selectedIndex + 1);
    const chart = new Chart(canvas.current, {
      type: 'line',
      data: {
        datasets: [{
          data: visible.map((reading) => ({ x: reading.cycle, y: reading.sensors[sensor] })),
          borderColor: '#54dfc1',
          backgroundColor: 'rgba(84, 223, 193, 0.08)',
          borderWidth: 2,
          fill: 'start',
          pointRadius: (context) => context.dataIndex === selectedIndex ? 4 : 0,
          pointBackgroundColor: '#54dfc1',
          pointHitRadius: 8,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        interaction: { mode: 'nearest', axis: 'x', intersect: false },
        scales: {
          x: {
            type: 'linear',
            min: readings[0].cycle,
            max: Math.max(readings.at(-1).cycle, readings[0].cycle + 1),
            grid: { color: '#223249' },
            ticks: { color: '#a5b5ca', precision: 0, maxTicksLimit: 8 },
            title: { display: true, text: 'Operating cycle', color: '#a5b5ca' },
          },
          y: {
            grid: { color: '#223249' },
            ticks: { color: '#a5b5ca', maxTicksLimit: 6 },
          },
        },
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              title: (items) => `Cycle ${items[0].parsed.x}`,
              label: (item) => `Sensor ${sensor + 1} · ${item.parsed.y}`,
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

function PredictionCard({ reading }) {
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
      setEstimate(await getPrediction(reading, controller.signal));
    } catch (error) {
      if (!controller.signal.aborted) setError(error.message);
    } finally {
      if (!controller.signal.aborted) setPending(false);
    }
  }

  return (
    <section className="panel prediction" aria-labelledby="prediction-heading">
      <div>
        <h2 id="prediction-heading">Remaining life</h2>
        <p className="muted">Random Forest estimate at cycle {reading.cycle}</p>
      </div>
      <div className="prediction-value" aria-live="polite">
        {estimate === null ? <p className="muted">Run a prediction to see how many cycles the model estimates are left.</p> : (
          <>
            <strong>{formatEstimate.format(estimate)}</strong>
            <span>cycles remaining</span>
          </>
        )}
      </div>
      {error && <p className="error" role="alert">{error}</p>}
      <button type="button" className="primary" onClick={predict} disabled={pending}>
        {pending ? 'Predicting…' : hostedDemo ? 'Show remaining-life estimate' : 'Predict remaining life'}
      </button>
    </section>
  );
}

function EngineWorkspace({ engine }) {
  const [readings, setReadings] = useState(null);
  const [error, setError] = useState(null);
  const [attempt, setAttempt] = useState(0);
  const [selectedIndex, setSelectedIndex] = useState(null);
  const [sensor, setSensor] = useState(10);

  useEffect(() => {
    const controller = new AbortController();
    getEngine(engine, controller.signal)
      .then((data) => setReadings(data.readings))
      .catch((error) => {
        if (!controller.signal.aborted) setError(error.message);
      });
    return () => controller.abort();
  }, [engine, attempt]);

  if (error) {
    return <LoadError message={error} retry={() => { setError(null); setAttempt(attempt + 1); }} />;
  }
  if (!readings) return <p className="panel message" role="status">Loading engine {engine}…</p>;

  const index = selectedIndex ?? readings.length - 1;
  const reading = readings[index];
  const finalCycle = readings.at(-1).cycle;

  return (
    <>
      <section className="panel timeline" aria-label="Prediction cycle">
        <div className="cycle-heading">
          <label htmlFor="cycle">Prediction cycle</label>
          <output htmlFor="cycle"><strong>{reading.cycle}</strong> / {finalCycle}</output>
        </div>
        <input
          id="cycle"
          type="range"
          min="0"
          max={readings.length - 1}
          value={index}
          aria-valuetext={`Cycle ${reading.cycle}`}
          disabled={readings.length === 1}
          onChange={(event) => setSelectedIndex(Number(event.target.value))}
        />
        <div className="range-labels"><span>Cycle {readings[0].cycle}</span><span>Last recorded cycle</span></div>
      </section>
      <div className="workspace">
        <section className="panel sensor-panel" aria-labelledby="sensor-heading">
          <div className="panel-header">
            <div>
              <h2 id="sensor-heading">Sensor history</h2>
              <p className="muted">Engine {engine} through cycle {reading.cycle}</p>
            </div>
            <label className="sensor-control">
              <span className="sr-only">Sensor</span>
              <select value={sensor} onChange={(event) => setSensor(Number(event.target.value))}>
                {reading.sensors.map((_, index) => <option key={index} value={index}>Sensor {index + 1}</option>)}
              </select>
            </label>
          </div>
          <div className="latest-reading">
            <strong>{formatReading.format(reading.sensors[sensor])}</strong>
            <span>Reading at cycle {reading.cycle}</span>
          </div>
          <SensorChart readings={readings} selectedIndex={index} sensor={sensor} />
        </section>
        <PredictionCard key={reading.cycle} reading={reading} />
      </div>
    </>
  );
}

function LoadError({ message, retry }) {
  return (
    <div className="panel message" role="alert">
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
          <a href="https://github.com/jattu8602/Predict-X#readme">Method &amp; results</a>
        </div>
      </header>
      <main className="shell">
        <div className="page-header">
          <div>
            <p className="dataset">NASA C-MAPSS / FD001{hostedDemo && " · Hosted demo"}</p>
            <h1>Engine health dashboard</h1>
            <p className="muted">
              {hostedDemo
                ? 'Explore simulated sensor readings and see saved Random Forest estimates for each engine and cycle.'
                : 'Explore simulated sensor readings and estimate how many operating cycles an engine has left.'}
            </p>
          </div>
          {engines?.length > 0 && (
            <label className="engine-control">
              <span>Demo engine</span>
              <select value={engine} onChange={(event) => setEngine(Number(event.target.value))}>
                {engines.map(({ engine_id }) => <option key={engine_id} value={engine_id}>Engine {engine_id}</option>)}
              </select>
            </label>
          )}
        </div>
        {error && <LoadError message={error} retry={() => { setError(null); setAttempt(attempt + 1); }} />}
        {!engines && !error && <p className="panel message" role="status">Loading demo engines…</p>}
        {engines?.length === 0 && <p className="panel message">There are no demo engines to display.</p>}
        {engine !== null && <EngineWorkspace key={engine} engine={engine} />}
      </main>
    </>
  );
}

createRoot(document.getElementById('root')).render(<StrictMode><App /></StrictMode>);

export const hostedDemo = import.meta.env.VITE_PREDICTX_HOSTED_DEMO === '1';

async function request(url, options) {
  let response;
  try {
    response = await fetch(url, options);
  } catch (error) {
    if (error.name === 'AbortError') throw error;
    throw new Error(hostedDemo ? 'Could not reach the demo data.' : 'Could not reach the Predict-X API.');
  }

  if (!response.ok) {
    const problem = await response.json().catch(() => null);
    const message = typeof problem?.detail === 'string' ? problem.detail : null;
    throw new Error(message || `The request failed (HTTP ${response.status}).`);
  }
  return response.json();
}

export function getEngines(signal) {
  return request(hostedDemo ? './demo/engines.json' : '/api/engines', { signal });
}

export function getEngine(engine, signal) {
  return request(hostedDemo ? `./demo/engine-${engine}.json` : `/api/engines/${engine}`, { signal });
}

export async function getPrediction(reading, signal) {
  if (hostedDemo) return reading.predicted_rul_cycles;
  const response = await request('/api/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(reading),
    signal,
  });
  return response.predicted_rul_cycles;
}

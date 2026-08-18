/* ResearchAI — optional backend seam.
 *
 * The demo runs entirely in the browser. If you later put a service in front
 * of it, set RA_API_BASE to its origin; the app will POST the mission config
 * to `${RA_API_BASE}/api/compose` and render whatever proposal JSON comes back
 * (same shape as RA.engine.compose). If the call fails, it falls back to the
 * local engine, so the demo never breaks.
 */
window.RA_API_BASE = '';

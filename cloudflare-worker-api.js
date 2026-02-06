/**
 * BRAIAIN Community Stats API
 * Cloudflare Worker + KV
 * 
 * SETUP:
 * 1. Go to Cloudflare Dashboard → Workers & Pages → Create Worker
 * 2. Paste this code
 * 3. Go to Settings → Variables → KV Namespace Bindings
 * 4. Create a KV namespace called "BRAIAIN_STATS"
 * 5. Bind it with variable name: STATS
 * 6. Add a Custom Domain: api.braiain.com
 * 7. Deploy!
 * 
 * ENDPOINTS:
 *   POST /vote   { id: "d2r1", correct: true }
 *   GET  /stats?id=d2r1  → { id, pct, total, correct, incorrect }
 *   GET  /health         → { ok: true }
 */

export default {
  async fetch(request, env) {
    // CORS headers for braiain.com
    const corsHeaders = {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type',
      'Access-Control-Max-Age': '86400',
    };

    // Handle preflight
    if (request.method === 'OPTIONS') {
      return new Response(null, { status: 204, headers: corsHeaders });
    }

    const url = new URL(request.url);
    const path = url.pathname;

    try {
      // ── Health check ──
      if (path === '/health') {
        return json({ ok: true, ts: Date.now() }, corsHeaders);
      }

      // ── POST /vote ──
      if (path === '/vote' && request.method === 'POST') {
        const body = await request.json();
        const { id, correct } = body;

        if (!id || typeof correct !== 'boolean') {
          return json({ error: 'Bad request: need id (string) and correct (boolean)' }, corsHeaders, 400);
        }

        // Read current stats from KV
        const key = `round:${id}`;
        const existing = await env.STATS.get(key, 'json');
        const stats = existing || { correct: 0, incorrect: 0, total: 0 };

        // Increment
        if (correct) {
          stats.correct++;
        } else {
          stats.incorrect++;
        }
        stats.total = stats.correct + stats.incorrect;

        // Write back (expire after 90 days)
        await env.STATS.put(key, JSON.stringify(stats), { expirationTtl: 7776000 });

        return json({ ok: true, id, total: stats.total }, corsHeaders);
      }

      // ── GET /stats ──
      if (path === '/stats' && request.method === 'GET') {
        const id = url.searchParams.get('id');
        if (!id) {
          return json({ error: 'Missing id parameter' }, corsHeaders, 400);
        }

        const key = `round:${id}`;
        const stats = await env.STATS.get(key, 'json');

        if (!stats || stats.total === 0) {
          return json({ id, pct: 0, total: 0, correct: 0, incorrect: 0 }, corsHeaders);
        }

        const pct = (stats.correct / stats.total) * 100;

        return json({
          id,
          pct: Math.round(pct * 10) / 10,
          total: stats.total,
          correct: stats.correct,
          incorrect: stats.incorrect
        }, corsHeaders);
      }

      // ── 404 ──
      return json({ error: 'Not found' }, corsHeaders, 404);

    } catch (err) {
      return json({ error: 'Internal error', detail: err.message }, corsHeaders, 500);
    }
  }
};

function json(data, corsHeaders, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: {
      'Content-Type': 'application/json',
      ...corsHeaders
    }
  });
}

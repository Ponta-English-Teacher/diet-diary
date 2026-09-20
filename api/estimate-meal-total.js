// Vercel serverless function: POST /api/estimate-meal-total
//
// Whole-photo calorie sanity check — a SECOND, independent look at the meal
// photo, after the client has already computed a component-based subtotal
// (FOOD_DB matches + per-item AI estimates). This never replaces that
// pipeline; it only flags when the subtotal looks implausibly low for what
// the photo actually shows (e.g. a composite dish like curry rice that got
// split into "ご飯" + "カレー" and under-counted) and offers an independent
// whole-plate range so the app can reconcile — never silently overwrite —
// the displayed total. Third (and last) place that reads the OpenAI API key.
//
// Request body:  { imageBase64: string, items?: [{name, amount, unit}], subtotalKcal?: number|null }
// Success reply: { ok: true, estimateLow: number, estimateHigh: number, subtotalLooksLow: boolean }
// "Can't estimate": { ok: true, estimateLow: null, estimateHigh: null, subtotalLooksLow: false }
// Failure reply: { ok: false, error: string }  (with a matching HTTP status)

const OPENAI_CHAT_URL = 'https://api.openai.com/v1/chat/completions';
const OPENAI_MODEL = process.env.OPENAI_VISION_MODEL || 'gpt-4o-mini';

const MAX_BASE64_CHARS = 6_000_000; // same generous cap as /api/analyze-meal
const MAX_ITEMS_TEXT = 12;
const BASE64_RE = /^[A-Za-z0-9+/]+={0,2}$/;

const SYSTEM_PROMPT = `You are a nutrition sanity-check assistant for a Japanese diet diary app.
You will see a meal photo and a component-based calorie subtotal that was
built by identifying individual food items in the photo and estimating each
one's calories separately. That approach can under-count: it can miss food,
underestimate portion sizes, or split a single composite dish (curry rice,
a rice bowl, ramen, a hamburger steak, etc.) into simpler ingredients whose
combined estimate misses sauce, oil, or how the parts really combine.

Your job is an INDEPENDENT whole-plate sanity check, not a replacement
calculation. Look at the entire photo as a whole and give a realistic total
calorie RANGE for everything visible, based on the full portion size — do
not just re-derive the same item list you're given.

Then say whether the given component subtotal looks implausibly LOW for what
the photo actually shows.

Respond with ONLY a JSON object in exactly this shape, no other text:
{"estimateLow": number, "estimateHigh": number, "subtotalLooksLow": boolean}
If you cannot reasonably estimate from the photo, respond with
{"estimateLow": null, "estimateHigh": null, "subtotalLooksLow": false}.`;

function sendJson(res, status, payload) {
  res.status(status).setHeader('Content-Type', 'application/json');
  res.end(JSON.stringify(payload));
}

function fail(res, status, error) {
  sendJson(res, status, { ok: false, error });
}

function normalizeImageDataUrl(imageBase64) {
  if (imageBase64.startsWith('data:image/')) return imageBase64;
  return `data:image/jpeg;base64,${imageBase64}`;
}

function rawBase64Payload(imageBase64) {
  const commaIdx = imageBase64.indexOf(',');
  return imageBase64.startsWith('data:') && commaIdx !== -1
    ? imageBase64.slice(commaIdx + 1)
    : imageBase64;
}

function sanitizeItemsForPrompt(rawItems) {
  if (!Array.isArray(rawItems)) return [];
  const out = [];
  for (const raw of rawItems) {
    if (!raw || typeof raw !== 'object') continue;
    const name = typeof raw.name === 'string' ? raw.name.trim() : '';
    if (!name) continue;
    const amount = Number(raw.amount);
    const unit = typeof raw.unit === 'string' ? raw.unit.trim() : '';
    out.push({
      name,
      amount: Number.isFinite(amount) ? amount : null,
      unit: unit || null
    });
    if (out.length >= MAX_ITEMS_TEXT) break;
  }
  return out;
}

module.exports = async function handler(req, res) {
  if (req.method !== 'POST') {
    res.setHeader('Allow', 'POST');
    return fail(res, 405, 'Method not allowed. Use POST.');
  }

  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) {
    return fail(res, 500, 'Server is not configured (missing OpenAI API key).');
  }

  let body = req.body;
  if (typeof body === 'string') {
    try { body = JSON.parse(body); } catch { return fail(res, 400, 'Request body is not valid JSON.'); }
  }
  if (!body || typeof body !== 'object') {
    return fail(res, 400, 'Request body must be a JSON object.');
  }

  const { imageBase64, items, subtotalKcal } = body;

  if (!imageBase64 || typeof imageBase64 !== 'string') {
    return fail(res, 400, 'imageBase64 is required and must be a string.');
  }
  if (imageBase64.length > MAX_BASE64_CHARS) {
    return fail(res, 413, 'Image is too large.');
  }
  const rawPayload = rawBase64Payload(imageBase64);
  if (rawPayload.length < 100 || !BASE64_RE.test(rawPayload)) {
    return fail(res, 400, 'imageBase64 does not look like valid base64 image data.');
  }
  if (subtotalKcal != null && typeof subtotalKcal !== 'number') {
    return fail(res, 400, 'subtotalKcal must be a number if provided.');
  }

  const safeItems = sanitizeItemsForPrompt(items);
  const dataUrl = normalizeImageDataUrl(imageBase64);

  const itemsText = safeItems.length
    ? safeItems.map(it => `- ${it.name}${it.amount != null ? ` (${it.amount}${it.unit || ''})` : ''}`).join('\n')
    : '(no items provided)';
  const subtotalText = typeof subtotalKcal === 'number'
    ? `${Math.round(subtotalKcal)} kcal`
    : 'unknown (could not be computed)';
  const userText = `Detected items and amounts:\n${itemsText}\n\nComponent-based subtotal: ${subtotalText}\n\nGive your independent whole-plate estimate and say whether that subtotal looks implausibly low.`;

  let openaiResp;
  try {
    openaiResp = await fetch(OPENAI_CHAT_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${apiKey}`
      },
      body: JSON.stringify({
        model: OPENAI_MODEL,
        max_tokens: 150,
        response_format: { type: 'json_object' },
        messages: [
          { role: 'system', content: SYSTEM_PROMPT },
          {
            role: 'user',
            content: [
              { type: 'text', text: userText },
              { type: 'image_url', image_url: { url: dataUrl } }
            ]
          }
        ]
      })
    });
  } catch (e) {
    console.error('estimate-meal-total OpenAI fetch failed:', e);
    return fail(res, 502, 'Could not reach OpenAI.');
  }

  if (!openaiResp.ok) {
    let detail = '';
    try { detail = (await openaiResp.json()).error?.message || ''; } catch {}
    return fail(res, 502, `OpenAI request failed${detail ? ': ' + detail : ''}.`);
  }

  let completion;
  try {
    completion = await openaiResp.json();
  } catch {
    return fail(res, 502, 'OpenAI returned an unreadable response.');
  }

  const content = completion?.choices?.[0]?.message?.content;
  if (!content) {
    return fail(res, 502, 'OpenAI response did not contain a result.');
  }

  let parsed;
  try {
    parsed = JSON.parse(content);
  } catch {
    return fail(res, 502, 'Could not parse the estimate as JSON.');
  }

  const low = Number(parsed.estimateLow);
  const high = Number(parsed.estimateHigh);
  if (parsed.estimateLow == null || parsed.estimateHigh == null || !Number.isFinite(low) || !Number.isFinite(high) || low < 0 || high < low) {
    // Can't produce a usable range — never fabricate one.
    return sendJson(res, 200, { ok: true, estimateLow: null, estimateHigh: null, subtotalLooksLow: false });
  }

  return sendJson(res, 200, {
    ok: true,
    estimateLow: Math.round(low),
    estimateHigh: Math.round(high),
    subtotalLooksLow: parsed.subtotalLooksLow === true
  });
};

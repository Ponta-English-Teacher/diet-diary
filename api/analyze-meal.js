// Vercel serverless function: POST /api/analyze-meal
//
// Accepts a meal photo (base64) and returns structured food/portion guesses
// from an OpenAI vision-capable model. This is the ONLY place that reads the
// OpenAI API key — it is never sent to, or readable by, the browser.
//
// Request body:  { imageBase64: string, mealType?: string }
// Success reply: { ok: true, items: [{ name, amount, unit, confidence }] }
// Failure reply: { ok: false, error: string }   (with a matching HTTP status)
//
// Calories are deliberately NOT requested from the model — Diet Diary
// computes those locally from FOOD_DB once an item is matched client-side.

const OPENAI_CHAT_URL = 'https://api.openai.com/v1/chat/completions';
const OPENAI_MODEL = process.env.OPENAI_VISION_MODEL || 'gpt-4o-mini';

const MAX_BASE64_CHARS = 6_000_000; // ~4.5MB decoded — generous for a downscaled JPEG
const MAX_ITEMS = 12;
const VALID_CONFIDENCE = new Set(['high', 'medium', 'low']);
const BASE64_RE = /^[A-Za-z0-9+/]+={0,2}$/;

const SYSTEM_PROMPT = `You are a food-recognition assistant for a Japanese diet diary app.
Look at the meal photo and identify the distinct food items visible.
For each item, estimate a reasonable serving amount and a simple unit
(use grams "g" for most foods, or a countable unit like "個" for things like eggs).
Use common Japanese food names (e.g. 鶏むね肉, 豆腐, ご飯, サラダ).
Do NOT estimate or mention calories — that is calculated separately.
Respond with ONLY a JSON object in exactly this shape, no other text:
{"items":[{"name":"string","amount":number,"unit":"string","confidence":"high"|"medium"|"low"}]}
List at most ${MAX_ITEMS} items. If you cannot identify any food, return {"items":[]}.`;

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

function sanitizeItems(rawItems) {
  if (!Array.isArray(rawItems)) return [];
  const items = [];
  for (const raw of rawItems) {
    if (!raw || typeof raw !== 'object') continue;
    const name = typeof raw.name === 'string' ? raw.name.trim() : '';
    if (!name) continue;

    const amountNum = Number(raw.amount);
    const amount = Number.isFinite(amountNum) ? amountNum : null;
    const unit = typeof raw.unit === 'string' ? raw.unit.trim() : '';
    const confidence = VALID_CONFIDENCE.has(raw.confidence) ? raw.confidence : 'medium';

    items.push({ name, amount, unit, confidence });
    if (items.length >= MAX_ITEMS) break;
  }
  return items;
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

  const { imageBase64, mealType } = body;

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
  if (mealType != null && typeof mealType !== 'string') {
    return fail(res, 400, 'mealType must be a string if provided.');
  }

  const dataUrl = normalizeImageDataUrl(imageBase64);
  const userText = mealType
    ? `This photo is of a ${mealType} (meal type). Identify the foods and estimate portions.`
    : 'Identify the foods in this meal photo and estimate portions.';

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
        max_tokens: 600,
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
    console.error('OpenAI fetch failed:', e);
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
    return fail(res, 502, 'Could not parse the analysis result as JSON.');
  }

  const items = sanitizeItems(parsed.items);
  return sendJson(res, 200, { ok: true, items });
};

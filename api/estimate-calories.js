// Vercel serverless function: POST /api/estimate-calories
//
// Text-only calorie estimate for a single food, used when a food doesn't
// match the client's local FOOD_DB (manual entry or photo detection). This
// is the second (and only other) place that reads the OpenAI API key — it is
// never sent to, or readable by, the browser. Mirrors api/analyze-meal.js's
// conventions; that file is untouched by this one.
//
// Two request shapes, matching how the client already models a food:
//   { name: string, unit: string }     -> "basis" mode: a per-100(g/ml) or
//                                          per-1(count unit) rate, so the
//                                          client can reuse it like a FOOD_DB
//                                          entry as the amount changes.
//   { name: string, portion: 's'|'m'|'l' } -> "flat" mode: a single estimate
//                                          for one serving of that size.
//
// Success reply (basis):  { ok: true, mode: 'basis', kcal: number, per: number, unit: string }
// Success reply (flat):   { ok: true, mode: 'flat', kcal: number }
// "Can't estimate" reply: { ok: true, mode: ..., kcal: null }  (never a fabricated number)
// Failure reply:           { ok: false, error: string }  (with a matching HTTP status)

const OPENAI_CHAT_URL = 'https://api.openai.com/v1/chat/completions';
const OPENAI_MODEL = process.env.OPENAI_TEXT_MODEL || process.env.OPENAI_VISION_MODEL || 'gpt-4o-mini';

const MAX_NAME_CHARS = 80;
const COUNT_UNITS = new Set(['個', '枚', '本', '杯', '切れ', '皿', '人前', '粒']);
const VALID_PORTIONS = new Set(['s', 'm', 'l']);

const SYSTEM_PROMPT_BASIS = `You are a calorie estimation assistant for a Japanese diet diary app.
Given a food name and a measurement unit, estimate a typical calorie value for that food using ONLY that unit.
- If the unit is "g" or "ml", respond with the typical kcal per 100 of that unit.
- If the unit is a countable unit (such as 個, 枚, 本, 杯, 切れ, 皿, 人前, 粒), respond with the typical kcal for 1 of that unit.
Use common Japanese home-cooking/food knowledge. If the food name is ambiguous, assume the most common everyday interpretation.
Respond with ONLY a JSON object in exactly this shape, no other text:
{"kcal": number, "per": number, "unit": "string"}
"unit" must be exactly the unit given in the request. "per" must be 100 for g/ml, or 1 for a countable unit.
If you cannot reasonably estimate this food, respond with {"kcal": null}.`;

const SYSTEM_PROMPT_PORTION = `You are a calorie estimation assistant for a Japanese diet diary app.
Given a food name and a rough serving size (S = small, M = medium/standard, L = large),
estimate a typical total calorie count for ONE serving of that size.
Use common Japanese home-cooking/food knowledge. If the food name is ambiguous, assume the most common everyday interpretation.
Respond with ONLY a JSON object in exactly this shape, no other text:
{"kcal": number}
If you cannot reasonably estimate this food, respond with {"kcal": null}.`;

function sendJson(res, status, payload) {
  res.status(status).setHeader('Content-Type', 'application/json');
  res.end(JSON.stringify(payload));
}

function fail(res, status, error) {
  sendJson(res, status, { ok: false, error });
}

async function callOpenAi(apiKey, systemPrompt, userText) {
  const openaiResp = await fetch(OPENAI_CHAT_URL, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${apiKey}`
    },
    body: JSON.stringify({
      model: OPENAI_MODEL,
      max_tokens: 100,
      response_format: { type: 'json_object' },
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: userText }
      ]
    })
  });
  if (!openaiResp.ok) {
    let detail = '';
    try { detail = (await openaiResp.json()).error?.message || ''; } catch {}
    const err = new Error(`OpenAI request failed${detail ? ': ' + detail : ''}.`);
    err.status = 502;
    throw err;
  }
  const completion = await openaiResp.json();
  const content = completion?.choices?.[0]?.message?.content;
  if (!content) {
    const err = new Error('OpenAI response did not contain a result.');
    err.status = 502;
    throw err;
  }
  try {
    return JSON.parse(content);
  } catch {
    const err = new Error('Could not parse the estimate as JSON.');
    err.status = 502;
    throw err;
  }
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

  const name = typeof body.name === 'string' ? body.name.trim() : '';
  if (!name) {
    return fail(res, 400, 'name is required and must be a non-empty string.');
  }
  if (name.length > MAX_NAME_CHARS) {
    return fail(res, 400, 'name is too long.');
  }

  const unit = typeof body.unit === 'string' ? body.unit.trim() : '';
  const portion = typeof body.portion === 'string' ? body.portion.trim().toLowerCase() : '';

  let mode, systemPrompt, userText;
  if (unit) {
    mode = 'basis';
    const kind = COUNT_UNITS.has(unit) ? '1' : '100';
    systemPrompt = SYSTEM_PROMPT_BASIS;
    userText = `Food: ${name}\nUnit: ${unit}\nEstimate kcal per ${kind} ${unit}.`;
  } else if (VALID_PORTIONS.has(portion)) {
    mode = 'flat';
    systemPrompt = SYSTEM_PROMPT_PORTION;
    userText = `Food: ${name}\nServing size: ${portion.toUpperCase()}\nEstimate total kcal for one ${portion.toUpperCase()} serving.`;
  } else {
    return fail(res, 400, 'Provide either a non-empty unit, or a portion of "s", "m", or "l".');
  }

  let parsed;
  try {
    parsed = await callOpenAi(apiKey, systemPrompt, userText);
  } catch (e) {
    console.error('estimate-calories OpenAI call failed:', e);
    return fail(res, e.status || 502, e.message || 'Could not reach OpenAI.');
  }

  const kcalNum = Number(parsed.kcal);
  if (parsed.kcal == null || !Number.isFinite(kcalNum) || kcalNum < 0) {
    // AI couldn't produce a usable number — never fabricate one.
    return sendJson(res, 200, { ok: true, mode, kcal: null });
  }

  if (mode === 'basis') {
    const per = COUNT_UNITS.has(unit) ? 1 : 100;
    return sendJson(res, 200, { ok: true, mode, kcal: Math.round(kcalNum), per, unit });
  }
  return sendJson(res, 200, { ok: true, mode, kcal: Math.round(kcalNum) });
};

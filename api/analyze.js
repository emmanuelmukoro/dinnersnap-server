// /api/analyze.js — Vision + Spoonacular + LLM fallback, with tastier LLM prompt and debug logs
export const config = {
  runtime: "edge",
};

const SPOON_KEY = process.env.SPOON_KEY || "";
const OPENAI_API_KEY = process.env.OPENAI_API_KEY || "";
const GCV_KEY = process.env.GCV_KEY || "";

// ===== Helpers =====
const okJson = (data, status = 200) =>
  new Response(JSON.stringify(data), {
    status,
    headers: { "content-type": "application/json" },
  });

const bad = (msg, status = 400) => okJson({ error: msg }, status);

const uniqLower = (arr) => [...new Set(arr.map((s) => String(s).toLowerCase().trim()))];

// strict Levenshtein (small & fast enough for our shortlist)
function lev(a, b) {
  if (a === b) return 0;
  const m = a.length, n = b.length;
  if (!m) return n;
  if (!n) return m;
  let prev = new Array(n + 1);
  let cur = new Array(n + 1);
  for (let j = 0; j <= n; j++) prev[j] = j;
  for (let i = 1; i <= m; i++) {
    cur[0] = i;
    const ca = a.charCodeAt(i - 1);
    for (let j = 1; j <= n; j++) {
      const cb = b.charCodeAt(j - 1);
      const cost = ca === cb ? 0 : 1;
      cur[j] = Math.min(
        cur[j - 1] + 1,
        prev[j] + 1,
        prev[j - 1] + cost
      );
    }
    const t = prev; prev = cur; cur = t;
  }
  return prev[n];
}

// ===== OCR -> tokens (more selective) =====
function tokensFromVisionResponse(r) {
  const ocrRaw = (r.textAnnotations?.[0]?.description || "").toLowerCase();
  const RAW_TOKENS = ocrRaw
    .split(/[^a-z]+/g)
    .map((t) => t.trim())
    .filter((t) => t.length >= 4);

  const FOODISH_HINTS = new Set([
    "ingredients","organic","sauce","soup","canned","tin","dried","fresh","frozen",
    "beans","peas","lentils","broccoli","banana","tomato","onion","garlic","squash",
    "chickpea","chickpeas","feta","rice","pasta","oil","olive","seed","seeds","flour",
    "oats","milk","yogurt","cheese","lemon","pepper","spinach","mushroom","potato",
    "coconut","cream","creamed","coconutcream"
  ]);

  return RAW_TOKENS.filter((t) => FOODISH_HINTS.has(t));
}

// ===== Whitelist + synonyms + strict fuzzy =====
const WHITELIST = [
  "banana","broccoli","chickpeas","tomatoes","onion","garlic","olive oil","pasta",
  "courgette","feta","orzo","rice","egg","spring onion","lemon","coconut cream","coconut milk",
  "butternut squash","squash","carrot","pepper","bell pepper","spinach",
  "potato","mushroom","cheese","yogurt","milk",
  "chicken","chicken breast","beef","pork","fish","salmon","tuna",
  "bread","tortilla","wrap","beans","kidney beans","black beans","lentils",
  "cucumber","lettuce","cabbage","kale","avocado","apple","orange","pear",
  "oats","flour","sugar","butter","salt","pepper"
];

const MAP = {
  bananas: "banana",
  chickpea: "chickpeas",
  garbanzo: "chickpeas",
  "garbanzo bean": "chickpeas",
  "garbanzo beans": "chickpeas",
  zucchini: "courgette",
  courgettes: "courgette",
  tomato: "tomatoes",
  onions: "onion",
  eggs: "egg",
  brockley: "broccoli",
  "coconutcream": "coconut cream",
};

function nearestWhitelistStrict(term) {
  let best = null, bestDist = 2; // allow distance ≤ 1 only
  for (const w of WHITELIST) {
    const d = lev(term, w);
    if (d < bestDist) { bestDist = d; best = w; }
  }
  return bestDist <= 1 ? best : null;
}

function cleanPantry(rawTerms) {
  const out = new Set();
  for (const tRaw of uniqLower(rawTerms)) {
    const t = MAP[tRaw] || tRaw;
    if (WHITELIST.includes(t)) { out.add(t); continue; }
    if (t.length >= 5) {
      const n = nearestWhitelistStrict(t);
      if (n) { out.add(n); continue; }
    }
    // else drop
  }
  return [...out];
}

// ===== Vision =====
async function callVision(imageBase64) {
  // imageBase64 is a full data URL when it reaches here (`data:image/jpeg;base64,...`)
  const url = `https://vision.googleapis.com/v1/images:annotate?key=${GCV_KEY}`;
  const body = {
    requests: [{
      image: { content: imageBase64.split(",")[1] || "" },
      features: [
        { type: "TEXT_DETECTION" },
        { type: "LABEL_DETECTION", maxResults: 50 },
        { type: "OBJECT_LOCALIZATION" }
      ]
    }]
  };
  const res = await fetch(url, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`Vision ${res.status}`);
  const data = await res.json();
  return data.responses?.[0] || {};
}

function tokensFromLabels(r) {
  const labels = r.labelAnnotations || [];
  return labels
    .filter(l => (l.description || "").length >= 3 && (l.score || 0) >= 0.6)
    .map(l => l.description.toLowerCase());
}

function tokensFromObjects(r) {
  const objs = r.localizedObjectAnnotations || [];
  return objs
    .filter(o => (o.name || "").length >= 3 && (o.score || 0) >= 0.6)
    .map(o => o.name.toLowerCase());
}

// ===== Spoonacular =====
async function spoonacularByIngredients(pantry, prefs) {
  if (!SPOON_KEY) return [];

  // prefer savory; avoid dessert keywords
  const exclude = ["dessert","ice cream","cupcake","cookie","brownie","cake","sweet","pudding"];
  const base = `https://api.spoonacular.com/recipes/complexSearch`;
  const params = new URLSearchParams({
    apiKey: SPOON_KEY,
    includeIngredients: pantry.join(","),
    number: "6",
    sort: "max-used-ingredients",
    addRecipeInformation: "true",
    instructionsRequired: "true",
    fillIngredients: "true",
    ranking: "2", // balance popularity vs. used-ingredients
  });

  if (prefs?.time) params.set("maxReadyTime", String(prefs.time));
  if (prefs?.diet && prefs.diet !== "none") params.set("diet", prefs.diet);
  // filter out desserts by type & exclude terms
  params.set("type", "main course");

  const url = `${base}?${params.toString()}`;
  const res = await fetch(url);
  if (!res.ok) return [];
  const data = await res.json();
  const results = Array.isArray(data?.results) ? data.results : [];

  const filtered = results.filter(r => {
    const title = (r.title || "").toLowerCase();
    return !exclude.some(k => title.includes(k));
  });

  // map into your app shape
  return filtered.map((r, idx) => ({
    id: r.id ?? `spoon-${idx}`,
    title: r.title,
    time: r.readyInMinutes ?? 20,
    cost: 2.5,                 // we don’t get cost reliably here
    energy: "hob",
    score: r.healthScore ? Math.round((r.healthScore/100)*100)/100 : undefined,
    badges: [],
    ingredients: (r.extendedIngredients || []).map(ing => ({
      name: (ing.name || "").toLowerCase(),
      have: pantry.includes((ing.name || "").toLowerCase())
    })),
    steps: Array.isArray(r.analyzedInstructions?.[0]?.steps)
      ? r.analyzedInstructions[0].steps.map(s => ({
          id: s.number,
          text: s.step
        }))
      : [],
  }));
}

// ===== LLM fallback (tastier, savory, with seasonings) =====
async function llmComposeRecipe(pantry, prefs) {
  if (!OPENAI_API_KEY) return null;

  const savoryBias = `
You are writing a QUICK, SAVORY, TASTY dinner recipe using the pantry items.
Do NOT suggest desserts unless the pantry is obviously dessert-only.
Assume the user has basic pantry staples and seasonings:
salt, black pepper, oil, butter, soy sauce, vinegar/lemon, garlic, onion, ginger,
paprika, cumin, coriander, chilli flakes, curry powder, oregano, thyme, rosemary,
all-purpose seasoning, and simple fresh herbs if helpful (e.g. chives, parsley).
If the dish would be plain, actively include appropriate seasonings from that list.
Prefer a single-pan or single-pot approach when possible.
If there is coconut cream/milk + legumes (e.g., chickpeas), build a proper curry base
(garlic/onion/ginger + spices + simmer) so it tastes great.
Always target ≤ ${prefs?.time ?? 25} minutes unless impossible.
`;

  const user = `
PANTRY: ${JSON.stringify(pantry)}
DIET: ${prefs?.diet ?? "none"} • SERVINGS: ${prefs?.servings ?? 2} • ENERGY: ${prefs?.energyMode ?? "hob"}
RETURN JSON with:
{
  "id": "llm-1",
  "title": "Name",
  "time": number,
  "cost": 2.5,
  "energy": "hob" | "oven" | "airfryer" | "microwave",
  "ingredients": [{"name": "string", "have": true|false}, ...],
  "steps": [{"id": 1, "text": "short action"}, ...],
  "badges": ["15-min","one-pan", ...]
}
- Use pantry items as "have": true. Add reasonable extras (spices, aromatics) as "have": false.
- Keep steps compact (<= 8), but include the seasoning/flavour steps so it’s NOT bland.
`;

  const body = {
    model: "gpt-4o-mini",
    temperature: 0.6,
    messages: [
      { role: "system", content: savoryBias.trim() },
      { role: "user", content: user.trim() }
    ],
    response_format: { type: "json_object" }
  };

  const res = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: {
      "content-type": "application/json",
      Authorization: `Bearer ${OPENAI_API_KEY}`,
    },
    body: JSON.stringify(body),
  });

  if (!res.ok) return null;
  const data = await res.json();
  let txt = data?.choices?.[0]?.message?.content || "{}";
  try {
    const r = JSON.parse(txt);
    // normalize minimal fields for the app
    r.id = r.id || "llm-1";
    r.energy = r.energy || "hob";
    if (!Array.isArray(r.ingredients)) r.ingredients = [];
    if (!Array.isArray(r.steps)) r.steps = [];
    return r;
  } catch {
    return null;
  }
}

// ===== Main handler =====
export default async function handler(req) {
  if (req.method !== "POST") return bad("POST only", 405);

  let body = {};
  try { body = await req.json(); } catch {}
  const { imageBase64, pantryOverride, prefs, debug } = body || {};

  if (!imageBase64 && !pantryOverride) {
    return bad("imageBase64 required (or pantryOverride)");
  }

  const debugInfo = {
    source: imageBase64 ? "vision" : "pantryOverride",
    pantryFrom: [],
    cleanedPantry: [],
    spoonacularCount: 0,
    usedLLM: false
  };

  // ---- Build pantry terms ----
  let pantryTerms = [];
  if (imageBase64) {
    const r = await callVision(imageBase64);
    const tOcr = tokensFromVisionResponse(r);
    const tLbl = tokensFromLabels(r);
    const tObj = tokensFromObjects(r);
    debugInfo.pantryFrom = { ocr: tOcr, labels: tLbl, objects: tObj };
    pantryTerms = cleanPantry([...tOcr, ...tLbl, ...tObj]);
  } else {
    pantryTerms = cleanPantry(
      Array.isArray(pantryOverride)
        ? pantryOverride.map(x => (typeof x === "string" ? x : x?.name || ""))
        : []
    );
  }

  // Ensure we include directly typed “good” items if simple mistakes occur
  // (e.g., "chickpeas", "coconut cream", "banana", "pasta")
  debugInfo.cleanedPantry = pantryTerms;

  // ---- 1) Try Spoonacular
  let recipes = [];
  try {
    recipes = await spoonacularByIngredients(pantryTerms, prefs);
  } catch (e) {
    console.log("Spoonacular error", e?.message || e);
  }
  debugInfo.spoonacularCount = recipes.length;

  // ---- 2) LLM fallback if nothing decent
  if (!recipes.length) {
    const llm = await llmComposeRecipe(pantryTerms, prefs);
    if (llm) {
      debugInfo.usedLLM = true;
      recipes = [llm];
    }
  }

  console.log("analyze debug:", debugInfo);

  return okJson({
    pantry: pantryTerms,
    recipes,
    debug: debug ? debugInfo : undefined
  });
}

// api/analyze.js — Vercel serverless: Vision OCR → pantry → recipes (Spoonacular + LLM fallback)

export default async function handler(req, res) {
  try {
    if (req.method !== "POST") {
      return res.status(405).json({ error: "Use POST" });
    }

    const { imageBase64, pantryOverride, prefs = {} } = req.body || {};

    // Step 1: build pantry
    let pantry = [];
    if (Array.isArray(pantryOverride) && pantryOverride.length) {
      pantry = cleanPantry(pantryOverride.map(String).map((s) => s.toLowerCase()));
    } else if (imageBase64) {
      const vision = await callVision(imageBase64);
      pantry = cleanPantry(vision);
    } else {
      return res.status(400).json({ error: "imageBase64 required (or pantryOverride)" });
    }

    // Step 2: fetch recipes (Spoonacular, constrained to savory mains)
    let recipes = [];
    try {
      recipes = await fetchSpoonacular(pantry, prefs);
    } catch (e) {
      console.error("Spoonacular error:", e);
    }

    // Step 3: LLM fallback to ensure we always return something savory
    if (!recipes.length) {
      try {
        recipes = await llmFallback(pantry, prefs);
      } catch (e) {
        console.error("LLM fallback error:", e);
      }
    }

    return res.status(200).json({ pantry, recipes });
  } catch (e) {
    console.error(e);
    return res.status(500).json({ error: "Server error" });
  }
}

/* ---------------- Vision (OCR) ---------------- */

async function callVision(imageBase64) {
  if (!process.env.GCV_KEY) return [];
  const url = `https://vision.googleapis.com/v1/images:annotate?key=${process.env.GCV_KEY}`;
  const payload = {
    requests: [
      {
        image: { content: imageBase64.split(",").pop() },
        features: [
          { type: "TEXT_DETECTION" },
          { type: "LABEL_DETECTION", maxResults: 10 },
          { type: "LOGO_DETECTION" },
          { type: "OBJECT_LOCALIZATION" },
        ],
      },
    ],
  };
  const r = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
  const j = await r.json();
  const r0 = j?.responses?.[0] || {};

  // ---- OCR tokens (aggressive cleanup) ----
  const ocrRaw = (r0.textAnnotations?.[0]?.description || "").toLowerCase();
  const RAW_TOKENS = ocrRaw
    .split(/[^a-z]+/g)
    .map((t) => t.trim())
    .filter((t) => t.length >= 4); // ignore tiny tokens

  const FOODISH_HINTS = new Set([
    "ingredients","organic","sauce","soup","canned","tin","dried","fresh","frozen",
    "beans","peas","lentils","broccoli","banana","tomato","onion","garlic","squash",
    "chickpea","chickpeas","feta","rice","pasta","oil","olive","seed","seeds","flour",
    "oats","milk","yogurt","cheese","lemon","pepper","spinach","mushroom","potato",
    "coconut","cream","coconutcream","coconutmilk"
  ]);
  const ocrTokens = RAW_TOKENS.filter((t) => FOODISH_HINTS.has(t));

  // ---- labels/logos/objects as words ----
  const labels = (r0.labelAnnotations || []).map((x) => (x.description || "").toLowerCase());
  const logos = (r0.logoAnnotations || []).map((x) => (x.description || "").toLowerCase());
  const objects = (r0.localizedObjectAnnotations || []).map((x) => (x.name || "").toLowerCase());

  // some basic object → ingredient mapping
  const mappedObjects = objects.map((o) => (o.includes("banana") ? "banana" : o.includes("broccoli") ? "broccoli" : o));

  // combine raw terms
  const rawTerms = [...ocrTokens, ...labels, ...logos, ...mappedObjects];
  return rawTerms;
}

/* ---------------- Pantry cleaning ---------------- */

const WHITELIST = [
  "banana","broccoli","chickpeas","tomatoes","onion","garlic","olive oil","pasta",
  "courgette","feta","orzo","rice","egg","spring onion","lemon",
  "butternut squash","squash","carrot","pepper","bell pepper","spinach",
  "potato","mushroom","cheese","yogurt","milk",
  "coconut milk","coconut cream",
  // proteins
  "chicken","chicken breast","beef","pork","fish","salmon","tuna",
  // tins & pulses
  "beans","kidney beans","black beans","lentils",
  // basics
  "bread","tortilla","wrap","oats","flour","sugar","butter","salt","pepper"
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
  "coconut creme": "coconut cream",
  "coconut milk": "coconut milk",
};

function uniqLower(arr) {
  return Array.from(new Set(arr.map((x) => String(x || "").toLowerCase())));
}

function cleanPantry(rawTerms) {
  const out = new Set();
  for (const tRaw of uniqLower(rawTerms)) {
    const t = MAP[tRaw] || tRaw;

    // exact whitelist
    if (WHITELIST.includes(t)) { out.add(t); continue; }

    // strict fuzzy (len>=5 && distance<=1)
    if (t.length >= 5) {
      const nearest = nearestWhitelistStrict(t);
      if (nearest) { out.add(nearest); continue; }
    }
    // else drop it
  }
  return [...out];
}

function nearestWhitelistStrict(term) {
  let best = null, bestDist = 2;
  for (const w of WHITELIST) {
    const d = lev(term, w);
    if (d < bestDist) { bestDist = d; best = w; }
  }
  return bestDist <= 1 ? best : null;
}

// Levenshtein
function lev(a, b) {
  const m = a.length, n = b.length;
  const dp = Array.from({ length: m + 1 }, () => new Array(n + 1).fill(0));
  for (let i = 0; i <= m; i++) dp[i][0] = i;
  for (let j = 0; j <= n; j++) dp[0][j] = j;
  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      dp[i][j] = Math.min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost);
    }
  }
  return dp[m][n];
}

/* ---------------- Spoonacular (main courses only) ---------------- */

async function fetchSpoonacular(pantry, prefs) {
  if (!process.env.SPOON_KEY) return [];
  const include = encodeURIComponent(pantry.join(","));
  const params =
    `includeIngredients=${include}` +
    `&type=main%20course` +
    `&ignorePantry=true` +
    `&instructionsRequired=true` +
    `&addRecipeInformation=true` +
    `&maxReadyTime=${Math.max(10, Math.min(90, Number(prefs?.time || 30)))}` +
    `&number=8&sort=min-missing-ingredients&sortDirection=asc`;
  const url = `https://api.spoonacular.com/recipes/complexSearch?${params}&apiKey=${process.env.SPOON_KEY}`;

  const r = await fetch(url);
  const j = await r.json();
  if (!j?.results) return [];

  const BAD = /dessert|sweet|ice\s*cream|jam|smoothie|shake|cookie|brownie|cake|cupcake|pancake|muffin|fudge/i;

  const items = j.results
    .filter((x) => !BAD.test(x.title))
    .map((x) => {
      const used = (x.usedIngredients || []).map((i) => i.name?.toLowerCase?.() || "");
      const missed = (x.missedIngredients || []).map((i) => i.name?.toLowerCase?.() || "");
      const all = [...used, ...missed];
      return {
        id: `spn_${x.id}`,
        title: x.title,
        time: x.readyInMinutes ?? prefs?.time ?? 20,
        cost: 2.5,
        energy: "hob",
        ingredients: all.map((n) => ({ name: n, have: pantry.includes(n) })),
        steps: [],
        badges: ["spoonacular"],
      };
    });

  return items;
}

/* ---------------- LLM fallback (savory only) ---------------- */

async function llmFallback(pantry, prefs) {
  if (!process.env.OPENAI_API_KEY) return [];
  const system = `You are a recipe generator. Output ONLY JSON.
Rules:
- Create 1–3 quick, savory DINNER recipes (not dessert, not smoothie, not jam, not ice cream).
- Use the given pantry as primary ingredients.
- Ready in <= ${prefs?.time || 25} minutes.
- Prefer single-pan methods (hob) when possible.
- JSON shape:
{"recipes":[{"id":"llm_1","title":"...","time":15,"cost":2.2,"energy":"hob",
"ingredients":[{"name":"...","have":true}],"steps":[{"id":"s1","text":"..."}],"badges":["fallback"]}]}
`;

  const user = JSON.stringify({ pantry, prefs });

  const r = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${process.env.OPENAI_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: "gpt-4o-mini",
      messages: [
        { role: "system", content: system },
        { role: "user", content: user },
      ],
      temperature: 0.3,
    }),
  });
  const j = await r.json();
  const text = j?.choices?.[0]?.message?.content || "{}";

  let parsed = {};
  try { parsed = JSON.parse(text); } catch {}
  const list = Array.isArray(parsed?.recipes) ? parsed.recipes : [];

  // Minimal normalization
  return list.map((x, idx) => ({
    id: x.id || `llm_${idx + 1}`,
    title: x.title || "Quick dinner",
    time: Number(x.time || prefs?.time || 20),
    cost: Number(x.cost || 2.0),
    energy: x.energy || "hob",
    ingredients: Array.isArray(x.ingredients)
      ? x.ingredients.map((i) => ({
          name: (i.name || "").toLowerCase(),
          have: Boolean(i.have),
        }))
      : pantry.map((n) => ({ name: n, have: true })),
    steps: Array.isArray(x.steps)
      ? x.steps.map((s, i) => ({ id: s.id || `s${i + 1}`, text: s.text || "" }))
      : [{ id: "s1", text: "Cook everything." }],
    badges: Array.isArray(x.badges) ? x.badges : ["fallback"],
  }));
}

// api/analyze.js — Vision ➜ cleaned pantry (labels + OCR + strict fuzzy) ➜ Spoonacular ➜ (fallback) OpenAI
// Set env vars in Vercel: GCV_KEY, SPOON_KEY, OPENAI_API_KEY

export default async function handler(req, res) {
  try {
    if (req.method !== "POST") return res.status(405).json({ error: "Use POST" });

    const { imageBase64, pantryOverride, prefs = {} } = req.body || {};

    let pantry = [];

    if (Array.isArray(pantryOverride) && pantryOverride.length) {
      // Client submitted current pantry (e.g., after manual fixes)
      pantry = cleanPantry(pantryOverride.map(String));
    } else {
      if (!imageBase64) return res.status(400).json({ error: "imageBase64 required (or pantryOverride)" });
      const visionTerms = await callVision(imageBase64, process.env.GCV_KEY);
      pantry = cleanPantry(visionTerms);
    }

    // Spoonacular first
    let recipes = [];
    if (process.env.SPOON_KEY && pantry.length) {
      recipes = await findSpoonacularRecipes(pantry, prefs, process.env.SPOON_KEY);
    }

    // Fallback to LLM if nothing found
    if ((!recipes || recipes.length === 0) && process.env.OPENAI_API_KEY && pantry.length) {
      const r = await llmRecipe(pantry, prefs, process.env.OPENAI_API_KEY);
      if (r) recipes = [r];
    }

    return res.status(200).json({ pantry, recipes: recipes || [] });
  } catch (err) {
    console.error("analyze error:", err);
    return res.status(500).json({ error: err?.message || "server error" });
  }
}

/* ------------------- Google Vision ------------------- */
async function callVision(imageBase64, apiKey) {
  if (!apiKey) return [];
  const content = imageBase64.includes("base64,")
    ? imageBase64.split("base64,")[1]
    : imageBase64;

  const body = {
    requests: [{
      image: { content },
      features: [
        { type: "LABEL_DETECTION", maxResults: 50 },
        { type: "OBJECT_LOCALIZATION", maxResults: 50 },
        { type: "TEXT_DETECTION", maxResults: 1 }, // OCR
      ],
    }],
  };

  const resp = await fetch(
    `https://vision.googleapis.com/v1/images:annotate?key=${apiKey}`,
    { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }
  );
  if (!resp.ok) throw new Error(`Vision ${resp.status}: ${await resp.text().catch(()=> "")}`);

  const json = await resp.json();
  const r = json?.responses?.[0] || {};

  const labels = [];
  (r.labelAnnotations || []).forEach((x) => labels.push(x.description));
  (r.localizedObjectAnnotations || []).forEach((x) => labels.push(x.name));

  // ---------- OCR tokens (aggressive cleanup) ----------
  const ocrRaw = (r.textAnnotations?.[0]?.description || "").toLowerCase();

  // split, drop very short tokens, keep plausible “foodish” words only
  const RAW_TOKENS = ocrRaw
    .split(/[^a-z]+/g)
    .map((t) => t.trim())
    .filter((t) => t.length >= 4); // ignore tiny tokens like "so", "uk", etc.

  const FOODISH_HINTS = new Set([
    "ingredients","organic","sauce","soup","canned","tin","dried","fresh","frozen",
    "beans","peas","lentils","broccoli","banana","tomato","onion","garlic","squash",
    "chickpea","chickpeas","feta","rice","pasta","olive","seed","seeds","flour",
    "oats","milk","yogurt","cheese","lemon","pepper","spinach","mushroom","potato",
  ]);

  // keep token if it looks like food-ish; whitelist matching happens later
  const ocrTokens = RAW_TOKENS.filter(t => FOODISH_HINTS.has(t));

  return uniqLower([...labels, ...ocrTokens]);
}

/* ------------------- Pantry cleaning (whitelist + synonyms + strict fuzzy) ------------------- */
function uniqLower(arr) { const s = new Set(); arr.forEach(a => s.add(String(a||"").trim().toLowerCase())); return [...s]; }

const WHITELIST = [
  "banana","broccoli","chickpeas","tomatoes","onion","garlic","olive oil","pasta",
  "courgette","feta","orzo","rice","egg","spring onion","lemon",
  "butternut squash","squash","carrot","pepper","bell pepper","spinach",
  "potato","mushroom","cheese","yogurt","milk",
  "chicken","chicken breast","beef","pork","fish","salmon","tuna",
  "bread","tortilla","wrap","beans","kidney beans","black beans","lentils",
  "cucumber","lettuce","cabbage","kale","avocado","apple","orange","pear",
  "oats","flour","sugar","butter","salt","pepper" // NOTE: generic "oil" removed
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
  brockley: "broccoli", // observed typo
};

function cleanPantry(rawTerms) {
  const out = new Set();

  for (const tRaw of uniqLower(rawTerms)) {
    const t = MAP[tRaw] || tRaw;

    // exact whitelist hit
    if (WHITELIST.includes(t)) { out.add(t); continue; }

    // STRICT fuzzy: only for length >=5 and edit distance <=1
    if (t.length >= 5) {
      const nearest = nearestWhitelistStrict(t);
      if (nearest) { out.add(nearest); continue; }
    }
    // else drop it
  }
  return [...out];
}

function nearestWhitelistStrict(term) {
  let best = null, bestDist = 2; // accept only <=1
  for (const w of WHITELIST) {
    const d = lev(term, w);
    if (d < bestDist) { bestDist = d; best = w; }
  }
  return bestDist <= 1 ? best : null;
}

// Levenshtein
function lev(a, b) {
  if (a === b) return 0;
  const al = a.length, bl = b.length;
  if (al === 0) return bl;
  if (bl === 0) return al;
  const dp = Array.from({ length: al + 1 }, (_, i) => [i]);
  for (let j = 0; j <= bl; j++) dp[0][j] = j;
  for (let i = 1; i <= al; i++) {
    for (let j = 1; j <= bl; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      dp[i][j] = Math.min(
        dp[i - 1][j] + 1,
        dp[i][j - 1] + 1,
        dp[i - 1][j - 1] + cost
      );
    }
  }
  return dp[al][bl];
}

/* ------------------- Spoonacular ------------------- */
async function findSpoonacularRecipes(pantry, prefs, key) {
  const include = pantry.slice(0, 6).join(",");
  const maxReadyTime = clamp(prefs?.time, 10, 120) ?? 30;
  const diet = toSpoonDiet(prefs?.diet);

  const url = new URL("https://api.spoonacular.com/recipes/complexSearch");
  url.searchParams.set("apiKey", key);
  url.searchParams.set("includeIngredients", include);
  url.searchParams.set("instructionsRequired", "true");
  url.searchParams.set("addRecipeInformation", "true");
  url.searchParams.set("fillIngredients", "true");
  url.searchParams.set("number", "6");
  url.searchParams.set("maxReadyTime", String(maxReadyTime));
  url.searchParams.set("sort", "min-missing-ingredients");
  if (diet) url.searchParams.set("diet", diet);

  const resp = await fetch(url.toString());
  if (!resp.ok) { console.warn("Spoonacular error:", resp.status); return []; }
  const data = await resp.json();
  return (data?.results || []).map(spoonToRecipe);
}

function spoonToRecipe(r) {
  const used = (r.usedIngredients || []).length;
  const total = used + (r.missedIngredients || []).length || 1;
  return {
    id: `sp_${r.id}`,
    title: r.title,
    time: r.readyInMinutes ?? 20,
    cost: 2.5,
    energy: "hob",
    steps: (r.analyzedInstructions?.[0]?.steps || []).map((s, i) => ({ id: `s${i+1}`, text: s.step })),
    ingredients: (r.missedIngredients || []).concat(r.usedIngredients || [])
      .map(ing => ({ name: (ing.name||"").toLowerCase(), have: !!ing.used })),
    score: Math.round((used / total) * 100) / 100,
    badges: ["spoonacular"],
  };
}

function toSpoonDiet(diet) {
  switch ((diet||"").toLowerCase()) {
    case "vegetarian": return "vegetarian";
    case "vegan": return "vegan";
    case "pescatarian": return "pescetarian";
    default: return "";
  }
}

function clamp(n, min, max) { const x = Number(n); if (Number.isNaN(x)) return min; return Math.max(min, Math.min(max, x)); }

/* ------------------- OpenAI fallback ------------------- */
async function llmRecipe(pantry, prefs, openaiKey) {
  const prompt = `
You're a concise cooking assistant. Create ONE quick dinner recipe using ONLY these items when possible:
${pantry.join(", ")}.
Constraints: max ${(prefs?.time ?? 20)} minutes; simple steps; UK-friendly.
Return strict JSON:
{"id":"llm_1","title":"...","time":15,"cost":2.0,"energy":"hob","ingredients":[{"name":"...","have":true}],"steps":[{"id":"s1","text":"..."}]}
No extra commentary.`;

  const resp = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${openaiKey}` },
    body: JSON.stringify({
      model: "gpt-4o-mini",
      messages: [{ role: "user", content: prompt }],
      temperature: 0.2,
      response_format: { type: "json_object" },
    }),
  });

  if (!resp.ok) { console.warn("OpenAI error:", resp.status); return null; }

  const json = await resp.json();
  const content = json?.choices?.[0]?.message?.content || "{}";
  try {
    const r = JSON.parse(content);
    return {
      id: r.id || "llm_1",
      title: r.title || "Quick dinner",
      time: r.time ?? 15,
      cost: r.cost ?? 2.0,
      energy: r.energy || "hob",
      ingredients: Array.isArray(r.ingredients) ? r.ingredients : [],
      steps: Array.isArray(r.steps) ? r.steps : [],
      badges: ["ai"],
      score: 0.8,
    };
  } catch { return null; }
}
